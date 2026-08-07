"""Configurable classical and hybrid QuFeX galaxy classifiers."""

from __future__ import annotations

from typing import Any

try:
    import pennylane as qml
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on machine-specific PyTorch
    raise ImportError(
        "qmla.model requires PyTorch. Install the CUDA/CPU build appropriate for this machine; "
        "PyTorch is intentionally not declared as a project dependency."
    ) from exc

from qmla.config import AppConfig


class ConvBlock(nn.Module):
    """Repeated Conv-BatchNorm-ReLU operations with optional max pooling."""

    def __init__(self, in_channels: int, out_channels: int, repetitions: int, *, pool: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for _ in range(repetitions):
            layers.extend(
                [
                    nn.Conv2d(current, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current = out_channels
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class QuFeXLayer(nn.Module):
    """Eight-qubit QuFeX v1 circuit operating on four 2x2x2 channel groups.

    Circuit and output ordering follow Jain and Kalev, arXiv:2501.13165v1,
    and the authors' Qu-Net reference implementation.
    """

    def __init__(
        self,
        *,
        backend: str = "default.qubit",
        diff_method: str = "backprop",
        shots: int = 0,
        input_angle_scale: float = torch.pi,
    ) -> None:
        super().__init__()
        self.qubits = 8
        self.input_angle_scale = float(input_angle_scale)
        self.theta = nn.Parameter(torch.empty(4, dtype=torch.float32))
        nn.init.uniform_(self.theta, -0.1, 0.1)
        device = qml.device(backend, wires=self.qubits, shots=None if shots == 0 else shots)

        @qml.qnode(device, interface="torch", diff_method=diff_method)
        def circuit(inputs: torch.Tensor, theta: torch.Tensor) -> Any:
            qml.AngleEmbedding(
                inputs * self.input_angle_scale,
                wires=range(self.qubits),
                rotation="Y",
            )

            # U1: nearest-neighbour, translationally shared convolution gates.
            for first, second in ((0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6)):
                qml.RX(theta[0], wires=first)
                qml.RZ(theta[1], wires=second)
                qml.CNOT(wires=(first, second))

            # V1: pooling gates; QuFeX retains rather than discards control qubits.
            for first, second in ((0, 1), (2, 3), (4, 5), (6, 7)):
                qml.CZ(wires=(first, second))

            # U2 and V2 operate at the next hierarchical scale.
            for first, second in ((0, 2), (4, 6), (2, 4)):
                qml.RX(theta[2], wires=first)
                qml.RY(theta[3], wires=second)
                qml.CNOT(wires=(first, second))
            for first, second in ((0, 2), (4, 6)):
                qml.CZ(wires=(first, second))

            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(self.qubits))

        self._circuit = circuit

    @property
    def quantum_parameter_count(self) -> int:
        return self.theta.numel()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (8, 2, 2):
            raise ValueError(f"QuFeXLayer expects [batch, 8, 2, 2], received {tuple(inputs.shape)}")
        batch = inputs.shape[0]

        # NCHW -> four groups of NHWC 2x2x2 values. Flattening in this order
        # reproduces TensorFlow/Keras Flatten from the cited reference code.
        grouped = inputs.reshape(batch, 4, 2, 2, 2).permute(0, 1, 3, 4, 2)
        circuit_inputs = grouped.reshape(batch * 4, 8)
        outputs = self._circuit(circuit_inputs, self.theta)
        if isinstance(outputs, (tuple, list)):
            outputs = torch.stack(tuple(outputs), dim=-1)
        elif outputs.ndim == 2 and outputs.shape[0] == 8 and outputs.shape[1] != 8:
            outputs = outputs.transpose(0, 1)
        outputs = outputs.to(dtype=inputs.dtype)

        # Interleaved even/odd qubit outputs become the two channels in each map.
        maps = outputs.reshape(batch, 4, 2, 2, 2).permute(0, 1, 4, 2, 3)
        return maps.reshape(batch, 8, 2, 2)


class GalaxyClassifier(nn.Module):
    """CNN encoder -> optional QuFeX residual -> CNN/MLP classifier."""

    def __init__(self, config: AppConfig, *, mode: str | None = None) -> None:
        super().__init__()
        self.mode = mode or config.model.mode
        if self.mode not in {"qufex", "classical"}:
            raise ValueError("mode must be 'qufex' or 'classical'")

        encoder: list[nn.Module] = []
        current_channels = 3
        for channels in config.model.encoder_channels:
            encoder.append(
                ConvBlock(
                    current_channels,
                    channels,
                    config.model.convolutions_per_block,
                    pool=True,
                )
            )
            current_channels = channels
        self.encoder = nn.Sequential(*encoder)
        self.compression = nn.Sequential(
            nn.Conv2d(current_channels, config.model.compression_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(config.model.compression_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((config.model.quantum_spatial_size,) * 2),
        )

        self.qufex: QuFeXLayer | None = None
        if self.mode == "qufex":
            self.qufex = QuFeXLayer(
                backend=config.quantum.backend,
                diff_method=config.quantum.diff_method,
                shots=config.quantum.shots,
                input_angle_scale=config.quantum.input_angle_scale,
            )

        post_layers: list[nn.Module] = []
        current_channels = config.model.compression_channels
        for channels in config.model.post_quantum_channels:
            post_layers.append(ConvBlock(current_channels, channels, 1, pool=False))
            current_channels = channels
        self.post_quantum = nn.Sequential(*post_layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        head: list[nn.Module] = []
        current_features = current_channels
        for hidden in config.model.classifier_hidden_neurons:
            head.extend([nn.Linear(current_features, hidden), nn.ReLU(inplace=True), nn.Dropout(config.model.dropout)])
            current_features = hidden
        head.append(nn.Linear(current_features, len(config.evaluation.class_names)))
        self.classifier = nn.Sequential(*head)

    @property
    def quantum_parameter_count(self) -> int:
        return 0 if self.qufex is None else self.qufex.quantum_parameter_count

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.compression(self.encoder(inputs))
        if self.qufex is not None:
            # Quantum simulation remains float32 even when surrounding CNNs use AMP.
            with torch.autocast(device_type=features.device.type, enabled=False):
                quantum_features = self.qufex(features.float())
            features = features + quantum_features.to(dtype=features.dtype)
        features = self.post_quantum(features)
        features = self.global_pool(features).flatten(1)
        return self.classifier(features)
