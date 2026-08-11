"""Configurable classical and hybrid QuFeX galaxy classifiers."""

from __future__ import annotations

from typing import Any

try:
    import pennylane as qml
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on machine-specific PyTorch
    raise ImportError(
        "qmla.model requires PyTorch. Install the CUDA/CPU build appropriate for this machine."
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
    """Published 8(1), 4(1), or 4(2) QuFeX over a [B, 16, 2, 2] bottleneck."""

    def __init__(
        self,
        *,
        backend: str = "default.qubit",
        diff_method: str = "backprop",
        shots: int = 0,
        qubits: int = 8,
        filters: int = 1,
        input_angle_scale: float = torch.pi,
    ) -> None:
        super().__init__()
        if (qubits, filters) not in {(8, 1), (4, 1), (4, 2)}:
            raise ValueError("QuFeXLayer supports qubits/filters 8/1, 4/1, or 4/2")
        self.qubits = qubits
        self.filters = filters
        self.output_channels = 32 if filters == 2 else 16
        self.input_angle_scale = float(input_angle_scale)
        self.theta = nn.Parameter(torch.empty(filters, 4, dtype=torch.float32))
        nn.init.uniform_(self.theta, -0.1, 0.1)
        self._circuits = tuple(
            self._make_circuit(
                backend=backend,
                diff_method=diff_method,
                shots=shots,
                second_filter=index == 1,
            )
            for index in range(filters)
        )

    def _make_circuit(
        self,
        *,
        backend: str,
        diff_method: str,
        shots: int,
        second_filter: bool,
    ) -> Any:
        device = qml.device(backend, wires=self.qubits, shots=None if shots == 0 else shots)

        @qml.qnode(device, interface="torch", diff_method=diff_method)
        def circuit(inputs: torch.Tensor, theta: torch.Tensor) -> Any:
            # Keep the simulator's initial state on the Torch input device.
            initial_state = torch.zeros(
                2**self.qubits,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            initial_state[0] = 1.0
            qml.StatePrep(initial_state, wires=range(self.qubits))

            if second_filter:
                for wire in range(self.qubits):
                    qml.Hadamard(wires=wire)
                qml.AngleEmbedding(
                    inputs * self.input_angle_scale,
                    wires=range(self.qubits),
                    rotation="Z",
                )
                first_pairs = ((0, 1), (2, 3), (1, 2))
                for first, second in first_pairs:
                    qml.RY(theta[0], wires=first)
                    qml.RX(theta[1], wires=second)
                    qml.CNOT(wires=(first, second))
                for first, second in ((0, 1), (2, 3)):
                    qml.CZ(wires=(first, second))
                qml.RZ(theta[2], wires=0)
                qml.RX(theta[3], wires=2)
                qml.CNOT(wires=(0, 2))
                qml.CZ(wires=(0, 2))
            else:
                qml.AngleEmbedding(
                    inputs * self.input_angle_scale,
                    wires=range(self.qubits),
                    rotation="Y",
                )
                first_pairs = (
                    ((0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6))
                    if self.qubits == 8
                    else ((0, 1), (2, 3), (1, 2))
                )
                for first, second in first_pairs:
                    qml.RX(theta[0], wires=first)
                    qml.RZ(theta[1], wires=second)
                    qml.CNOT(wires=(first, second))
                pooling_pairs = (
                    ((0, 1), (2, 3), (4, 5), (6, 7))
                    if self.qubits == 8
                    else ((0, 1), (2, 3))
                )
                for first, second in pooling_pairs:
                    qml.CZ(wires=(first, second))
                second_pairs = ((0, 2), (4, 6), (2, 4)) if self.qubits == 8 else ((0, 2),)
                for first, second in second_pairs:
                    qml.RX(theta[2], wires=first)
                    qml.RY(theta[3], wires=second)
                    qml.CNOT(wires=(first, second))
                final_pairs = ((0, 2), (4, 6)) if self.qubits == 8 else ((0, 2),)
                for first, second in final_pairs:
                    qml.CZ(wires=(first, second))

            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(self.qubits))

        return circuit

    @property
    def quantum_parameter_count(self) -> int:
        return self.theta.numel()

    @staticmethod
    def _stack_measurements(outputs: Any, inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)):
            outputs = torch.stack(tuple(outputs), dim=-1)
        return outputs.to(dtype=inputs.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (16, 2, 2):
            raise ValueError(f"QuFeXLayer expects [batch, 16, 2, 2], received {tuple(inputs.shape)}")
        batch = inputs.shape[0]
        if self.qubits == 8:
            # tf.split(..., axis=-1) into adjacent channel pairs + Keras Flatten.
            grouped = inputs.reshape(batch, 8, 2, 2, 2).permute(0, 1, 3, 4, 2)
            circuit_inputs = grouped.reshape(batch * 8, 8)
            outputs = self._stack_measurements(
                self._circuits[0](circuit_inputs, self.theta[0]),
                inputs,
            )
            # Gather even/odd measurements into the two output maps per group.
            maps = outputs.reshape(batch, 8, 2, 2, 2).permute(0, 1, 4, 2, 3)
            return maps.reshape(batch, 16, 2, 2)

        # Four-qubit notebooks apply the same filter(s) to each individual map.
        circuit_inputs = inputs.reshape(batch * 16, 4)
        filter_maps = []
        for index, circuit in enumerate(self._circuits):
            outputs = self._stack_measurements(circuit(circuit_inputs, self.theta[index]), inputs)
            filter_maps.append(outputs.reshape(batch, 16, 2, 2))
        if self.filters == 1:
            return filter_maps[0]
        # Match per-input-map concatenation: c0/f0, c0/f1, c1/f0, c1/f1, ...
        return torch.stack(filter_maps, dim=2).reshape(batch, 32, 2, 2)


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
        bottleneck: list[nn.Module] = []
        if current_channels != config.model.compression_channels:
            bottleneck.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        config.model.compression_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(config.model.compression_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        # This is an identity for the source-faithful 64 -> 2 default path,
        # while keeping custom classical encoders shape-safe.
        bottleneck.append(nn.AdaptiveAvgPool2d((config.model.quantum_spatial_size,) * 2))
        self.compression = nn.Sequential(*bottleneck)

        self.qufex: QuFeXLayer | None = None
        if self.mode == "qufex":
            self.qufex = QuFeXLayer(
                backend=config.quantum.backend,
                diff_method=config.quantum.diff_method,
                shots=config.quantum.shots,
                qubits=config.quantum.qubits,
                filters=config.quantum.filters,
                input_angle_scale=config.quantum.input_angle_scale,
            )

        post_layers: list[nn.Module] = []
        current_channels = (
            config.model.compression_channels
            if self.qufex is None
            else self.qufex.output_channels
        )
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
            residual = (
                torch.cat((features, features), dim=1)
                if self.qufex.filters == 2
                else features
            )
            features = residual + quantum_features.to(dtype=features.dtype)
        features = self.post_quantum(features)
        features = self.global_pool(features).flatten(1)
        return self.classifier(features)
