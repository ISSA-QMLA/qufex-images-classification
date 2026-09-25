"""Configurable classical and hybrid QuFeX galaxy classifiers."""

from __future__ import annotations

from typing import Any
from dataclasses import asdict

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on machine-specific PyTorch
    raise ImportError(
        "qmla.model requires PyTorch. Install the CUDA/CPU build appropriate for this machine."
    ) from exc

from qmla.config import AppConfig, ReplacementConfig


def activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
            "tanh": nn.Tanh, "identity": nn.Identity}[name]()


def normalization(name: str, channels: int) -> nn.Module:
    if name == "batch":
        return nn.BatchNorm2d(channels)
    if name == "group":
        return nn.GroupNorm(1, channels)
    return nn.Identity()


class ConvBlock(nn.Module):
    """Repeated Conv-BatchNorm-ReLU operations with optional max pooling."""

    def __init__(self, in_channels: int, out_channels: int, repetitions: int, *,
                 pool: str = "none", pool_size: int = 2, kernel_size: int = 3,
                 norm: str = "batch", act: str = "relu") -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for _ in range(repetitions):
            layers.extend(
                [
                    nn.Conv2d(current, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=norm == "none"),
                    normalization(norm, out_channels),
                    activation(act),
                ]
            )
            current = out_channels
        if pool != "none":
            layers.append((nn.MaxPool2d if pool == "max" else nn.AvgPool2d)(pool_size))
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
        import pennylane as qml

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

    def apply_filter(self, inputs: torch.Tensor, index: int = 0) -> torch.Tensor:
        """Execute one shared filter on a batch of already encoded input vectors."""
        return self._stack_measurements(self._circuits[index](inputs, self.theta[index]), inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (16, 2, 2):
            raise ValueError(f"QuFeXLayer expects [batch, 16, 2, 2], received {tuple(inputs.shape)}")
        batch = inputs.shape[0]
        if self.qubits == 8:
            # tf.split(..., axis=-1) into adjacent channel pairs + Keras Flatten.
            grouped = inputs.reshape(batch, 8, 2, 2, 2).permute(0, 1, 3, 4, 2)
            circuit_inputs = grouped.reshape(batch * 8, 8)
            outputs = self.apply_filter(circuit_inputs)
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


class CNNReplacement(nn.Module):
    """Independent filters shared over exactly the same groups as QuFeX."""

    def __init__(self, config: ReplacementConfig, qubits: int, filters: int):
        super().__init__()
        self.group_channels = 2 if qubits == 8 else 1
        self.filters = filters
        self.output_channels = 16 * filters
        networks = []
        for _ in range(filters):
            layers = []
            current = self.group_channels
            for channels in config.hidden_channels:
                layers.append(ConvBlock(current, channels, 1, kernel_size=config.kernel_size,
                                        norm=config.normalization, act=config.activation))
                current = channels
            layers.extend([nn.Conv2d(current, self.group_channels, config.output_kernel_size,
                                     padding=config.output_kernel_size // 2), activation(config.output_activation)])
            networks.append(nn.Sequential(*layers))
        self.networks = nn.ModuleList(networks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if tuple(inputs.shape[1:]) != (16, 2, 2):
            raise ValueError("CNN replacement requires [batch, 16, 2, 2]")
        batch = inputs.shape[0]
        grouped = inputs.reshape(-1, self.group_channels, 2, 2)
        outputs = [network(grouped).reshape(batch, 16, 2, 2) for network in self.networks]
        return outputs[0] if self.filters == 1 else torch.stack(outputs, dim=2).reshape(batch, 32, 2, 2)


def residual_for_filters(features: torch.Tensor, filters: int) -> torch.Tensor:
    return features.repeat_interleave(filters, dim=1) if filters > 1 else features


def pack_patches(inputs: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] -> [B*C/2*H/2*W/2,2,2,2], adjacent channel pairs."""
    if inputs.ndim != 4 or any(d < 2 or d % 2 for d in inputs.shape[1:]):
        raise ValueError("Patch extraction requires even positive C, H, W")
    b, c, h, w = inputs.shape
    return inputs.reshape(b, c // 2, 2, h // 2, 2, w // 2, 2).permute(
        0, 1, 3, 5, 2, 4, 6).reshape(-1, 2, 2, 2)


def unpack_patches(patches: torch.Tensor, shape: tuple) -> torch.Tensor:
    b, c, h, w = shape
    return patches.reshape(b, c // 2, h // 2, w // 2, 2, 2, 2).permute(
        0, 1, 4, 2, 5, 3, 6).reshape(shape)


class PatchQuFeXLayer(QuFeXLayer):
    """Existing 8(1) circuit, shared across nonoverlapping two-channel patches."""

    def __init__(self, *, chunk_size: int = 256, **kwargs) -> None:
        super().__init__(**kwargs)
        if (self.qubits, self.filters) != (8, 1) or chunk_size < 1:
            raise ValueError("PatchQuFeX requires 8/1 and positive chunk_size")
        self.chunk_size = chunk_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        patches = pack_patches(inputs)
        # Match the published spatial-major, channel-interleaved encoding.
        vectors = patches.permute(0, 2, 3, 1).reshape(-1, 8)
        outputs = torch.cat([self.apply_filter(part) for part in vectors.split(self.chunk_size)])
        maps = outputs.reshape(-1, 2, 2, 2).permute(0, 3, 1, 2)
        return unpack_patches(maps, inputs.shape)


class PatchCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 2, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return unpack_patches(self.network(pack_patches(inputs)), inputs.shape)


class GalaxyClassifier(nn.Module):
    """Configurable QuFeX, grouped CNN replacement, or direct CNN."""

    def __init__(self, config: AppConfig, *, mode: str | None = None) -> None:
        super().__init__()
        config = config.for_model(mode or config.run.model)
        self.mode = config.run.model
        arch = config.architecture

        encoder: list[nn.Module] = []
        current_channels = 3
        for channels in arch.encoder_channels:
            encoder.append(
                ConvBlock(
                    current_channels,
                    channels,
                    arch.convolutions_per_block,
                    pool=arch.pooling, pool_size=arch.pool_size, kernel_size=arch.kernel_size,
                    norm=arch.normalization, act=arch.activation,
                )
            )
            current_channels = channels
        self.encoder = nn.Sequential(*encoder)
        bottleneck: list[nn.Module] = []
        if self.mode != "direct_cnn":
            if arch.projection == "conv" or current_channels != arch.compression_channels:
                bottleneck.append(ConvBlock(current_channels, arch.compression_channels, 1,
                                             kernel_size=1, norm=arch.normalization, act=arch.activation))
            bottleneck.append(nn.AdaptiveAvgPool2d(arch.quantum_spatial_size))
            current_channels = arch.compression_channels
        self.compression = nn.Sequential(*bottleneck)

        self.qufex: QuFeXLayer | None = None
        self.replacement: nn.Module | None = None
        if self.mode == "qufex":
            self.qufex = QuFeXLayer(**asdict(config.quantum))
        elif self.mode == "cnn_replacement":
            self.replacement = CNNReplacement(config.architectures.replacement, config.quantum.qubits, config.quantum.filters)
        elif self.mode == "compression_qufex":
            self.qufex = PatchQuFeXLayer(chunk_size=arch.circuit_chunk_size, **asdict(config.quantum))
        elif self.mode == "compression_patch_cnn":
            self.replacement = PatchCNN()
        elif self.mode == "compression_cnn":
            self.replacement = nn.Sequential(nn.Conv2d(current_channels, current_channels, 3, padding=1),
                                             nn.ReLU(), nn.Conv2d(current_channels, current_channels, 1))
        self.filters = config.quantum.filters if self.mode != "direct_cnn" else 1

        post_layers: list[nn.Module] = []
        current_channels *= self.filters
        for channels in arch.post_channels:
            post_layers.append(ConvBlock(current_channels, channels, arch.post_convolutions,
                                         kernel_size=arch.post_kernel_size, norm=arch.normalization, act=arch.activation))
            current_channels = channels
        self.post_quantum = nn.Sequential(*post_layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        head: list[nn.Module] = []
        current_features = current_channels
        for hidden in arch.classifier_hidden_neurons:
            head.extend([nn.Linear(current_features, hidden), activation(arch.activation), nn.Dropout(arch.dropout)])
            current_features = hidden
        head.append(nn.Linear(current_features, len(config.evaluation.class_names)))
        self.classifier = nn.Sequential(*head)
        if config.training.paired_randomness:
            # Independent streams: extractor allocation must not change common weights.
            for offset, module in enumerate((self.encoder, self.compression, self.post_quantum, self.classifier)):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(config.training.seeds[0] + 1009 * (offset + 1))
                    for child in module.modules():
                        if hasattr(child, "reset_parameters"):
                            child.reset_parameters()

    @property
    def quantum_parameter_count(self) -> int:
        return 0 if self.qufex is None else self.qufex.quantum_parameter_count

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.compression(self.encoder(inputs))
        if self.qufex is not None:
            # Quantum simulation remains float32 even when surrounding CNNs use AMP.
            with torch.autocast(device_type=features.device.type, enabled=False):
                quantum_features = self.qufex(features.float())
            residual = residual_for_filters(features, self.filters)
            features = residual + quantum_features.to(dtype=features.dtype)
        elif self.replacement is not None:
            features = residual_for_filters(features, self.filters) + self.replacement(features)
        features = self.post_quantum(features)
        features = self.global_pool(features).flatten(1)
        return self.classifier(features)
