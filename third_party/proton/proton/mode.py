from dataclasses import dataclass, field
from triton._C.libtriton import proton as triton_proton

metric_types = {"cycle": triton_proton.MetricType.CYCLE}

buffer_strategies = {
    "circular": triton_proton.BufferStrategy.CIRCULAR,
    "flush": triton_proton.BufferStrategy.FLUSH,
}

buffer_types = {
    "shared": triton_proton.BufferType.SHARED,
    "global": triton_proton.BufferType.GLOBAL,
    "local": triton_proton.BufferType.LOCAL,
}

sampling_strategies = {
    "none": triton_proton.SamplingStrategy.NONE,
}

granularities = {
    "cta": triton_proton.Granularity.CTA,
    "warp": triton_proton.Granularity.WARP,
    "warp_2": triton_proton.Granularity.WARP_2,
    "warp_4": triton_proton.Granularity.WARP_4,
    "warp_8": triton_proton.Granularity.WARP_8,
    "warp_group": triton_proton.Granularity.WARP_GROUP,
    "warp_group_2": triton_proton.Granularity.WARP_GROUP_2,
    "warp_group_4": triton_proton.Granularity.WARP_GROUP_4,
    "warp_group_8": triton_proton.Granularity.WARP_GROUP_8,
}


@dataclass(frozen=True)
class BaseMode:
    name: str


@dataclass(frozen=True)
class PCSampling(BaseMode):
    name = field(default="pc_sampling", init=False)
    interval: int = 1000

    def __post_init__(self):
        if self.interval <= 0:
            raise ValueError("Interval must be a positive integer.")

    def __str__(self):
        return f"{self.name}:interval={self.interval}"


@dataclass(frozen=True)
class InstrumentationMode(BaseMode):
    """Common base class for instrumentation modes with shared configuration."""
    metric_type: triton_proton.MetricType = field(default=triton_proton.MetricType.CYCLE, init=False)
    sampling_strategy: triton_proton.SamplingStrategy = triton_proton.SamplingStrategy.NONE
    sampling_options: str = ""
    granularity: triton_proton.Granularity = triton_proton.Granularity.WARP
    buffer_strategy: triton_proton.BufferStrategy = triton_proton.BufferStrategy.CIRCULAR
    buffer_type: triton_proton.BufferType = triton_proton.BufferType.SHARED
    buffer_size: int = 0

    def __str__(self):
        return (f"{self.name}:metric_type={self.metric_type}:sampling_strategy={self.sampling_strategy}"
                f":sampling_options={self.sampling_options}:granularity={self.granularity}"
                f":buffer_strategy={self.buffer_strategy}:buffer_type={self.buffer_type}"
                f":buffer_size={self.buffer_size}")


@dataclass(frozen=True)
class Default(InstrumentationMode):
    name: str = field(default="default", init=False)


@dataclass(frozen=True)
class MMA(InstrumentationMode):
    name: str = field(default="mma", init=False)
