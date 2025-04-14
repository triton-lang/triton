from dataclasses import dataclass, field
from triton._C.libtriton import proton as triton_proton

metric_types = {"cycle": triton_proton.METRIC_TYPE.CYCLE}

buffer_strategies = {
    "circular": triton_proton.BUFFER_STRATEGY.CIRCULAR,
    "flush": triton_proton.BUFFER_STRATEGY.FLUSH,
}

buffer_types = {
    "shared": triton_proton.BUFFER_TYPE.SHARED,
    "global": triton_proton.BUFFER_TYPE.GLOBAL,
    "local": triton_proton.BUFFER_TYPE.LOCAL,
}

sampling_strategies = {
    "none": triton_proton.SAMPLING_STRATEGY.NONE,
}

granularities = {
    "cta": triton_proton.GRANULARITY.CTA,
    "warp": triton_proton.GRANULARITY.WARP,
    "warp_2": triton_proton.GRANULARITY.WARP_2,
    "warp_4": triton_proton.GRANULARITY.WARP_4,
    "warp_8": triton_proton.GRANULARITY.WARP_8,
    "warp_group": triton_proton.GRANULARITY.WARP_GROUP,
    "warp_group_2": triton_proton.GRANULARITY.WARP_GROUP_2,
    "warp_group_4": triton_proton.GRANULARITY.WARP_GROUP_4,
    "warp_group_8": triton_proton.GRANULARITY.WARP_GROUP_8,
}


@dataclass(frozen=True)
class BaseMode:
    name: str


@dataclass(frozen=True)
class PCSampling(BaseMode):
    name: str = field(default="pc_sampling", init=False)
    interval: int = 1000

    def __post_init__(self):
        if self.interval <= 0:
            raise ValueError("Interval must be a positive integer.")

    def __str__(self):
        return f"{self.name}:interval={self.interval}"


@dataclass(frozen=True)
class InstrumentationMode(BaseMode):
    """Common base class for instrumentation modes with shared configuration."""
    metric_type: triton_proton.METRIC_TYPE = field(default=triton_proton.METRIC_TYPE.CYCLE, init=False)
    sampling_strategy: triton_proton.SAMPLING_STRATEGY = triton_proton.SAMPLING_STRATEGY.NONE
    sampling_options: str = ""
    granularity: triton_proton.GRANULARITY = triton_proton.GRANULARITY.WARP
    buffer_strategy: triton_proton.BUFFER_STRATEGY = triton_proton.BUFFER_STRATEGY.CIRCULAR
    buffer_type: triton_proton.BUFFER_TYPE = triton_proton.BUFFER_TYPE.SHARED
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
