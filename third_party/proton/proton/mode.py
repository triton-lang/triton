from dataclasses import dataclass, field
from triton._C.libtriton import proton as triton_proton
from typing import List
from enum import Enum

metric_types = {"cycle": triton_proton.METRIC_TYPE.CYCLE}

buffer_strategies = {
    "circular": triton_proton.BUFFER_STRATEGY.CIRCULAR,
    "flush": triton_proton.BUFFER_STRATEGY.FLUSH,
}

buffer_types = {
    "shared": triton_proton.BUFFER_TYPE.SHARED,
    "global": triton_proton.BUFFER_TYPE.GLOBAL,
}

sampling_strategies = {
    "none": triton_proton.SAMPLING_STRATEGY.NONE,
    "selective": triton_proton.SAMPLING_STRATEGY.SELECTIVE,
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


class Optimize(Enum):
    TIMESHIFT = "time_shift"
    SCHED_STORES = "sched_stores"
    SCHED_BARRIERS = "sched_barriers"
    CLOCK32 = "clock32"

    def __str__(self):
        return self.value


optimizations = {
    "time_shift": Optimize.TIMESHIFT,
    "sched_stores": Optimize.SCHED_STORES,
    "sched_barriers": Optimize.SCHED_BARRIERS,
    "clock32": Optimize.CLOCK32,
}


@dataclass(frozen=True)
class BaseMode:
    name: str


@dataclass(frozen=True)
class PCSampling(BaseMode):
    name: str = field(default="pcsampling", init=False)
    interval: int = 1000

    def __post_init__(self):
        if self.interval <= 0:
            raise ValueError("Interval must be a positive integer.")

    def __str__(self):
        return f"{self.name}:interval={self.interval}"


@dataclass(frozen=True)
class InstrumentationMode(BaseMode):
    """Common base class for instrumentation modes with shared configuration."""
    metric_type: triton_proton.METRIC_TYPE = triton_proton.METRIC_TYPE.CYCLE
    sampling_strategy: triton_proton.SAMPLING_STRATEGY = triton_proton.SAMPLING_STRATEGY.NONE
    sampling_options: str = ""
    granularity: triton_proton.GRANULARITY = triton_proton.GRANULARITY.WARP
    buffer_strategy: triton_proton.BUFFER_STRATEGY = triton_proton.BUFFER_STRATEGY.CIRCULAR
    buffer_type: triton_proton.BUFFER_TYPE = triton_proton.BUFFER_TYPE.SHARED
    buffer_size: int = 0
    optimizations: List[Optimize] = field(default_factory=list)

    def __post_init__(self):
        # automatically map string inputs to enums using the global lookup dicts
        mappings = [
            ("metric_type", metric_types),
            ("sampling_strategy", sampling_strategies),
            ("granularity", granularities),
            ("buffer_strategy", buffer_strategies),
            ("buffer_type", buffer_types),
        ]
        for field_name, lookup in mappings:
            value = getattr(self, field_name)
            if isinstance(value, str):
                if value not in lookup:
                    raise ValueError(f"Unknown {field_name}: {value}")
                object.__setattr__(self, field_name, lookup[value])

        values_str = getattr(self, "optimizations")
        if isinstance(values_str, str):
            values = [value.strip() for value in values_str.split(",")] if len(values_str) > 0 else []
            for value in values:
                if value not in optimizations:
                    raise ValueError(f"Unknown optimization: {value}")
            object.__setattr__(self, "optimizations", [optimizations[value] for value in values])

    def __str__(self):
        optimizations_str = ",".join([str(opt) for opt in self.optimizations])
        return (f"{self.name}:metric_type={self.metric_type}:sampling_strategy={self.sampling_strategy}"
                f":sampling_options={self.sampling_options}:granularity={self.granularity}"
                f":buffer_strategy={self.buffer_strategy}:buffer_type={self.buffer_type}"
                f":buffer_size={self.buffer_size}:optimizations={optimizations_str}")


@dataclass(frozen=True)
class Default(InstrumentationMode):
    name: str = field(default="default", init=False)


@dataclass(frozen=True)
class MMA(InstrumentationMode):
    name: str = field(default="mma", init=False)
