from __future__ import annotations

from enum import Enum


class TranslatorTarget(str, Enum):
    """Target architecture for the Triton-to-Gluon translator.

    Known targets are listed as explicit members for discoverability.
    Unknown ``gfx*`` strings are accepted via ``_missing_()`` so that
    new AMD architectures work without adding an enum member.
    """

    NVIDIA = "nvidia"
    # AMD targets currently exercised by the translator test suite:
    GFX1250 = "gfx1250"
    GFX942 = "gfx942"
    GFX950 = "gfx950"

    @classmethod
    def _missing_(cls, value: object) -> "TranslatorTarget | None":
        """Allow any ``gfx*`` string as a valid AMD target."""
        if isinstance(value, str) and value.startswith("gfx"):
            obj = str.__new__(cls, value)
            obj._value_ = value
            return obj
        return None

    @property
    def is_amd(self) -> bool:
        return self != TranslatorTarget.NVIDIA

    @property
    def tensor_descriptor_import(self) -> str:
        """Return the import statement for the target's tensor descriptor module."""
        if self.is_amd:
            return "from triton.experimental.gluon.language.amd.gfx1250.tdm import tensor_descriptor"
        return "from triton.experimental.gluon.language.nvidia.hopper.tma import tensor_descriptor"

    @property
    def helpers_module(self) -> str:
        """Return the helpers module path for this target."""
        base = "triton.tools.triton_to_gluon_translator"
        if self.is_amd:
            return f"{base}.amd_helpers"
        return f"{base}.nvidia_helpers"
