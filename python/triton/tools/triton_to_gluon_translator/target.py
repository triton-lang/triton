from __future__ import annotations

from enum import Enum


class TranslatorTarget(str, Enum):
    """Target architecture for the Triton-to-Gluon translator.

    Known targets are listed as explicit members for discoverability.
    Unknown ``gfx*`` strings are accepted via ``_missing_()`` so that
    new AMD architectures work without adding an enum member.
    """

    GENERIC = "generic"
    SM80 = "sm80"
    SM90 = "sm90"
    SM100 = "sm100"
    SM103 = "sm103"
    # AMD targets currently exercised by the translator test suite:
    GFX90A = "gfx90a"
    GFX1250 = "gfx1250"
    GFX942 = "gfx942"
    GFX950 = "gfx950"

    @classmethod
    def _missing_(cls, value: object) -> "TranslatorTarget | None":
        if value not in cls._value2member_map_:
            return None
        if isinstance(value, str):
            return cls(value)
        return None

    @property
    def is_amd(self) -> bool:
        return self in (
            TranslatorTarget.GFX90A,
            TranslatorTarget.GFX942,
            TranslatorTarget.GFX950,
            TranslatorTarget.GFX1250,
        )

    @property
    def is_nvidia(self) -> bool:
        return self in (
            TranslatorTarget.SM80,
            TranslatorTarget.SM90,
            TranslatorTarget.SM100,
            TranslatorTarget.SM103,
        )

    @property
    def tensor_descriptor_import(self) -> str:
        module = "amd.gfx1250.tdm" if self.is_amd else "nvidia.hopper.tma"
        return f"from triton.experimental.gluon.language.{module} import tensor_descriptor"

    @property
    def helpers_module(self) -> str:
        base = "triton.tools.triton_to_gluon_translator"

        if self.is_amd:
            return f"{base}.amd_helpers"

        if self.is_nvidia:
            if self in (TranslatorTarget.SM100, TranslatorTarget.SM103):
                return f"{base}.blackwell_helpers"
            if self in (TranslatorTarget.SM90):
                return f"{base}.hopper_helpers"
            if self in (TranslatorTarget.SM80):
                return f"{base}.nvidia_helpers"

        return f"{base}.common_helpers"
