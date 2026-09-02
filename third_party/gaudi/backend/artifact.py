# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import struct
from typing import Any, Mapping


_MAGIC = b"GAK1"
_HEADER = struct.Struct("<4sHHIQ")
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_ELF_BYTES = 64 * 1024 * 1024


class ArtifactError(ValueError):
    """Raised when a cached Gaudi artifact is malformed or incompatible."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass(frozen=True)
class GaudiKernelArtifactV1:
    """Self-validating TPC kernel artifact consumed by the Bridge launcher.

    The envelope deliberately contains no filesystem paths.  A runtime may
    materialize the embedded ELF in a private cache, while the digest passed in
    ``nodeParams`` remains stable across workers.
    """

    manifest: Mapping[str, Any]
    elf: bytes

    @staticmethod
    def _hash_payload(manifest: Mapping[str, Any], elf: bytes) -> str:
        unhashed = dict(manifest)
        unhashed.pop("artifact_hash", None)
        return hashlib.sha256(_canonical_json(unhashed) + b"\0" + elf).hexdigest()

    @classmethod
    def create(cls, manifest: Mapping[str, Any], elf: bytes) -> "GaudiKernelArtifactV1":
        complete = {
            "abi": "GaudiKernelArtifactV1",
            "abi_major": 1,
            "abi_minor": 0,
            **dict(manifest),
            "elf_sha256": hashlib.sha256(elf).hexdigest(),
        }
        complete["artifact_hash"] = cls._hash_payload(complete, elf)
        artifact = cls(complete, bytes(elf))
        artifact.validate()
        return artifact

    @property
    def artifact_hash(self) -> str:
        return str(self.manifest["artifact_hash"])

    def validate(self) -> None:
        if self.manifest.get("abi") != "GaudiKernelArtifactV1":
            raise ArtifactError("unsupported Gaudi kernel artifact ABI")
        if self.manifest.get("abi_major") != 1:
            raise ArtifactError("unsupported Gaudi kernel artifact major version")
        if self.manifest.get("target") != "gaudi2":
            raise ArtifactError("GaudiKernelArtifactV1 currently supports target=gaudi2 only")
        if self.manifest.get("engine") != "tpc":
            raise ArtifactError("kernel artifacts must contain a TPC engine node")
        if not self.elf.startswith(b"\x7fELF"):
            raise ArtifactError("TPC payload is not an ELF object")
        if len(self.elf) > _MAX_ELF_BYTES:
            raise ArtifactError("TPC ELF exceeds the artifact size limit")
        if hashlib.sha256(self.elf).hexdigest() != self.manifest.get("elf_sha256"):
            raise ArtifactError("TPC ELF digest mismatch")
        if self._hash_payload(self.manifest, self.elf) != self.manifest.get("artifact_hash"):
            raise ArtifactError("Gaudi artifact digest mismatch")

    def to_bytes(self) -> bytes:
        self.validate()
        manifest = _canonical_json(self.manifest)
        if len(manifest) > _MAX_MANIFEST_BYTES:
            raise ArtifactError("Gaudi artifact manifest exceeds the size limit")
        return _HEADER.pack(_MAGIC, 1, 0, len(manifest), len(self.elf)) + manifest + self.elf

    @classmethod
    def from_bytes(cls, payload: bytes) -> "GaudiKernelArtifactV1":
        if len(payload) < _HEADER.size:
            raise ArtifactError("truncated Gaudi artifact header")
        magic, major, minor, manifest_len, elf_len = _HEADER.unpack_from(payload)
        if magic != _MAGIC or major != 1:
            raise ArtifactError("unsupported Gaudi artifact envelope")
        if minor > 0:
            raise ArtifactError("artifact requires a newer Gaudi runtime")
        if manifest_len > _MAX_MANIFEST_BYTES or elf_len > _MAX_ELF_BYTES:
            raise ArtifactError("Gaudi artifact section exceeds the size limit")
        expected = _HEADER.size + manifest_len + elf_len
        if len(payload) != expected:
            raise ArtifactError("truncated or trailing data in Gaudi artifact")
        manifest_start = _HEADER.size
        manifest_end = manifest_start + manifest_len
        try:
            manifest = json.loads(payload[manifest_start:manifest_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactError("invalid Gaudi artifact manifest") from exc
        artifact = cls(manifest, payload[manifest_end:])
        artifact.validate()
        return artifact
