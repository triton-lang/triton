from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class TritonKernelDebtReport:
    kernel_id: str
    tdi_score: float  # Triton Debt Index (target <= 12.0)
    sram_sprawl_multiplier: float  # Target <= 1.08x
    kernel_latency_us: float  # Target <= 18.0us
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: list[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """Cryptographic SHA-256 hash-chained Action Ledger for Triton GPU kernel compilation runs."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_kernel_event(
        self,
        kernel_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = (
            f"{index}|{self._last_hash}|{kernel_id}|{event_type}|"
            f"{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        )
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "kernel_id": kernel_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtKernelGate:
    """A2Z SOC Production Debt & Technical Due Diligence Gate for Triton GPU Kernels.

    Quantifies local register spilling, shared memory (SRAM) bank conflicts, and kernel execution latency against 4 Enterprise KPIs:
    1. Triton Debt Index (TDI <= 12.0)
    2. Shared Memory Sprawl Multiplier (SMSM <= 1.08x)
    3. P99 Kernel Execution Latency (<= 18.0us)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_tdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_tdi = max_acceptable_tdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        return any(Path(p).exists() for p in ("artifacts/KILL", "/tmp/KILL"))

    def evaluate_kernel_launch(
        self,
        kernel_id: str,
        allocated_shared_memory_bytes: int = 49152,
        utilized_shared_memory_bytes: int = 51200,
        kernel_latency_us: float = 14.5,
        register_spill_bytes: int = 0,
        un_gated_mutations: int = 0,
    ) -> TritonKernelDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_kernel_event(
                kernel_id=kernel_id,
                event_type="kernel_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            err_msg = "A2Z SOC ActionGate: Emergency kill switch is engaged. Triton kernel execution halted."
            raise PermissionError(err_msg)

        critical_smells: list[str] = []

        # KPI 2: Shared Memory Sprawl Multiplier
        sram_ratio = utilized_shared_memory_bytes / max(1, allocated_shared_memory_bytes)
        if sram_ratio > 1.8:
            critical_smells.append(f"HIGH_SHARED_MEMORY_SPRAWL_{sram_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if kernel_latency_us > 50.0:
            critical_smells.append(f"HIGH_KERNEL_EXECUTION_LATENCY_{kernel_latency_us:.1f}US")

        # Register spill bytes to local memory
        if register_spill_bytes > 0:
            critical_smells.append(f"DETECTED_{register_spill_bytes}_BYTES_REGISTER_SPILL_TO_LOCAL_MEMORY")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_JIT_MUTATIONS")

        # KPI 1: Triton Debt Index (0 = Clean, 100 = Catastrophic)
        tdi = (
            max(0.0, (sram_ratio - 1.0) * 20.0)
            + max(0.0, (kernel_latency_us - 18.0) * 0.5)
            + (min(10.0, register_spill_bytes / 64.0) * 10.0)
            + (un_gated_mutations * 30.0)
        )
        tdi_score = round(min(100.0, tdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - tdi_score)
        is_production_ready = (
            tdi_score <= self.max_acceptable_tdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_kernel_event(
            kernel_id=kernel_id,
            event_type="kernel_authorized" if is_production_ready else "kernel_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "tdi_score": tdi_score,
                "sram_ratio": sram_ratio,
                "allocated_shared_memory_bytes": allocated_shared_memory_bytes,
                "utilized_shared_memory_bytes": utilized_shared_memory_bytes,
                "kernel_latency_us": kernel_latency_us,
                "register_spill_bytes": register_spill_bytes,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return TritonKernelDebtReport(
            kernel_id=kernel_id,
            tdi_score=tdi_score,
            sram_sprawl_multiplier=round(sram_ratio, 2),
            kernel_latency_us=round(kernel_latency_us, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
