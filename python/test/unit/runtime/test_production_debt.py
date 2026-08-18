import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../../../python/triton/runtime/production_debt.py",
)
spec = importlib.util.spec_from_file_location("triton_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["triton_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtKernelGate = production_debt_mod.ProductionDebtKernelGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtKernelGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtKernelGate(
            never_equate_intent_to_approval=True,
            max_acceptable_tdi=12.0,
        )

    def test_clean_kernel_launch_passes_readiness(self) -> None:
        report = self.gate.evaluate_kernel_launch(
            kernel_id="flash_attention_fp8_kernel_sm90",
            allocated_shared_memory_bytes=49152,
            utilized_shared_memory_bytes=51200,
            kernel_latency_us=14.5,
            register_spill_bytes=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.tdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_kernel_launch_fails_debt(self) -> None:
        report = self.gate.evaluate_kernel_launch(
            kernel_id="uncalibrated_triton_kernel",
            allocated_shared_memory_bytes=49152,
            utilized_shared_memory_bytes=135000,  # 2.74x SRAM sprawl
            kernel_latency_us=120.0,  # High latency
            register_spill_bytes=256,  # 256 bytes spilled to local memory
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.tdi_score, 50.0)
        self.assertIn("HIGH_SHARED_MEMORY_SPRAWL_2.75X", report.critical_smells)
        self.assertIn("HIGH_KERNEL_EXECUTION_LATENCY_120.0US", report.critical_smells)
        self.assertIn("DETECTED_256_BYTES_REGISTER_SPILL_TO_LOCAL_MEMORY", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_JIT_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_kernel_launch("kernel-1")
        self.gate.evaluate_kernel_launch("kernel-2")
        self.gate.evaluate_kernel_launch("kernel-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
