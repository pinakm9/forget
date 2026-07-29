import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULES_DIR = Path(__file__).resolve().parents[1] / "modules"
sys.path.insert(0, str(MODULES_DIR))

import gpu_memory


class GpuMemoryProfileTests(unittest.TestCase):
    def test_train_writes_one_cuda_measurement_per_batch(self):
        mb = 1024**2

        @gpu_memory.collect_memory_usage
        def train(folder):
            @gpu_memory.profile_gpu_memory
            def process_batch():
                return "batch-result"

            return [process_batch(), process_batch()]

        with tempfile.TemporaryDirectory() as folder, mock.patch.multiple(
            gpu_memory.torch.cuda,
            is_available=mock.Mock(return_value=True),
            current_device=mock.Mock(return_value=2),
            synchronize=mock.Mock(),
            memory_allocated=mock.Mock(return_value=100 * mb),
            reset_peak_memory_stats=mock.Mock(),
            max_memory_allocated=mock.Mock(
                side_effect=[125 * mb, 140 * mb]
            ),
        ):
            self.assertEqual(
                train(folder), ["batch-result", "batch-result"]
            )
            with open(Path(folder) / "memory_usage.csv", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual([row["step"] for row in rows], ["1", "2"])
        self.assertEqual([row["device"] for row in rows], ["cuda:2", "cuda:2"])
        self.assertEqual(
            [float(row["peak_memory_difference_mb"]) for row in rows],
            [25.0, 40.0],
        )

    def test_cpu_run_is_non_intrusive_and_records_zero_gpu_memory(self):
        @gpu_memory.collect_memory_usage
        def train(folder):
            @gpu_memory.profile_gpu_memory
            def process_batch(value):
                return value + 1

            return process_batch(4)

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
            gpu_memory.torch.cuda, "is_available", return_value=False
        ):
            self.assertEqual(train(folder), 5)
            with open(Path(folder) / "memory_usage.csv", newline="") as file:
                row = next(csv.DictReader(file))

        self.assertEqual(row["cuda_available"], "False")
        self.assertEqual(float(row["peak_memory_difference_mb"]), 0.0)

    def test_csv_is_written_when_process_batch_raises(self):
        @gpu_memory.collect_memory_usage
        def train(folder):
            @gpu_memory.profile_gpu_memory
            def process_batch():
                raise RuntimeError("failed batch")

            process_batch()

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
            gpu_memory.torch.cuda, "is_available", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "failed batch"):
                train(folder)
            with open(Path(folder) / "memory_usage.csv", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)

    def test_profiled_function_outside_train_has_no_profiling_side_effects(self):
        @gpu_memory.profile_gpu_memory
        def process_batch():
            return 7

        with mock.patch.object(
            gpu_memory.torch.cuda, "is_available"
        ) as is_available:
            self.assertEqual(process_batch(), 7)
            is_available.assert_not_called()


if __name__ == "__main__":
    unittest.main()
