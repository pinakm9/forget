import ast
import unittest
from pathlib import Path


MODULES_DIR = Path(__file__).resolve().parents[1] / "modules"
WEIGHT_NAMES = {
    "kl_weight",
    "orthogonality_weight",
    "uniformity_weight",
    "forget_weight",
}
EXPECTED_ORDER = (
    "kl_weight",
    "orthogonality_weight",
    "uniformity_weight",
    "forget_weight",
)


class LossWeightOrderingTests(unittest.TestCase):
    def test_weight_tuples_match_processor_unpacking_order(self):
        mismatches = []
        tuples_checked = 0

        for path in MODULES_DIR.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Tuple):
                    continue

                names = tuple(
                    element.id if isinstance(element, ast.Name) else None
                    for element in node.elts
                )
                if len(names) != 4 or set(names) != WEIGHT_NAMES:
                    continue

                tuples_checked += 1
                if names != EXPECTED_ORDER:
                    mismatches.append(
                        f"{path.relative_to(MODULES_DIR)}:{node.lineno}: {names}"
                    )

        self.assertGreater(tuples_checked, 0)
        self.assertEqual(mismatches, [])

    def test_unused_uniformity_weight_is_not_exposed(self):
        paths = [
            *[
                MODULES_DIR / "imagenet" / name
                for name in (
                    "ascent.py",
                    "ascent_descent.py",
                    "gandikota.py",
                    "ortho.py",
                    "ortho_s.py",
                    "salun.py",
                    "surgery.py",
                    "surgery_a.py",
                    "train.py",
                )
            ],
            MODULES_DIR / "celeba_male" / "vae_fu.py",
            MODULES_DIR / "celeba_male" / "vae_ifb.py",
            MODULES_DIR / "mnist" / "vae_fu.py",
            MODULES_DIR / "mnist" / "vae_ret.py",
            MODULES_DIR / "mnist" / "vae_salun.py",
        ]
        occurrences = []

        for path in paths:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == "uniformity_weight":
                    occurrences.append(
                        f"{path.relative_to(MODULES_DIR)}:{node.lineno}"
                    )
                elif isinstance(node, ast.arg) and node.arg == "uniformity_weight":
                    occurrences.append(
                        f"{path.relative_to(MODULES_DIR)}:{node.lineno}"
                    )

        self.assertEqual(occurrences, [])


if __name__ == "__main__":
    unittest.main()
