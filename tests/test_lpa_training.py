import importlib.util
import unittest

from glm53_setup.config import (
    HIDDEN_SIZE,
    MODEL,
    MODEL_LAYERS,
    REVISION,
    TEACHER_PRECISION,
)
from glm53_setup.validation.train_lpa import validate_teacher


class TeacherProvenanceTests(unittest.TestCase):
    def test_teacher_identity_is_required_and_checked_before_fitting(self):
        teacher = {
            "model": MODEL,
            "revision": REVISION,
            "layers": MODEL_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "precision": TEACHER_PRECISION,
        }
        self.assertEqual(validate_teacher({"teacher": teacher}), teacher)
        for key, value in [
            ("revision", "0" * 40),
            ("hidden_size", 1),
            ("precision", "other"),
        ]:
            with self.assertRaises(ValueError):
                validate_teacher({"teacher": {**teacher, key: value}})
        with self.assertRaises(ValueError):
            validate_teacher({})


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Torch training environment required"
)
class ProjectedRidgeTests(unittest.TestCase):
    def test_recovers_known_linear_map_and_regularizes_singular_gram(self):
        import torch

        from glm53_setup.validation.train_lpa import solve_projected_ridge

        torch.manual_seed(7)
        x = torch.randn(100, 3, dtype=torch.float64)
        expected = torch.randn(3, 5, dtype=torch.float64)
        fit = solve_projected_ridge(x.T @ x, x.T @ (x @ expected), 1e-9)
        self.assertTrue(torch.allclose(fit, expected, atol=1e-7))
        x[:, 2] = x[:, 1]
        fit = solve_projected_ridge(x.T @ x, x.T @ (x @ expected), 0.001)
        self.assertTrue(torch.isfinite(fit).all())


if __name__ == "__main__":
    unittest.main()
