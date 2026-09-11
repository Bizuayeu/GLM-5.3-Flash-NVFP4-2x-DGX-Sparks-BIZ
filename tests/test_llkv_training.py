import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Torch training environment required"
)
class ProjectedRidgeTests(unittest.TestCase):
    def test_dense_export_preserves_diagonal_and_lowrank_prediction(self):
        import torch

        from glm53_setup.validation.train_llkv import densify_affine_weights

        torch.manual_seed(9)
        x = torch.randn(11, 8, dtype=torch.float64)
        w = {
            "mean": torch.randn(8, dtype=torch.float64),
            "scale": torch.randn(8, dtype=torch.float64),
            "down": torch.randn(8, 3, dtype=torch.float64),
            "up": torch.randn(3, 8, dtype=torch.float64),
            "bias": torch.randn(8, dtype=torch.float64),
        }
        expected = x * w["scale"] + ((x - w["mean"]) @ w["down"]) @ w["up"] + w["bias"]
        d = densify_affine_weights({4: w})[4]
        actual = x + ((x - d["mean"]) @ d["down"]) @ d["up"] + d["bias"]
        self.assertTrue(torch.allclose(expected, actual, atol=1e-10))

    def test_recovers_known_linear_map_and_regularizes_singular_gram(self):
        import torch

        from glm53_setup.validation.train_llkv import solve_projected_ridge

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
