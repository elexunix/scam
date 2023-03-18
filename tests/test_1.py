import torch
from torch.optim import Optimizer
import unittest

class TestSCAM(unittest.TestCase):
    def test_step(self):
        # Define a simple quadratic function in two variables.
        def closure():
            x = torch.randn(2, requires_grad=True)
            y = 2 * x[0]  2 + x[1]  2
            return y

        # Initialize the optimizer with some test parameters.
        x = torch.randn(2, requires_grad=True)
        optimizer = SCAM([x], lr=0.1)

        # Optimize the function for a few iterations.
        for i in range(100):
            optimizer.zero_grad()
            loss = optimizer.step(closure)
            if loss < 1e-6:
                break

        # Check that the optimizer has found the minimum of the function.
        self.assertLess(loss, 1e-6)
        self.assertLess(torch.abs(x[0]), 1e-3)
        self.assertLess(torch.abs(x[1]), 1e-3)

if name == 'main':
    unittest.main()
