import numpy as np
from scipy.optimize import rosen
from scam_optimizer import ScamOptimizer


def test_rosenbrock_convergence():
    # Define the Rosenbrock function with known global minimum
    fun = lambda x: rosen(x)
    global_min = np.array([1., 1.]) # Global minimum at (1, 1)

    # Random starting point for the optimizer
    x0 = np.random.uniform(low=-5, high=5, size=2)

    # Initialize optimizer
    optimizer = ScamOptimizer(fun=fun, x0=x0, bounds=[(-5, 5), (-5, 5)], maxiter=1000)

    # Run optimization
    result = optimizer.optimize()

    # Check if optimizer converges to the global minimum
    assert np.allclose(result.x, global_min, rtol=1e-3)
