import numpy as np
import pandas as pd


def generate_data(a: float, b: float, noise: float, n_points: int, random_state: int | None = None) -> pd.DataFrame:
    """Generate synthetic linear data y = a*x + b + error.

    Args:
        a: slope
        b: intercept
        noise: standard deviation of gaussian noise
        n_points: number of samples
        random_state: optional random seed for reproducibility

    Returns:
        DataFrame with columns ['x', 'y']
    """
    rng = np.random.RandomState(random_state) if random_state is not None else np.random
    x = rng.rand(n_points) * 10
    error = rng.randn(n_points) * noise
    y = a * x + b + error
    return pd.DataFrame({"x": x, "y": y})
