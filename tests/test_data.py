import pandas as pd

from data import generate_data


def test_generate_data_shape_and_columns():
    df = generate_data(a=1.5, b=0.5, noise=0.1, n_points=50, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 50
