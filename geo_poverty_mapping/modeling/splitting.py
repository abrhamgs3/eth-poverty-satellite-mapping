"""Train/test split, keeping all feature and target variants aligned."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def train_test_split(
    df: pd.DataFrame, test_fraction: float, rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    shuffled = rng.permutation(n)
    cut = int(n * test_fraction)
    df_test = df.iloc[shuffled[:cut]].reset_index(drop=True)
    df_train = df.iloc[shuffled[cut:]].reset_index(drop=True)
    return df_train, df_test
