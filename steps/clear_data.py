import logging

import pandas as pd
from zenml import step


@step
def clear_data(df: pd.DataFrame) -> pd.DataFrame:
    pass
