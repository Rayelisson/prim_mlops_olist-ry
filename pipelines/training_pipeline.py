from zenml import pipeline

from steps.clear_data import clear_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model


@pipeline()
def training_pipeline(data_path: str):
    df = ingest_df(data_path=data_path)
    clear_df(df)
    train_model(df)
    evaluate_model(df)
