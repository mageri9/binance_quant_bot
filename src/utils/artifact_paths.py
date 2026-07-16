import os


def get_oos_path(model_path: str) -> str:
    """
    Возвращает путь к parquet-файлу с Out-of-Sample предсказаниями,
    соответствующему pickle-артефакту модели по пути model_path.
    Файл лежит рядом с моделью:
      models/saved_models/lgbm_BTCUSDT_1h.pkl
      -> models/saved_models/lgbm_BTCUSDT_1h_oos.parquet
    """
    base, _ = os.path.splitext(model_path)
    return f"{base}_oos.parquet"