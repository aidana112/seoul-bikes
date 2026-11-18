"""Готовит данные. Запуск из корня проекта: python src/prepare.py

Читает data/raw/SeoulBikeData.csv и пишет data/clean.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "data" / "raw" / "SeoulBikeData.csv", encoding="latin1")

# короткие имена вместо "Temperature(°C)" и похожих
raw.columns = ["date", "count", "hour", "temp", "humidity", "wind", "visibility",
               "dew_point", "solar", "rainfall", "snowfall", "season", "holiday", "functioning"]

# 295 часов сервис не работал, и аренд там всегда 0
df = raw[raw["functioning"] == "Yes"].drop(columns="functioning").copy()

# из даты берём только выходной или нет, по выходным спрос по часам другой
date = pd.to_datetime(df["date"], dayfirst=True)
df["weekend"] = np.where(date.dt.day_of_week >= 5, "yes", "no")
df = df.drop(columns="date")

df.to_csv(ROOT / "data" / "clean.csv", index=False)
print(raw.shape, "->", df.shape)
