"""Обучает модель спроса на велосипеды. Запуск из корня проекта: python src/train.py

Читает параметры из params.json. Пишет модель в models/model.joblib, а метрики в metrics.json.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
params = json.loads((ROOT / "params.json").read_text(encoding="utf-8"))

df = pd.read_csv(ROOT / "data" / "clean.csv")
num_cols = ["temp", "humidity", "wind", "visibility", "dew_point", "solar", "rainfall", "snowfall"]
cat_cols = ["season", "holiday", "weekend"]
X = df[["hour"] + num_cols + cat_cols]
y = df["count"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"])

num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
if params["hour_bins"] > 0:
    # час режем на интервалы, у каждого интервала свой вес
    bins = KBinsDiscretizer(n_bins=params["hour_bins"], encode="onehot", strategy="uniform")
    prep = make_column_transformer((num, num_cols), (bins, ["hour"]), (cat, cat_cols))
else:
    prep = make_column_transformer((num, ["hour"] + num_cols), (cat, cat_cols))

regressors = {
    "linear": LinearRegression(),
    "ridge": Ridge(alpha=params["alpha"]),
    "lasso": Lasso(alpha=params["alpha"], max_iter=50000),
}
model = make_pipeline(prep, regressors[params["model"]])

# эксперименты сравниваем по кросс-валидации на train, тест для выбора не нужен
cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
cv_rmse = -cross_val_score(model, X_train, y_train, cv=5,
                           scoring="neg_root_mean_squared_error").mean()
model.fit(X_train, y_train)
pred = model.predict(X_test)

metrics = {
    "cv_r2": round(float(cv_r2), 4),
    "cv_rmse": round(float(cv_rmse)),
    "test_r2": round(float(r2_score(y_test, pred)), 4),
    "test_rmse": round(float(root_mean_squared_error(y_test, pred))),
    "test_mae": round(float(mean_absolute_error(y_test, pred))),
}

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "models" / "model.joblib")
(ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(params["model"], metrics)
