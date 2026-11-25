"""Обучает классификатор часов высокого спроса. Запуск из корня проекта: python src/train_clf.py

Читает параметры из params_clf.json. Пишет модель в models/model_clf.joblib, а метрики в metrics_clf.json.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
params = json.loads((ROOT / "params_clf.json").read_text(encoding="utf-8"))
HIGH_DEMAND = 1000  # час с 1000 аренд и больше считаем часом высокого спроса

df = pd.read_csv(ROOT / "data" / "clean.csv")
num_cols = ["temp", "humidity", "wind", "visibility", "dew_point", "solar", "rainfall", "snowfall"]
cat_cols = ["season", "holiday", "weekend"]
# count в признаки не берём, из него получена цель
X = df[["hour"] + num_cols + cat_cols]
y = (df["count"] >= HIGH_DEMAND).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"], stratify=y)

num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
bins = KBinsDiscretizer(n_bins=params["hour_bins"], encode="onehot", strategy="uniform")
prep = make_column_transformer((num, num_cols), (bins, ["hour"]), (cat, cat_cols))

classifiers = {
    "logreg": LogisticRegression(C=params["C"], class_weight=params["class_weight"], max_iter=1000),
    "tree": DecisionTreeClassifier(max_depth=4, class_weight=params["class_weight"],
                                   random_state=params["random_state"]),
}
model = make_pipeline(prep, classifiers[params["model"]])

# вероятности на кросс-валидации, к ним применяем порог из params_clf.json
cv_proba = cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba")[:, 1]
cv_pred = (cv_proba >= params["threshold"]).astype(int)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= params["threshold"]).astype(int)

metrics = {
    "cv_f1": round(float(f1_score(y_train, cv_pred)), 4),
    "cv_roc_auc": round(float(roc_auc_score(y_train, cv_proba)), 4),
    "cv_precision": round(float(precision_score(y_train, cv_pred)), 4),
    "cv_recall": round(float(recall_score(y_train, cv_pred)), 4),
    "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "test_precision": round(float(precision_score(y_test, pred)), 4),
    "test_recall": round(float(recall_score(y_test, pred)), 4),
    "test_f1": round(float(f1_score(y_test, pred)), 4),
}

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "models" / "model_clf.joblib")
(ROOT / "metrics_clf.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(params["model"], metrics)
