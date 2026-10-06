"""Train and compare models, save the best full pipeline."""
import sys, json; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from app.data import CATEGORICAL, NUMERIC, FEATURES, TARGET, MODEL_PATH, METRICS_PATH, load_clean

df = load_clean()
X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df[TARGET], test_size=0.2, random_state=42)

def make(model):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                             ("num", StandardScaler(), NUMERIC)])
    return Pipeline([("prep", pre), ("model", model)])

candidates = {"Linear Regression": LinearRegression(),
              "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
              "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=14, n_jobs=-1, random_state=42)}
results, best_name, best_pipe, best_r2 = {}, None, None, -1
for name, model in candidates.items():
    pipe = make(model).fit(X_train, y_train); pred = pipe.predict(X_test)
    results[name] = {"MAE": round(mean_absolute_error(y_test, pred)),
                     "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred)))),
                     "R2": round(r2_score(y_test, pred), 4)}
    print(f"{name:18s}", results[name])
    if results[name]["R2"] > best_r2: best_name, best_pipe, best_r2 = name, pipe, results[name]["R2"]

joblib.dump(best_pipe, MODEL_PATH)
METRICS_PATH.write_text(json.dumps({"best_model": best_name, "results": results}, indent=2))
print(f"Saved best model ({best_name}) to {MODEL_PATH}")
