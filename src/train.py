from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PROCESSED_FILE = DATA_DIR / "processed_data.npz"
MODEL_FILE = RESULTS_DIR / "best_knn_model.joblib"


def treinar_modelo():
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError("Artefatos de pré-processamento não encontrados. Execute src/preprocess.py primeiro.")

    dados = np.load(PROCESSED_FILE)
    X_train, y_train = dados["X_train"], dados["y_train"]

    print(f"[train] Calibrando KNN com {len(X_train)} amostras via TimeSeriesSplit...")

    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 13, 17],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }

    grid_search = GridSearchCV(
        estimator=KNeighborsRegressor(),
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    melhor_modelo = grid_search.best_estimator_
    print(f"[train] Hiperparâmetros ótimos encontrados: {grid_search.best_params_}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(melhor_modelo, MODEL_FILE)
    print(f"[train] Modelo salvo em: {MODEL_FILE}")


if __name__ == "__main__":
    treinar_modelo()
