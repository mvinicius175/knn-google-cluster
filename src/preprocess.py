import os
import ast
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "borg_traces_data.csv"
PROCESSED_FILE = DATA_DIR / "processed_data.npz"
SCALER_FILE = DATA_DIR / "scaler.joblib"


def gerar_dados_fallback(caminho_csv: Path) -> pd.DataFrame:
    """Geração de dados de fallback para se o dataset não estiver disponível."""
    np.random.seed(42)
    n_pontos = 2500
    t = np.arange(n_pontos)

    sazonalidade = 0.25 * np.sin(2 * np.pi * t / 288) + 0.15 * np.cos(2 * np.pi * t / 72)
    ruido = np.random.normal(0, 0.04, n_pontos)
    rajadas = np.zeros(n_pontos)
    indices_rajadas = np.random.choice(n_pontos, size=40, replace=False)
    rajadas[indices_rajadas] = np.random.uniform(0.2, 0.4, size=40)

    cpu = np.clip(0.40 + sazonalidade + ruido + rajadas, 0.05, 0.98)

    df = pd.DataFrame({
        "timestamp": t * 300,
        "cpu_usage": cpu
    })
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho_csv, index=False)
    print(f"[preprocess] Dataset ausente. Amostra de fallback gerada em: {caminho_csv}")
    return df


def carregar_dados() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return gerar_dados_fallback(DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    col_tempo = [c for c in df.columns if any(k in c.lower() for k in ["time", "timestamp", "date"])][0]
    nomes_cpu = ["cpu_usage", "average_usage", "avg_cpu_usage", "cpu", "usage", "load"]
    col_cpu = next((nome for nome in nomes_cpu if nome in df.columns), None)
    if col_cpu is None:
        col_cpu = next(
            c for c in df.columns
            if any(k in c.lower() for k in ["cpu", "usage", "load"])
            and "resource_request" not in c.lower()
            and "distribution" not in c.lower()
        )

    df = df[[col_tempo, col_cpu]].copy()
    df.columns = ["timestamp", "cpu_usage"]
    df["cpu_usage"] = df["cpu_usage"].map(_extrair_cpu)
    df = df.sort_values(by="timestamp").dropna().reset_index(drop=True)
    print(f"[preprocess] Dataset '{DATA_FILE.name}' carregado: {len(df)} amostras.")
    return df


def _extrair_cpu(valor) -> float:
    # Converte CPU numérica ou dicionário serializado em float.
    if isinstance(valor, dict):
        return valor.get("cpus", np.nan)
    if isinstance(valor, str):
        try:
            valor = ast.literal_eval(valor)
        except (SyntaxError, ValueError):
            return pd.to_numeric(valor, errors="coerce")
        if isinstance(valor, dict):
            return valor.get("cpus", np.nan)
    return pd.to_numeric(valor, errors="coerce")


def criar_janela_deslizante(serie: np.ndarray, window_size: int = 6):
    # Transforma a série temporal univariada em atributos supervisionados (lag features).
    X, y = [], []
    for i in range(len(serie) - window_size):
        X.append(serie[i: i + window_size])
        y.append(serie[i + window_size])
    return np.array(X), np.array(y)


def executar_preprocessamento(window_size: int = 6, test_ratio: float = 0.2):
    # Executa o pipeline de normalização, lag features e divisão treino/teste.
    df = carregar_dados()
    serie_original = df["cpu_usage"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    serie_normalizada = scaler.fit_transform(serie_original).flatten()

    X, y = criar_janela_deslizante(serie_normalizada, window_size=window_size)

    # Divisão sequencial para evitar vazamento de dados
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PROCESSED_FILE,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    joblib.dump(scaler, SCALER_FILE)

    print(f"[preprocess] Divisão concluída: {len(X_train)} treino | {len(X_test)} teste.")
    print(f"[preprocess] Dados processados salvos em: {DATA_DIR}")


if __name__ == "__main__":
    executar_preprocessamento()
