from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

PROCESSED_FILE = DATA_DIR / "processed_data.npz"
SCALER_FILE = DATA_DIR / "scaler.joblib"
MODEL_FILE = RESULTS_DIR / "best_knn_model.joblib"
METRICS_FILE = RESULTS_DIR / "metrics.txt"
PREDICTION_PLOT = RESULTS_DIR / "prediction.png"


def avaliar_modelo():
    if not (PROCESSED_FILE.exists() and MODEL_FILE.exists() and SCALER_FILE.exists()):
        raise FileNotFoundError("Artefatos ausentes. Execute src/preprocess.py e src/train.py primeiro.")

    dados = np.load(PROCESSED_FILE)
    X_test, y_test = dados["X_test"], dados["y_test"]

    modelo_knn = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    # Inferência com KNN e Baseline (Persistência temporal: y_pred(t) = y(t-1))
    y_pred_knn_norm = modelo_knn.predict(X_test)
    y_pred_base_norm = X_test[:, -1]

    # Desnormalização para avaliação na escala real da métrica
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_knn = scaler.inverse_transform(y_pred_knn_norm.reshape(-1, 1)).flatten()
    y_pred_base = scaler.inverse_transform(y_pred_base_norm.reshape(-1, 1)).flatten()

    def obter_metricas(real, pred):
        rmse = np.sqrt(mean_squared_error(real, pred))
        mae = mean_absolute_error(real, pred)
        mape = mean_absolute_percentage_error(real, pred) * 100
        r2 = r2_score(real, pred)
        return rmse, mae, mape, r2

    rmse_knn, mae_knn, mape_knn, r2_knn = obter_metricas(y_test_real, y_pred_knn)
    rmse_b, mae_b, mape_b, r2_b = obter_metricas(y_test_real, y_pred_base)

    # 1. Salvar relatório em results/metrics.txt
    relatorio = (
        "=================================================================\n"
        "AVALIAÇÃO DE DESEMPENHO: PREDIÇÃO DE RECURSOS COMPUTACIONAIS (KNN)\n"

        f"Hiperparâmetros Selecionados: {modelo_knn.get_params()}\n\n"
        f"{'Métrica':<12} | {'Baseline (Persistência)':<24} | {'KNN Otimizado':<15}\n"
        f"{'-'*58}\n"
        f"{'RMSE':<12} | {rmse_b:<24.5f} | {rmse_knn:<15.5f}\n"
        f"{'MAE':<12} | {mae_b:<24.5f} | {mae_knn:<15.5f}\n"
        f"{'MAPE (%)':<12} | {mape_b:<23.2f}% | {mape_knn:<14.2f}%\n"
        f"{'R²':<12} | {r2_b:<24.5f} | {r2_knn:<15.5f}\n"
        "=================================================================\n"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write(relatorio)
    print(f"[evaluate] Relatório gravado em: {METRICS_FILE}")
    print(relatorio)

    # 2. Gerar gráfico em results/prediction.png
    janela_plot = 150
    plt.figure(figsize=(13, 7))

    plt.subplot(2, 1, 1)
    plt.plot(y_test_real[-janela_plot:], label="Demanda Real (Google Borg Trace)", color="black", linewidth=1.5)
    plt.plot(y_pred_knn[-janela_plot:], label=f"Predição KNN (k={modelo_knn.n_neighbors})", color="#1f77b4", linestyle="--", linewidth=1.6)
    plt.plot(y_pred_base[-janela_plot:], label="Baseline (Persistência)", color="#ff7f0e", linestyle=":", alpha=0.7)
    plt.title("Estudo de Caso: Predição de Carga de CPU em Cluster Borg (Google)", fontsize=11, fontweight="bold")
    plt.ylabel("Uso de CPU")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(2, 1, 2)
    erro_knn = np.abs(y_test_real[-janela_plot:] - y_pred_knn[-janela_plot:])
    erro_base = np.abs(y_test_real[-janela_plot:] - y_pred_base[-janela_plot:])
    plt.fill_between(range(janela_plot), erro_knn, color="#1f77b4", alpha=0.3, label="Erro Absoluto (KNN)")
    plt.plot(erro_base, color="#ff7f0e", linestyle=":", label="Erro Absoluto (Baseline)", alpha=0.8)
    plt.xlabel("Passos Temporais (Amostras de Teste)")
    plt.ylabel("Erro Absoluto (|y - ŷ|)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT, dpi=300)
    plt.close()
    print(f"[evaluate] Gráfico salvo com sucesso em: {PREDICTION_PLOT}")


if __name__ == "__main__":
    avaliar_modelo()
