# Predição de uso de CPU com KNN

Projeto de previsão de séries temporais para estimar o uso de CPU em traces do cluster Google Borg. A solução transforma uma série temporal univariada em exemplos supervisionados com janelas de valores anteriores e utiliza um `KNeighborsRegressor` para prever o próximo valor.

## Objetivo técnico

O pipeline recebe um arquivo CSV com dados temporais e:

- identifica as colunas de tempo e uso de CPU;
- converte os valores de CPU para um formato numérico;
- normaliza a série com `MinMaxScaler`;
- cria atributos temporais por meio de janelas deslizantes;
- separa os exemplos sequencialmente em treino e teste;
- seleciona hiperparâmetros do KNN com validação cruzada temporal;
- compara as previsões do modelo com um baseline de persistência;
- salva os artefatos do processamento, modelo, métricas e visualização.

## Estrutura do repositório

```text
.
├── data/
│   ├── .gitkeep
│   └── borg_traces_data.csv       # entrada local, não versionada
├── results/
│   ├── .gitkeep
│   ├── best_knn_model.joblib      # modelo treinado, não versionado
│   ├── metrics.txt                # relatório gerado, não versionado
│   └── prediction.png             # gráfico gerado, não versionado
├── src/
│   ├── preprocess.py              # preparação da série temporal
│   ├── train.py                   # busca e treinamento do KNN
│   └── evaluate.py                # avaliação e geração dos resultados
├── .gitignore
├── requirements.txt
└── README.md
```


## Requisitos

- Python 3.9 ou superior

As bibliotecas utilizadas estão declaradas em `requirements.txt`:

- NumPy: operações numéricas e armazenamento dos arrays;
- pandas: leitura e tratamento do CSV;
- scikit-learn: normalização, modelo, validação cruzada e métricas;
- Matplotlib: geração do gráfico de avaliação;
- joblib: persistência do scaler e do modelo.

## Configuração do ambiente

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No Linux ou macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dados de entrada

O arquivo de entrada esperado é:

[data/borg_traces_data.csv](https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample)

O carregamento procura automaticamente:

- uma coluna de tempo cujo nome contenha `time`, `timestamp` ou `date`;
- uma coluna de CPU com um dos nomes preferenciais `cpu_usage`, `average_usage`, `avg_cpu_usage`, `cpu`, `usage` ou `load`.

Se não encontrar um nome preferencial, o script procura uma coluna cujo nome contenha `cpu`, `usage` ou `load`, ignorando colunas relacionadas a `resource_request` e `distribution`.

A coluna de CPU pode conter números ou strings representando dicionários. Para dicionários, o campo utilizado é `cpus`. Os dados são ordenados pelo tempo, valores inválidos são convertidos em ausentes e linhas inválidas são removidas.

Quando o CSV não existe, `preprocess.py` gera uma série sintética determinística como fallback e grava o arquivo no caminho esperado.

## Pipeline de processamento

### 1. Pré-processamento

Arquivo: `src/preprocess.py`

O procedimento padrão é `executar_preprocessamento(window_size=6, test_ratio=0.2)`.

1. Carrega o CSV ou cria o fallback.
2. Normaliza o uso de CPU para o intervalo `[0, 1]` com `MinMaxScaler`.
3. Cria janelas de seis observações consecutivas:

   ```text
   X[i] = [y[i], y[i+1], ..., y[i+5]]
   alvo[i] = y[i+6]
   ```

4. Divide os exemplos sequencialmente: 80% para treino e 20% para teste.
5. Salva os arrays e o scaler em `data/`.

Os dados processados são armazenados em `data/processed_data.npz` com as chaves:

```text
X_train, X_test, y_train, y_test
```

O scaler é armazenado em `data/scaler.joblib` e reutilizado durante a avaliação para converter as previsões de volta à escala original.

### 2. Treinamento

Arquivo: `src/train.py`

O treinamento utiliza `KNeighborsRegressor` e `TimeSeriesSplit` com cinco divisões. A busca em grade avalia:

```python
{
    "n_neighbors": [3, 5, 7, 9, 13, 17],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}
```

O critério de seleção é `neg_root_mean_squared_error`. O melhor estimador é salvo em `results/best_knn_model.joblib`.

### 3. Avaliação

Arquivo: `src/evaluate.py`

O script carrega os dados de teste, o modelo e o scaler. Ele calcula:

- previsão do KNN;
- baseline de persistência, usando o último valor da janela como previsão;
- RMSE, MAE, MAPE e R² para as duas abordagens.

Os valores são desnormalizados antes do cálculo das métricas. O relatório textual é salvo em `results/metrics.txt` e o gráfico das previsões e erros absolutos é salvo em `results/prediction.png`.

## Execução

Execute os comandos a partir da raiz do repositório e respeite a ordem abaixo:

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

Cada etapa depende dos artefatos gerados pela etapa anterior. Para regenerar o pipeline, execute novamente os três comandos.

## Artefatos gerados

| Caminho | Descrição |
| --- | --- |
| `data/processed_data.npz` | Arrays de treino e teste gerados pelas janelas temporais |
| `data/scaler.joblib` | Instância persistida do `MinMaxScaler` |
| `results/best_knn_model.joblib` | Melhor `KNeighborsRegressor` selecionado pela busca |
| `results/metrics.txt` | Métricas do KNN e do baseline |
| `results/prediction.png` | Visualização das previsões e dos erros absolutos |

## Observações de implementação

- A divisão de treino e teste preserva a ordem temporal e não embaralha os exemplos.
- O baseline usa `X_test[:, -1]`, representando a persistência do último valor observado.
- O mesmo scaler salvo no pré-processamento é usado para desnormalizar valores reais e previsões.
- O MAPE pode ser instável quando os valores reais são próximos de zero; por isso, as demais métricas também são calculadas e reportadas.
