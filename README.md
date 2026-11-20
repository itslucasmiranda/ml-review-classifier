# ML Review Classifier

Pipeline reproduzível para classificar avaliações em inglês como positivas ou negativas. O projeto usa TF-IDF com regressão logística e compara seu desempenho na validação com um `DummyClassifier` que sempre prevê a classe mais frequente no treino.

O modelo está disponível no [Hugging Face](https://huggingface.co/itslucasmiranda/ml-review-classifier).

## Dados

O projeto usa a base **UCI Sentiment Labelled Sentences**. A preparação baixa o arquivo oficial, confere seu SHA-256, valida os registros, remove duplicatas e divide os dados por domínio e classe.

Das 3.000 frases originais, 2.979 foram mantidas após a remoção de 21 duplicatas. A divisão contém 1.784 frases de treino, 593 de validação e 602 de teste. A auditoria registrou zero IDs em comum entre as partições.

A origem, a licença, o protocolo experimental e as limitações estão em [docs/DATASET.md](docs/DATASET.md). As estatísticas e os hashes da execução de referência estão em [reports/data_audit.json](reports/data_audit.json).

## Resultados

### Validação

| Modelo | Acurácia | F1 macro |
| --- | ---: | ---: |
| `DummyClassifier` | 50,08% | 33,37% |
| TF-IDF + regressão logística | 79,76% | 79,76% |

O [relatório de validação](reports/baseline_metrics.json) também inclui precisão e revocação macro para os dois modelos.

### Teste

O modelo TF-IDF com regressão logística obteve **81,56% de acurácia** e **81,56% de F1 macro** em 602 avaliações. O [relatório final](reports/final_evaluation.json) inclui métricas por classe e a matriz de confusão.

## Como reproduzir

Use Python 3.12 ou superior e Git. No PowerShell, clone o repositório e instale o pacote com as dependências de desenvolvimento.

```powershell
git clone https://github.com/itslucasmiranda/ml-review-classifier.git
cd ml-review-classifier
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

O clone usa a branch padrão. Para reproduzir os resultados de um PR ainda aberto, faça checkout da branch do PR antes da instalação. Em Linux ou macOS, crie o ambiente com `python3 -m venv .venv` e use `.venv/bin/python` no lugar do executável do Windows.

Na raiz do repositório, execute os comandos nesta ordem.

```powershell
.\.venv\Scripts\python.exe -m review_classifier.prepare_data
.\.venv\Scripts\python.exe -m review_classifier.train
.\.venv\Scripts\python.exe -m review_classifier.evaluate
```

1. A preparação gera `train.jsonl`, `validation.jsonl`, `test.jsonl` e `audit.json` em `data/processed/uci-v1/`.
2. O treinamento ajusta os dois modelos com o conjunto de treino e compara suas métricas na validação. Ele salva o pipeline TF-IDF com regressão logística em `models/tfidf_logistic_regression.joblib` e grava as métricas em `reports/baseline_metrics.json`.
3. A avaliação carrega o modelo salvo, calcula as métricas no conjunto de teste e grava `reports/final_evaluation.json`.

Se `data/processed/uci-v1/` já existir, comece pelo treinamento para reutilizar os dados preparados. A preparação recusa sobrescrever uma pasta de saída existente. Os diretórios `data/raw/`, `data/processed/` e `models/` são ignorados pelo Git.

## Testes e CI

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Os testes usam dados sintéticos e downloads simulados. Eles cobrem preparação dos dados, separação das partições, treinamento em amostras pequenas, serialização e cálculo de métricas. O GitHub Actions executa a suíte em Python 3.12 e 3.14.

## Organização

- `src/review_classifier/data.py` reúne a aquisição, a validação e a divisão dos dados.
- `src/review_classifier/prepare_data.py` disponibiliza o comando de preparação.
- `src/review_classifier/train.py` treina os modelos e calcula as métricas de validação.
- `src/review_classifier/evaluate.py` avalia o modelo salvo no conjunto de teste.
- `tests/` contém os testes automatizados.
- `reports/` contém a auditoria e os relatórios de métricas.
- `pyproject.toml` define a instalação do pacote e suas dependências.

## Desenvolvimento

O [CONTRIBUTING.md](CONTRIBUTING.md) descreve o fluxo de issues, branches, testes e revisão de pull requests.

## Autor

[Lucas Miranda](https://github.com/itslucasmiranda)