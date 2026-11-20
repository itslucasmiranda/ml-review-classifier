# ML Review Classifier

Pipeline de classificação de sentimento em avaliações textuais, com foco em dados verificáveis, experimentos reproduzíveis e comparação de modelos.

**Status:** baseline de classificação implementado e avaliado.

O projeto utiliza TF-IDF + Logistic Regression para
classificar avaliações em inglês como positivas ou negativas.

Resultados no conjunto de teste (602 avaliações):
- Accuracy: 81,56%
- Macro F1-score: 81,56%

Modelo publicado:
https://huggingface.co/itslucasmiranda/ml-review-classifier

## Problema e escopo

A primeira versão classifica frases de avaliações **em inglês** como positivas ou negativas. A base escolhida é a UCI Sentiment Labelled Sentences. Origem, atribuição, licença, normalização, divisão dos conjuntos e limitações estão em [docs/DATASET.md](docs/DATASET.md).

Esta etapa utiliza apenas a biblioteca padrão do Python em execução. Não precisa de GPU, chave de API ou download de LLM.

## Executar no Windows

Use Python 3.12 ou superior e Git. No PowerShell, entre na pasta do repositório:

```powershell
git clone https://github.com/itslucasmiranda/ml-review-classifier.git
cd ml-review-classifier
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m review_classifier.prepare_data
.\.venv\Scripts\python.exe -m pytest -q
```

O comando acima segue a `main`. Para revisar uma entrega ainda não integrada, faça checkout da branch indicada no PR antes da instalação.

Não é necessário ativar o ambiente virtual, pois os comandos usam diretamente seu executável. Em Linux/macOS, crie o ambiente com `python3 -m venv .venv` e use `.venv/bin/python`.

## Preparação dos dados

```powershell
.\.venv\Scripts\python.exe -m review_classifier.prepare_data
```

O programa baixa o ZIP oficial (cerca de 84 KB), confere seu SHA-256, valida as linhas, remove duplicatas globais e gera:

- `data/processed/uci-v1/train.jsonl`
- `data/processed/uci-v1/validation.jsonl`
- `data/processed/uci-v1/test.jsonl`
- `data/processed/uci-v1/audit.json`

Os diretórios `data/raw/` e `data/processed/` são ignorados pelo Git. Depois do primeiro download, o cache verificado permite preparar outra saída sem conexão.

Uma pasta de saída existente é rejeitada para preservar o experimento. Para repetir em outra pasta:

```powershell
.\.venv\Scripts\python.exe -m review_classifier.prepare_data --output data/processed/uci-v1-repeat
```

A mesma entrada e seed geram os mesmos arquivos. A opção `--seed` define um novo experimento; não deve ser alterada para procurar uma divisão com métricas mais favoráveis.

## Auditoria inicial

| Medida | Resultado real |
|---|---:|
| Frases originais | 3.000 |
| Duplicatas removidas | 21 |
| Frases restantes | 2.979 |
| Treino / validação / teste | 1.784 / 593 / 602 |
| IDs compartilhados entre conjuntos | 0 |

A separação é aproximadamente 60/20/20 e estratificada por domínio e classe. [Relatório agregado completo](reports/data_audit.json).

Esses são resultados de auditoria dos dados. Nenhuma acurácia ou F1 de modelo foi calculada nesta entrega. A ausência de IDs compartilhados não exclui paráfrases ou textos relacionados sem identificadores de origem disponíveis.

## Testes e CI

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Os testes usam dados sintéticos e downloads simulados: validam parsing, checksum, remoção de duplicatas e conflitos, separação dos conjuntos, determinismo, hashes e preservação de saídas existentes. Eles não baixam a base real nem treinam modelos. O GitHub Actions executa a suíte em Python 3.12 e 3.14.

## Organização

| Caminho | Responsabilidade |
|---|---|
| `src/review_classifier/data.py` | Aquisição, validação, deduplicação, separação e auditoria |
| `src/review_classifier/prepare_data.py` | Interface de linha de comando |
| `tests/` | Testes offline |
| `docs/DATASET.md` | Contrato dos dados e protocolo experimental |
| `reports/data_audit.json` | Estatísticas e hashes da execução de referência |
| `pyproject.toml` | Instalação do pacote e dependências de desenvolvimento |

## Entregas

1. [Issue #1](https://github.com/itslucasmiranda/ml-review-classifier/issues/1): dados, estrutura Python, preparação reproduzível e testes iniciais.
2. [Issue #2](https://github.com/itslucasmiranda/ml-review-classifier/issues/2): baseline `DummyClassifier`, TF-IDF + regressão logística e seleção na validação.
3. [Issue #3](https://github.com/itslucasmiranda/ml-review-classifier/issues/3): comparação final, métricas, matriz de confusão e análise de erros.

Após essas entregas, documentar uma demonstração de predição e os resultados efetivamente medidos. Pandas e scikit-learn serão adicionados conforme a necessidade das etapas de treinamento e avaliação.

## Desenvolvimento

[CONTRIBUTING.md](CONTRIBUTING.md) define o fluxo issue → branch → implementação → testes → commits → PR → revisão → merge. A `main` concentra entregas revisadas; as convenções ainda não são impostas por regras de proteção de branch.

## Autor

[Lucas Miranda](https://github.com/itslucasmiranda)
