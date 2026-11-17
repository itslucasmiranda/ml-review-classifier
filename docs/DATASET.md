# Contrato dos dados e plano experimental

## Problema

Classificar uma frase de avaliação em inglês como negativa (`0`) ou positiva (`1`). Esta primeira entrega prepara os dados; nenhum classificador foi treinado. Avaliações neutras e textos em português estão fora do escopo inicial.

## Fonte e atribuição

- Base: **Sentiment Labelled Sentences**, UCI, identificador 331.
- Fonte oficial: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
- Referência: Kotzias, D. (2015). *Sentiment Labelled Sentences* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57604
- Artigo de origem: Kotzias, D.; Denil, M.; de Freitas, N.; Smyth, P. (2015). *From Group to Individual Labels Using Deep Features*. KDD.
- Licença declarada pela UCI: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). A atribuição acima deve acompanhar redistribuições de dados derivados. Essa licença se refere à base, não atribui automaticamente uma licença ao código deste repositório.
- Fonte consultada e arquivo obtido em 17/11/2026.

O arquivo contém 3.000 frases, divididas entre avaliações de produtos (Amazon), filmes (IMDb) e restaurantes (Yelp). Originalmente, cada domínio possui 500 frases de cada classe. A seleção original buscou frases claramente positivas ou negativas, excluindo neutralidade.

## Versão fixada

Arquivo: `https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip`

Tamanho observado: **84.188 bytes**.

SHA-256: `afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343`.

O download e o cache são verificados antes do processamento. Se a origem mudar, a execução falha; uma nova versão exige análise e atualização deliberada do hash. O programa lê somente três membros específicos do ZIP, sem extrair arquivos arbitrários.

## Leitura e qualidade

Cada registro tem texto, tabulação e rótulo. A leitura usa a última tabulação como separador e somente LF como terminador de registro. Aspas pertencem ao texto, não delimitam campos CSV. Separadores Unicode podem aparecer dentro das frases do IMDb e não devem criar registros adicionais.

Texto vazio, rótulo fora de `0/1`, domínio inválido e linhas malformadas interrompem a execução. O texto original é preservado nas saídas; a normalização abaixo é usada apenas para identificar equivalências.

1. Normalizar Unicode com NFKC.
2. Aplicar `casefold`.
3. Colapsar espaços em branco.
4. Calcular o SHA-256 do texto normalizado.
5. Agrupar globalmente, inclusive entre domínios.
6. Para rótulos iguais, manter a primeira ocorrência ordenada por domínio, linha e texto original.
7. Se um grupo contiver rótulos conflitantes, excluir todas as suas ocorrências e registrar as contagens.

A escolha da ocorrência representativa é determinística. Duplicatas entre domínios são atribuídas a um único domínio representativo, o que pode alterar ligeiramente sua distribuição. Pontuação e acentos não são removidos.

## Separação dos conjuntos

Após a deduplicação, os registros são agrupados por **domínio e classe**. Dentro de cada grupo, são ordenados pelo hash de `seed:text_id`, com desempate pelo identificador. A seed padrão é `42`.

- Treino: `floor(0,6 × n)` registros por grupo.
- Validação: `floor(0,2 × n)` por grupo.
- Teste: todos os registros restantes.

Cada grupo precisa ter pelo menos cinco registros. As proporções são aproximadas devido ao arredondamento. A mesma entrada e seed produzem a mesma divisão mesmo que a ordem de entrada mude. Alterar o corpus ou a regra de normalização define um novo experimento e pode alterar a divisão.

Os IDs normalizados não podem aparecer em mais de um conjunto. Isso evita sobreposição segundo essa regra específica; não prova ausência de todos os tipos de vazamento.

## Auditoria da primeira execução

O arquivo [reports/data_audit.json](../reports/data_audit.json) contém somente estatísticas, configuração e hashes dos arquivos derivados. Os textos ficam fora do Git.

| Etapa | Registros |
|---|---:|
| Originais | 3.000 |
| Duplicatas equivalentes removidas | 21 |
| Registros removidos por conflito de rótulos | 0 |
| Após limpeza | 2.979 |
| Treino | 1.784 |
| Validação | 593 |
| Teste | 602 |
| Sobreposição de IDs entre conjuntos | 0 |

A auditoria foi executada com o arquivo real. Esses números descrevem os dados, não o desempenho de um modelo.

## Contrato das saídas

Cada arquivo `train.jsonl`, `validation.jsonl` e `test.jsonl` tem um objeto JSON por linha:

| Campo | Finalidade |
|---|---|
| `id` | Hash do texto normalizado, usado para rastreamento e verificação de sobreposição |
| `text` | Texto original usado futuramente como entrada do modelo |
| `label` | Alvo: 0 negativo, 1 positivo |
| `domain` | Origem, usada para estratificação e análise por domínio |
| `source_line` | Linha no arquivo original, para rastreabilidade |

`id`, `domain`, `source_line` e `label` não devem ser usados como atributos textuais de entrada do classificador. `audit.json` registra hashes SHA-256 dos JSONL, distribuição por classe/domínio, remoções e seed. Finais de linha e serialização são explícitos para permitir comparação entre sistemas.

## Protocolo para as próximas entregas

- Ajustar vocabulário e IDF exclusivamente no treino, dentro de um `Pipeline`.
- Treinar um `DummyClassifier` como referência e comparar com TF-IDF + regressão logística.
- Escolher configurações usando treino e validação. Não abrir erros individuais do teste para escolher hiperparâmetros.
- Fixar o candidato antes da avaliação final no teste.
- Publicar precision, recall e F1 por classe, macro-F1, matriz de confusão e análise de erros.
- Registrar seed, versões, hashes dos dados e parâmetros de cada experimento.

## Limitações

O corpus é pequeno, antigo, em inglês e balanceado por construção. Resultados nele não demonstram desempenho em comentários de clientes brasileiros nem na distribuição real de uma empresa. Não há classe neutra.

Não há identificadores de autores ou da avaliação original no formato fornecido. Frases relacionadas podem permanecer em conjuntos diferentes, e a deduplicação não identifica todas as paráfrases. A avaliação terá mistura dos mesmos domínios em todos os conjuntos; não medirá generalização para um domínio inteiramente novo.

Não é feita coleta de dados de clientes nem enriquecimento com informações pessoais. Os textos públicos originais não são republicados neste repositório. Antes de usar outra base, revisar procedência, licença e presença de informações pessoais.
