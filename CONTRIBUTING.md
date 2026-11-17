# Fluxo de desenvolvimento

As entregas devem ser pequenas o suficiente para revisar problema, implementação e evidência no mesmo PR.

1. **Issue:** registrar problema, escopo, critérios de aceite e limitações.
2. **Branch:** criar a partir da `main` atualizada. Exemplo: `feat/1-dataset-pipeline`.
3. **Implementação:** manter a mudança dentro do escopo da issue.
4. **Verificação:** rodar testes e o comando real afetado. Não confundir testes com dados sintéticos e resultados experimentais reais.
5. **Commits:** agrupar alterações relacionadas, com mensagens que expliquem a entrega.
6. **Pull request:** informar resultado, validação, riscos e `Closes #N` quando todos os critérios forem atendidos.
7. **Revisão e merge:** conferir o diff e os checks. Em trabalho individual, registrar a própria revisão sem alegar aprovação de outra pessoa.

## Mensagens de commit

- `feat: prepare verified review data and reproducible splits`
- `fix: preserve quoted review text during parsing`
- `test: cover duplicate and conflicting review labels`
- `docs: record dataset source and experiment protocol`
- `chore: initialize Python package`

Use o tipo correspondente à mudança. Não divida artificialmente uma alteração inseparável apenas para aumentar a contagem de commits.

## Comandos antes do PR

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
git status --short
```

Adicione arquivos específicos ao commit. Dados baixados, saídas processadas, modelos, ambientes virtuais e credenciais não devem entrar no Git. Estatísticas agregadas e relatórios experimentais revisados podem ser versionados.

## O que revisar neste projeto

- Existe separação entre dados de treino, validação e teste?
- O pré-processamento aprendido usa somente o treino?
- Duplicatas podem atravessar os conjuntos?
- Rótulos e metadados estão entrando indevidamente como atributos?
- As métricas vêm de uma execução identificável, com dados e parâmetros registrados?
- As limitações estão descritas sem prometer desempenho de produção?

Essas são convenções deste repositório. Não foram configuradas regras de proteção de branch que impeçam tecnicamente um push direto na `main`.
