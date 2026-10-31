# ML Review Classifier

> **Status: em desenvolvimento.** Projeto educacional e de portfólio. Métricas e modelos ainda serão implementados e avaliados.

## Objetivo
Construir um pipeline reproduzível de classificação de avaliações textuais, desde o tratamento de dados até a comparação de modelos e análise de erros.

## Plano experimental
1. Selecionar dataset público com licença e procedência documentadas.
2. Analisar distribuição das classes, duplicatas e valores ausentes.
3. Separar dados de treino e teste evitando vazamento.
4. Treinar baseline simples e modelo TF-IDF + classificador.
5. Avaliar precision, recall, F1-score e matriz de confusão.
6. Documentar erros, limitações e reprodutibilidade.

## Roadmap
- [ ] Definir dataset e problema de classificação
- [ ] Criar estrutura Python e pipeline de pré-processamento
- [ ] Implementar baseline e primeiro modelo
- [ ] Adicionar testes automatizados
- [ ] Comparar métricas e analisar erros
- [ ] Publicar relatório técnico e demonstração

## Stack planejada
Python, pandas, scikit-learn, pytest e GitHub Actions.

## Boas práticas
Sem dados pessoais, sem métricas inventadas e com separação clara entre treino e teste.

## Autor
[Lucas Miranda](https://github.com/itslucasmiranda)
