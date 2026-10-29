# Decisões - Trabalho Prático de Linguagem Natural

## Domínio escolhido

O domínio escolhido foi **Eleições 2026**.

## Perguntas antes e depois do pipeline

A tabela utiliza a configuração padrão atual da função: conversão para
minúsculas, remoção de pontuação, remoção de acentos, remoção de stopwords e
stemming.

| Pergunta antes da normalização                          | Tokens depois da normalização                      |
| ------------------------------------------------------- | -------------------------------------------------- |
| Qual é a data do primeiro turno?                        | `['dat', 'prim', 'turn']`                          |
| Qual é a data do segundo turno?                         | `['dat', 'segund', 'turn']`                        |
| Qual é o horário das votações?                          | `['horari', 'votaco']`                             |
| Quais são os candidatos do segundo turno à Presidência? | `['qual', 'candidat', 'segund', 'turn', 'presidenc']` |
| Como consultar o número de um candidato?                | `['consult', 'numer', 'candidat']`                 |
