# Decisões - Trabalho Prático de Linguagem Natural

## Domínio escolhido

O domínio escolhido foi **Eleições 2026**.

## Perguntas antes e depois do pipeline

A tabela utiliza a configuração padrão atual da função: conversão para
minúsculas, remoção de pontuação, remoção de acentos, remoção de stopwords e
stemming.

|  ID | Pergunta                                                   | Resposta                                                                                                                                                                                        | Tokens depois da normalização                               |
| --: | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
|   1 | Qual é a data do primeiro turno das Eleições 2026?         | O primeiro turno das Eleições 2026 será realizado em 4 de outubro de 2026.                                                                                                                      | `['dat', 'prim', 'turn', 'eleico', '2026']`                 |
|   2 | Qual é a data do segundo turno das Eleições 2026?          | O eventual segundo turno será realizado em 25 de outubro de 2026.                                                                                                                               | `['dat', 'segund', 'turn', 'eleico', '2026']`               |
|   3 | Qual é o horário da votação nas Eleições 2026?             | A votação ocorrerá das 8h às 17h, pelo horário de Brasília.                                                                                                                                     | `['horari', 'votaca', 'eleico', '2026']`                    |
|   4 | Quais cargos serão escolhidos no primeiro turno de 2026?   | No primeiro turno serão escolhidos presidente e vice-presidente da República, governador e vice-governador, dois senadores e seus suplentes, deputado federal e deputado estadual ou distrital. | `['qual', 'carg', 'escolh', 'prim', 'turn', '2026']`        |
|   5 | Quantas escolhas o eleitor fará na urna no primeiro turno? | O eleitor fará seis escolhas: deputado federal, deputado estadual ou distrital, primeiro senador, segundo senador, governador e presidente da República.                                        | `['quant', 'escolh', 'elei', 'far', 'urn', 'prim', 'turn']` |

## Decisão sobre negação

Nesta versão, palavras de negação não terão tratamento especial. Quando a
remoção de stopwords estiver ativada, termos como `não`, `nem` e `sem` serão
removidos conforme a lista padrão do NLTK.

Para manter a base coerente com essa decisão, as perguntas foram escritas de
forma afirmativa, sem depender de negações para definir seu sentido. Essa
decisão deverá ser revista caso perguntas negativas sejam incluídas futuramente.
