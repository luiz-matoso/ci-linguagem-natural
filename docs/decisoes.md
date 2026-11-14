# Decisões - Trabalho Prático de Linguagem Natural

As decisões da implementação da v1, a calibração do limiar e a análise de
erros estão em [dedisoes.md](../dedisoes.md), nome solicitado para esta etapa.
Este arquivo preserva o registro original da v0.

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

## v2 - embeddings densos

### Representação, modelo e reprodutibilidade

A v2 indexa somente as perguntas, como a v1, e mantém a mesma interface e o
mesmo desempate por ID crescente. Ela usa
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revisão
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, com
`sentence-transformers 5.7.0`, em CPU. Essa é a arquitetura sugerida no
enunciado, oferece suporte multilíngue e é menor que a alternativa avaliada.
Não há geração de texto nem uso de LLM.

Para conferir a escolha, comparamos
`sentence-transformers/distiluse-base-multilingual-cased-v2`, revisão
`bfe45d0732ca50787611c0fe107ba278c7f3f889`. Os quatro experimentos abaixo
usaram somente a validação. Cada configuração recebeu seu próprio limiar,
escolhido na mesma grade e pelo mesmo objetivo da v1: média entre acertos de
ranking aceitos no escopo e rejeições corretas fora do escopo.

| Modelo | Texto | Limiar | Acurácia@1 | MRR | Rejeição correta | Falsa rejeição | Objetivo |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM | original | **0,80** | 0,8333 | **0,9028** | **1,0000** | 0,3333 | **0,8333** |
| MiniLM | normalizado | 0,90 | 0,3333 | 0,5087 | 1,0000 | 0,6667 | 0,6250 |
| DistilUSE | original | 0,65 | 0,8333 | 0,8799 | 0,7500 | **0,0000** | 0,7917 |
| DistilUSE | normalizado | 0,75 | 0,3333 | 0,4640 | 0,8750 | 0,5000 | 0,5625 |

Escolhemos MiniLM com texto original porque teve o maior objetivo e o maior
MRR, além de rejeitar as 8 perguntas fora do escopo. O DistilUSE original
aceitou mais consultas válidas, mas rejeitou apenas 6/8 perguntas fora do
escopo. A validação tem só 20 itens; portanto, essa escolha é evidência local,
não uma conclusão geral sobre as arquiteturas.

### Ablação de normalização

O pipeline completo da v0 remove acentos, stopwords e aplica stemming. Nos
dois modelos, aplicá-lo reduziu Acurácia@1 na validação de 0,8333 para 0,3333.
No MiniLM, o MRR caiu de 0,9028 para 0,5087. Mantivemos o texto natural nos
dois lados da recuperação. A hipótese é que os Transformers foram treinados
com frases naturais e usam palavras funcionais, flexões e contexto; radicais
RSLP como `eleico` e a remoção de negações destroem pistas que o modelo sabe
representar. Este contraste funciona como a ablação exigida pelo enunciado.

### Cache de embeddings

Os 50 embeddings da base são calculados uma vez e persistidos em arquivos
NumPy compactados sob `.cache/embeddings/`. A chave SHA-256 inclui todos os
campos da base, textos efetivamente indexados, identificador e revisão do
modelo, versão da `sentence-transformers` e opções de normalização. Qualquer
mudança nesses elementos cria outro artefato; consultas reutilizam o cache e
calculam apenas um embedding. O cache é gerado automaticamente e não é
versionado. Metadados e dimensões são verificados na leitura; cache corrompido
gera erro explícito, em vez de produzir um ranking silenciosamente incorreto.

### Limiar da v2

A grade de 0,00 a 1,00, passo 0,05, foi executada exclusivamente nas 12
perguntas válidas e 8 fora do escopo de `data/validacao.csv`. O limiar 0,80
maximizou o objetivo em 0,8333: aceitou 8/12 acertos de ranking, rejeitou 8/8
casos fora do escopo e rejeitou falsamente 4/12 válidos. Em 0,65, a falsa
rejeição era zero, mas somente 6/8 casos fora do escopo eram rejeitados. A
escolha conservadora favorece não responder fora da base. Ela não corrige
erros de ranking e pode precisar de nova calibração se a base ou a população
de consultas mudar.

### Comparação final no teste

Modelo, normalização e limiar foram fixados antes de abrir o teste. O script
gerou automaticamente:

| Versão | Limiar | Acurácia@1 | MRR | Rejeição correta | Falsa rejeição |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 0,70 | 0,6667 (20/30) | 0,7414 | 1,0000 (5/5) | 0,6667 (20/30) |
| v2 | 0,80 | 0,8000 (24/30) | 0,8628 | 1,0000 (5/5) | 0,6000 (18/30) |

A v2 melhorou o ranking e reduziu a falsa rejeição, mas ainda recusou 18 das
30 perguntas válidas. Isso mostra a tensão entre rejeição correta e cobertura:
um embedding semanticamente melhor não elimina a necessidade de calibrar o
limiar, e o conjunto de validação pequeno favoreceu uma política conservadora.

### Análise de erros observados

Há 8 casos em que o primeiro item da v2 é correto e o da v1 é incorreto. Três
exemplos representativos são:

| Pergunta | Esperado | v1 | v2 | Hipótese |
| --- | ---: | ---: | ---: | --- |
| Que funções públicas serão eleitas no primeiro turno? | 4 | 1 | 4 | O embedding aproxima `funções públicas` de `cargos`; o TF-IDF se apoia em `primeiro turno` e favorece a data. |
| Como faço meu cadastro eleitoral pela primeira vez? | 15 | 35 | 15 | A v2 relaciona cadastro ao primeiro título; a v1 associa literalmente `primeira vez` ao primeiro acesso ao aplicativo. |
| Uma certidão de nascimento serve para me identificar na eleição? | 31 | 33 | 31 | A v2 captura a intenção de documento inválido; a v1 encontra vocabulário eleitoral genérico e aponta serviços do e-Título. |

Há 4 casos no sentido contrário, portanto foi possível registrar ao menos três:

| Pergunta | Esperado | v1 | v2 | Hipótese |
| --- | ---: | ---: | ---: | --- |
| Quais requisitos preciso cumprir pra transferir o título? | 19 | 19 | 18 | Os IDs 18 e 19 são semanticamente quase duplicados; a v1 preserva a pista lexical `requisitos/condições`, enquanto a v2 aproxima a ação geral de transferir. |
| Quais documentos com foto eu posso levar no dia de votar? | 28 | 28 | 30 | A menção a `foto` aproxima a v2 da pergunta específica sobre documento vencido; o TF-IDF mantém o foco na lista de documentos. |
| Minha identidade está vencida. Ela ainda vale para votar? | 30 | 30 | 12 | A v1 casa os termos raros `vencida` e `votar`; o embedding perde a condição documental e aproxima a pergunta de elegibilidade do voto. |

Esses casos sustentam a leitura esperada no enunciado: embeddings lidam melhor
com sinonímia e reformulações, enquanto TF-IDF pode vencer quando uma pista
lexical rara ou uma distinção muito fina entre perguntas próximas é decisiva.
Os IDs 18/19 também mostram que respostas/perguntas semanticamente próximas
continuam sendo pares distintos; o MRR usa o ID esperado, sem fundi-los.
