# Decisões da v1 - FAQ Eleições 2026

## 1. Escopo e organização

- Implementar a v1 descrita na seção 4.2 de `PjBL 2.pdf`, junto dos requisitos
  técnicos necessários para executá-la e avaliá-la: TF-IDF, cosseno próprio,
  ranking completo, rejeição, terminal e calibração separada do teste.
- Preservar a v0 e seus parâmetros. Em `normalizacao.py`, acrescentar apenas
  comentários explicativos, sem mudar o processamento.
- Manter módulos na raiz e arquivos em `data/`, seguindo a organização já
  existente. Não mover tudo para `src/` e `dados/`, pois isso não é necessário
  para separar as responsabilidades solicitadas pelo enunciado.
- Separar leitura da base (`base.py`), recuperação (`chatbot_tfidf.py`),
  terminal (`chat.py`) e avaliação (`avaliacao.py`). Não criar serviços,
  banco de dados, interface web, hierarquia de classes ou infraestrutura extra.
- Usar `dedisoes.md` na raiz, com a grafia solicitada. Acrescentar uma
  referência em `docs/decisoes.md`, preservando o histórico da v0.
- Deixar v2, embeddings, comparação entre versões e bônus para outra etapa.
  Não gerar resultados fictícios para uma versão ainda inexistente.

## 2. Base de conhecimento

- Reutilizar os 50 pares e as fontes de `data/base.csv` sem alterar o conteúdo
  eleitoral. Esta etapa implementa recuperação; não atualiza informações
  pela internet. Listar as fontes existentes também no README.
- Ler CSV com `csv.DictReader`, da biblioteca padrão, sem adicionar pandas.
  Usar `utf-8-sig` para aceitar o BOM presente na base original e também CSV
  sem BOM. Escrever novos arquivos em UTF-8.
- Representar cada par por uma `dataclass` imutável, `ParFAQ`, com ID inteiro,
  pergunta, resposta e fonte. Remover somente espaços nas extremidades ao ler.
- Exigir as quatro colunas e valores preenchidos, IDs positivos e únicos,
  e pelo menos um par. Não exigir 50 pares no carregador: bases pequenas são
  úteis nos testes. A base entregue permanece com 50 pares.
- Falhar com uma mensagem controlada na inicialização quando houver erro na
  base. Não ignorar linhas inválidas, para evitar associações incorretas.
- Resolver o caminho padrão a partir do arquivo do módulo. A API funciona
  mesmo quando o diretório atual não é a raiz do projeto. Caminhos informados
  por `--base` continuam relativos ao diretório de quem executa o comando.

## 3. Normalização herdada da v0

- Usar a mesma função `normalizar` nas perguntas da base e nas consultas,
  com todas as etapas padrão ativadas: tokenização NLTK em português,
  minúsculas, remoção de tokens de pontuação, acentos, stopwords e stemming.
- A tokenização separa unidades de texto; minúsculas e remoção de acentos
  reduzem variações de escrita; a filtragem de pontuação retira tokens sem
  caracteres alfanuméricos; stopwords reduzem palavras funcionais; RSLP
  aproxima algumas flexões sem precisar baixar um modelo de lematização.
  Esses são os motivos para reutilizar o pipeline, não uma afirmação de que
  cada etapa sempre melhora a recuperação.
- Manter o tratamento existente de pontuação: descartar tokens inteiramente
  não alfanuméricos, sem reescrever caracteres internos de tokens mistos.
  Preservar números, pois anos e turnos são relevantes para este domínio.
- Manter a lista padrão de stopwords e a decisão anterior sobre negação.
  Não adicionar tratamento de `não`, `nem` ou `sem` nesta etapa. Perguntas
  negativas podem perder informação; o limiar não corrige essa perda.
- Não criar exceções para siglas ou termos como `e-Título`: eles passam pelas
  mesmas etapas. Caixa e acentos deixam de distinguir termos. O RSLP também
  pode reduzir termos de forma inadequada ou deixar flexões relacionadas
  com radicais diferentes, como `eleição` e `eleições`.
- Permitir repassar opções da v0 pelo parâmetro `opcoes_normalizacao` da v1,
  para viabilizar uma futura ablação. Não escolher opções com base no teste.
  O limiar registrado corresponde apenas à configuração padrão.
- Preservar o quadro de cinco perguntas antes/depois em `docs/decisoes.md`,
  pois o pipeline não mudou.

## 4. Representação TF-IDF

- Usar `TfidfVectorizer`, permitido pelo enunciado. Ajustar vocabulário e IDF
  **somente nas perguntas da base**. Não indexar respostas nem fontes.
  Isso segue a descrição da v1 e evita que respostas extensas dominem a busca.
- Passar `normalizar` como analisador do vetorizador, via `functools.partial`.
  Assim, o scikit-learn recebe diretamente os tokens da v0, sem uma segunda
  tokenização. Desativar a conversão adicional para minúsculas e o padrão
  de tokenização do vetorizador.
- Usar unigramas, frequência bruta, IDF suavizado e normalização L2, padrões
  do vetorizador. Para N perguntas, o IDF é
  `log((1 + N) / (1 + df(t))) + 1`. Não usar TF sublinear, bigramas,
  dicionário de sinônimos, correção ortográfica ou expansão de consultas.
  A configuração simples oferece uma referência para comparação futura.
- Ajustar o vetorizador e calcular os vetores da base uma vez por instância.
  Consultas usam apenas `transform`; não alteram o vocabulário nem o IDF.
- Converter a matriz da base para um array NumPy denso em memória. Com
  somente 50 perguntas isso simplifica a fórmula do cosseno. Não salvar
  cache em disco nesta versão; a exigência de cache é da v2. Bases grandes
  justificariam manter a matriz esparsa e adaptar o cálculo.
- Informar um erro de vocabulário inválido se todas as perguntas ficarem
  vazias após a normalização.

## 5. Cosseno, ranking e contrato

- Implementar `similaridade_cossenos` com operações NumPy: produto matricial,
  normas L2 e divisão pela multiplicação das normas. Não chamar o cosseno do
  scikit-learn no código de produção, mesmo com vetores já normalizados.
- A função recebe um vetor de consulta e uma matriz com um documento por
  linha. Validar dimensões para impedir uma comparação incompatível.
- Quando qualquer norma é zero, definir o respectivo cosseno como zero com
  `np.divide(..., where=...)`, sem divisão por zero ou `NaN`.
- Limitar o resultado numérico ao intervalo [-1, 1] para compensar pequenos
  erros de ponto flutuante. Com TF-IDF, os escores são não negativos.
- Conferir a implementação contra `sklearn.metrics.pairwise.cosine_similarity`
  em testes automatizados, com tolerância absoluta e relativa de `1e-12`.
  Cobrir vetores aleatórios com sinais, vetores nulos e a matriz TF-IDF real.
- Retornar `Resultado(resposta, id, escore, ranking)`, conforme a interface
  do enunciado. Usar `dataclass` para acesso por atributo sem criar um
  protocolo ou uma classe abstrata antes de existir a v2.
- Ordenar todos os pares por escore decrescente e, em empate exato, por ID
  crescente. Não arredondar antes de ordenar nem aplicar tolerância a empates.
  A ordem das linhas do CSV não determina o desempate.
- Manter pares distintos mesmo que suas respostas sejam iguais. O ID
  identifica a pergunta da base, e fundir pares alteraria o ranking e o MRR.
  Na avaliação, só o ID esperado conta como acerto; uma resposta duplicada
  vinculada a outro ID não recebe crédito automático.
- Rejeitar com ID `-1` e a mensagem `Desculpe, não entendi. Pode reformular?`.
  Mesmo na rejeição, devolver o melhor escore e o ranking completo, inclusive
  escores zero, para permitir cálculo de MRR e análise do limiar.

## 6. Regra de rejeição e escolha do limiar

- Aceitar somente quando `melhor_escore > limiar`. Igualdade é rejeição,
  seguindo a palavra "supere" no enunciado. Validar o limiar no intervalo
  [0, 1], recusando também `NaN` e infinitos. Com limiar zero, consultas
  inteiramente desconhecidas continuam rejeitadas.
- Tornar o limiar configurável no construtor e por `--limiar` no terminal.
- Elaborar `data/validacao.csv` com 12 paráfrases do escopo e 8 consultas
  sem resposta na base, incluindo perguntas com sobreposição de palavras.
  Perguntas sobre outro assunto eleitoral também são consideradas fora do
  escopo quando não há resposta correspondente na base.
- Separar esse conjunto do teste: não repetir perguntas literais entre os
  dois arquivos nem copiar perguntas literais da base.
- Examinar 21 limiares: 0,00; 0,05; ...; 1,00. Maximizar a média entre
  **acertos aceitos no escopo** e **rejeições corretas fora do escopo**,
  dando o mesmo peso aos dois grupos apesar dos tamanhos diferentes.
  Um acerto aceito exige primeiro ID correto e escore acima do limiar.
- Em empate nesse objetivo, escolher o menor limiar para favorecer menor
  falsa rejeição. Fixar grade e critério antes de executar a avaliação final.
- Rodar `python avaliacao.py --calibrar` usando somente a validação e salvar
  todos os candidatos em `resultados/calibracao.csv`. A rotina de calibração
  não abre o teste nem modifica automaticamente o código.
- Adotar **0,70** como padrão, a partir desse resultado, antes de avaliar o
  teste. A escolha é conservadora e tem custo alto de falsa rejeição.

| Limiar | Acertos aceitos no escopo | Rejeição correta | Falsa rejeição | Objetivo |
| ---: | ---: | ---: | ---: | ---: |
| 0,50 | 2/12 | 2/8 | 3/12 | 0,2083 |
| 0,65 | 1/12 | 6/8 | 10/12 | 0,4167 |
| **0,70** | **1/12** | **8/8** | **10/12** | **0,5417** |
| 0,75 | 1/12 | 8/8 | 10/12 | 0,5417 |
| 1,00 | 0/12 | 8/8 | 12/12 | 0,5000 |

A Acurácia@1 da validação é 2/12 e o MRR é 0,4421. Apenas duas consultas
ficam acima de 0,70; uma é correta e a outra recupera o ID 45 no lugar do
46. Portanto, o limiar não resolve erros semânticos nem garante que toda
resposta aceita esteja correta. O conjunto pequeno e difícil produziu um
padrão bastante restritivo; uma ampliação futura deve usar nova validação,
sem ajustar o limiar para melhorar os números do teste atual.

## 7. Entradas e interface de terminal

- Oferecer `python chat.py --versao v1`, com argumentos para limiar e base.
  Usar `argparse` da biblioteca padrão. Aceitar apenas `v1` por enquanto.
- Exibir ID, melhor escore com três casas e resposta. Esse arredondamento é
  apenas visual; a API e os CSVs preservam os números usados nos cálculos.
- Encerrar com `sair` sem diferenciar maiúsculas, Ctrl+C ou fim de arquivo.
  Pergunta vazia não encerra a conversa: recebe rejeição.
- Strings vazias, apenas espaços, pontuação, emojis e vocabulário totalmente
  desconhecido produzem escore zero e ranking completo. Texto em outro idioma
  passa pelo mesmo pipeline: pode ser rejeitado ou encontrar palavras em
  comum. Não adicionar detector de idioma ou prometer rejeição automática.
- Fixar o limite de 10.000 caracteres por consulta. Rejeitar antes da
  tokenização, com ranking de zeros, sem truncar: cortar o texto poderia
  remover uma condição importante. O terminal também orienta a encurtar.
- Mostrar erro controlado para arquivos inválidos, recursos NLTK ausentes
  ou limiar inválido durante a inicialização. Não baixar recursos durante
  uma conversa e não esconder defeitos inesperados com um `except` genérico.
- O contrato público recebe `str`, como especificado; não converter objetos
  arbitrários em perguntas.

## 8. Avaliação e resultados

- Incluir `data/teste.csv` com 30 paráfrases e 5 consultas fora do escopo.
  Variar registro, ordem, sinônimos e escrita informal, incluindo o erro
  `pagra` e a grafia `e-titulo`. Não alterar os dados nem o limiar após
  observar os resultados desta execução.
- Esses conjuntos foram preparados nesta implementação com acesso à base,
  sem autoria independente. Essa limitação pode introduzir viés. Há poucos
  exemplos de rejeição e nem todo ID da base aparece no teste; os números
  são uma avaliação inicial, não uma estimativa ampla de desempenho.
- Validar colunas, perguntas preenchidas, conjuntos não vazios e IDs
  esperados existentes. `-1` representa ausência de resposta. Exigir pelo
  menos uma consulta de cada grupo para não relatar métricas indefinidas.
- Calcular Acurácia@1 sobre o primeiro ID do ranking das 30 consultas do
  escopo, mesmo quando o limiar rejeita. Calcular MRR pela posição do ID
  esperado no ranking completo. Rejeição correta usa só as 5 consultas fora
  do escopo; falsa rejeição usa só as 30 consultas válidas.
- Gravar as quatro métricas em `resultados/comparacao.csv`, com versão e
  limiar. Neste momento, há apenas a linha v1. Não inventar uma comparação
  com a v2. Cada execução substitui esse arquivo; ele representa o resultado
  do último comando de avaliação, sem acumular execuções antigas.
- Gerar os dois CSVs de resultados por script, sem editar seus valores à mão.

| Versão | Limiar | Acurácia@1 | MRR | Rejeição correta | Falsa rejeição |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 0,70 | 0,6667 (20/30) | 0,7414 | 1,0000 (5/5) | 0,6667 (20/30) |

O ranking acerta 20 consultas, mas só 10 das 30 recebem uma resposta aceita.
Todas as 10 aceitas estão corretas neste teste. Isso não equivale a uma
acurácia de atendimento de 66,67%: esse percentual descreve o ranking,
conforme a definição do enunciado, antes de aplicar a rejeição.

### Exemplos de erro e interpretação

| Pergunta | ID esperado | Primeiro ID | Escore | Saída | Hipótese |
| --- | ---: | ---: | ---: | --- | --- |
| Quando será a segunda rodada da eleição deste ano de 2026? | 2 | 2 | 0,4051 | Rejeição | A paráfrase perde os termos `data`, `segundo` e `turno`; o melhor resultado ainda é correto, mas fica abaixo do limiar. |
| Que funções públicas serão eleitas no primeiro turno? | 4 | 1 | 0,6040 | Rejeição | `Funções públicas` não equivale lexicalmente a `cargos`; os termos compartilhados de turno favorecem outra pergunta. |
| Como faço meu cadastro eleitoral pela primeira vez? | 15 | 35 | 0,6448 | Rejeição | `Primeira vez` aproxima a consulta do primeiro acesso ao e-Título; TF-IDF não distingue bem alistamento e acesso ao aplicativo. |
| Uma certidão de nascimento serve para me identificar na eleição? | 31 | 33 | 0,4369 | Rejeição | `Certidão` aparece na resposta do ID 31, mas não na pergunta indexada. Indexar só perguntas limita a cobertura desse vocabulário. |
| Minha fotografia sumiu do e-titulo. O que faço? | 36 | 34 | 0,5598 | Rejeição | `Fotografia` e `sumiu` não coincidem com `foto` e `ausente`; sobra a sobreposição do termo do aplicativo. |

Um exemplo positivo é `Como faço pra pagra minha multa eleitoral?`, aceito
com ID 41 e escore 0,7620. Mesmo com erro de digitação, outros termos
compartilhados são suficientes; isso não significa que exista corretor
ortográfico. Já `Como pagar o boleto da mensalidade da faculdade?` alcança
0,6476 no ID 41, mas é corretamente rejeitada por 0,70. Ela ilustra o motivo
de testar rejeição também com palavras presentes no domínio.

As hipóteses explicam limitações da representação esparsa e do pipeline
herdado. Não há ainda experimento de ablação nem análise v1 × v2; esses
itens da comparação final ficam para a implementação da próxima versão.

## 9. Dependências, reprodutibilidade e testes

- Manter Python 3.10+ e execução em CPU. Usar NLTK, NumPy e scikit-learn,
  permitidos pelo enunciado; usar apenas biblioteca padrão para CSV, tipos,
  caminhos, argumentos e testes. Não adicionar pandas, pytest ou recursos
  de embeddings à instalação da v1.
- Corrigir `requirements.txt` de UTF-16 para UTF-8. Preservar NLTK 3.10.4
  da v0 e declarar NumPy `>=2.0,<3.0` e scikit-learn `>=1.6,<2.0`.
  As faixas permitem que o pip escolha versões compatíveis com o Python
  instalado, inclusive 3.10. Não fixar versões que exigem um Python mais novo.
- Ambiente efetivamente verificado: Python 3.14.3, NLTK 3.10.4, NumPy 2.5.4
  e scikit-learn 1.9.1. Outras combinações permitidas não foram executadas
  aqui; versões e plataforma podem causar pequenas diferenças numéricas.
- Documentar o download explícito de `punkt_tab`, `stopwords` e `rslp`
  para `.venv/nltk_data`. O NLTK encontra esse diretório no ambiente virtual.
  Não adicionar o leitor de PDF utilizado para consultar o enunciado às
  dependências do aplicativo.
- Ignorar ambiente virtual, recursos locais do NLTK e bytecode no Git.
  Versionar dados e resultados pequenos, para permitir inspeção e reprodução.
- O chatbot e a avaliação não usam aleatoriedade. Fixar a semente 42 no
  teste de vetores aleatórios; manter desempate por ID para reprodutibilidade.
- Usar `unittest` e testes de integração reais com a base. Verificar fórmula,
  contrato, ranking completo, empate, duplicatas, igualdade no limiar, vetor
  nulo, entradas especiais, texto no limite e acima dele, base inválida,
  vocabulário vazio, opções da normalização e imutabilidade do IDF em consultas.
- Conferir métricas com um exemplo pequeno cujo resultado é conhecido,
  a regra de desempate da calibração, IDs inválidos e a separação dos conjuntos.
  Recuperar as 50 perguntas originais é uma verificação funcional adicional,
  sem apresentá-la como avaliação em paráfrases.
- Executar 16 testes automatizados, a calibração, a avaliação e verificações
  da CLI. Os testes automatizados passaram com sucesso no ambiente acima.
