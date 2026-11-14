# Decisões de Projeto

## 1. Contexto e objetivos

O projeto implementa um assistente de perguntas frequentes sobre Eleições 2026. A base contém 50 pares de pergunta e resposta, com fontes registradas
no arquivo `data/base.csv`. O objetivo é comparar três etapas do trabalho:

- v0: pipeline configurável de normalização de texto em português;
- v1: recuperação com TF-IDF e similaridade de cossenos;
- v2: recuperação com sentence embeddings pré-treinados.

Nas versões v1 e v2, as respostas são recuperadas da base. O bônus RAG
implementado acrescenta uma reformulação por LLM à resposta recuperada pela
v2, preservando a resposta original para comparação. A implementação e as
evidências experimentais do bônus estão descritas na Seção 8.

O desenvolvimento ocorreu em etapas. Durante a implementação da v1, os
módulos estavam na raiz e a v2 ainda não existia. Depois da conclusão da v2,
os módulos de domínio foram reunidos no pacote `src`, sem alterar os
algoritmos. `chat.py` e `avaliacao.py` permanecem como pontos de entrada na
raiz. Não foram introduzidas camadas de serviços, bancos de dados, interfaces
web ou hierarquias de classes adicionais.

## 2. Base de conhecimento e conjuntos de dados

### 2.1. Base de conhecimento

A base original foi preservada. A leitura usa `csv.DictReader`, da biblioteca
padrão, e codificação `utf-8-sig`, que aceita o BOM do arquivo original e
também arquivos UTF-8 sem BOM. Cada registro é representado pela `dataclass`
imutável `ParFAQ`, com ID, pergunta, resposta e fonte.

O carregador exige as quatro colunas, campos preenchidos, IDs positivos e
únicos e pelo menos um registro. Ele não exige exatamente 50 pares, pois
bases menores são usadas nos testes. Linhas inválidas causam erro explícito;
não são ignoradas. O caminho padrão é resolvido a partir do pacote, de modo
que a API não dependa do diretório de trabalho. Um caminho passado por
`--base` continua relativo ao diretório de quem executa o comando.

Somente as perguntas são indexadas nas duas versões. Respostas e fontes não
entram na representação. Essa decisão segue a especificação do assistente e
evita que respostas longas dominem a comparação, mas impede que termos
presentes apenas nas respostas auxiliem a recuperação.

### 2.2. Validação e teste

`data/validacao.csv` possui 12 perguntas do escopo e 8 perguntas sem resposta
na base. Ele é usado para escolher limiares, modelo denso e configuração de
normalização. Perguntas sobre temas eleitorais sem resposta correspondente
também são consideradas fora do escopo.

`data/teste.csv` possui 30 paráfrases do escopo e 5 perguntas fora dele. Há
variações de registro, ordem, sinônimos e escrita informal, incluindo o erro
`pagra` e a grafia `e-titulo`. Perguntas de validação e teste não se repetem e
não são cópias literais das perguntas da base.

Os conjuntos foram preparados com acesso à base, sem autoria independente.
Isso pode introduzir viés. Há poucos exemplos de rejeição, e nem todos os IDs
da base aparecem no teste; os resultados são uma avaliação do conjunto
entregue, não uma estimativa geral de desempenho.

Essa limitação não é corrigida ao reescrever o teste depois de conhecer seus
resultados. Para obter evidência independente em uma próxima coleta, um
integrante que não escreveu a parte correspondente da base ou um colega de
outro grupo deve preparar novas perguntas sem consultar os resultados dos
assistentes. Devem ser registrados autoria, data e procedimento de coleta;
os IDs esperados devem ser conferidos antes de executar a avaliação. Modelo,
pipeline e limiares devem ser congelados previamente. O conjunto atual e seus
resultados ficam preservados como primeira avaliação, sem atribuir a eles uma
independência que não tiveram.

## 3. v0 - Pipeline de normalização

### 3.1. Etapas do pipeline

A função `normalizar` devolve uma lista de tokens e permite ativar ou desativar
separadamente:

- tokenização com NLTK em português;
- conversão para minúsculas;
- remoção de tokens de pontuação;
- remoção de acentos;
- remoção de stopwords do NLTK;
- stemming com `RSLPStemmer`.

Tokens inteiramente não alfanuméricos são descartados, sem reescrever
caracteres internos de tokens mistos. Números são preservados porque anos,
turnos e quantidades são relevantes no domínio.

### 3.2. Justificativas

A tokenização separa unidades textuais. Minúsculas e remoção de acentos
reduzem variações de escrita. A filtragem de pontuação retira elementos sem
conteúdo alfanumérico. A remoção de stopwords reduz palavras funcionais, e o
RSLP aproxima algumas flexões sem exigir um modelo externo de lematização.
Essas decisões reduzem o vocabulário da v1, mas não implicam que todas as
etapas sejam adequadas para embeddings densos.

### 3.3. Exemplos antes e depois

A tabela usa a configuração padrão completa.

|  ID | Pergunta                                                   | Resposta                                                                                                                                                                                        | Tokens normalizados                                         |
| --: | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
|   1 | Qual é a data do primeiro turno das Eleições 2026?         | O primeiro turno das Eleições 2026 será realizado em 4 de outubro de 2026.                                                                                                                      | `['dat', 'prim', 'turn', 'eleico', '2026']`                 |
|   2 | Qual é a data do segundo turno das Eleições 2026?          | O eventual segundo turno será realizado em 25 de outubro de 2026.                                                                                                                               | `['dat', 'segund', 'turn', 'eleico', '2026']`               |
|   3 | Qual é o horário da votação nas Eleições 2026?             | A votação ocorrerá das 8h às 17h, pelo horário de Brasília.                                                                                                                                     | `['horari', 'votaca', 'eleico', '2026']`                    |
|   4 | Quais cargos serão escolhidos no primeiro turno de 2026?   | No primeiro turno serão escolhidos presidente e vice-presidente da República, governador e vice-governador, dois senadores e seus suplentes, deputado federal e deputado estadual ou distrital. | `['qual', 'carg', 'escolh', 'prim', 'turn', '2026']`        |
|   5 | Quantas escolhas o eleitor fará na urna no primeiro turno? | O eleitor fará seis escolhas: deputado federal, deputado estadual ou distrital, primeiro senador, segundo senador, governador e presidente da República.                                        | `['quant', 'escolh', 'elei', 'far', 'urn', 'prim', 'turn']` |

### 3.4. Limitações e perda de informação

A lista padrão de stopwords é mantida. Quando essa etapa está ativa, termos
como `não`, `nem` e `sem` são removidos. Isso pode inverter ou apagar parte do
sentido de perguntas negativas, e o limiar de rejeição não corrige essa perda.
A base foi escrita majoritariamente de forma afirmativa, mas a limitação deve
ser revista se perguntas negativas forem incorporadas.

Não há exceções para siglas ou termos como `e-Título`. Caixa e acentos deixam
de distingui-los, e o RSLP pode fundir palavras de sentidos diferentes ou não
aproximar flexões relacionadas. A v1 aceita opções alternativas pelo parâmetro
`opcoes_normalizacao`, mas o limiar registrado corresponde à configuração
padrão completa.

## 4. v1 - Recuperação com TF-IDF

### 4.1. Representação dos documentos

A v1 usa `TfidfVectorizer` ajustado somente sobre as perguntas da base. A
função `normalizar` é fornecida como analisador por `functools.partial`, sem
uma segunda tokenização ou conversão de caixa pelo scikit-learn.

São usados unigramas, frequência bruta, IDF suavizado e normalização L2,
configurações padrão do vetorizador. Para N documentos, o IDF segue a variante
`log((1 + N) / (1 + df(t))) + 1`. Não são usados TF sublinear, bigramas,
dicionários de sinônimos, correção ortográfica ou expansão de consultas.

O vocabulário e o IDF são ajustados uma vez por instância. Consultas usam
apenas `transform` e não alteram esses valores. A matriz da base é convertida
para um array NumPy denso; com 50 perguntas, isso simplifica o cálculo. A v1
não mantém cache em disco. Uma base que produza vocabulário vazio gera erro
controlado.

### 4.2. Similaridade, ranking e contrato

A similaridade de cossenos foi implementada com NumPy: produto matricial,
normas L2 e divisão pelo produto das normas. O código de produção não chama o
cosseno do scikit-learn. Vetores nulos recebem escore zero, sem divisão por
zero ou `NaN`, e o resultado é limitado ao intervalo [-1, 1] para compensar
pequenos erros numéricos.

Os testes comparam a implementação com
`sklearn.metrics.pairwise.cosine_similarity`, inclusive para vetores nulos e
aleatórios, com tolerâncias absoluta e relativa de `1e-12`.

As versões retornam `Resultado(resposta, id, escore, ranking)`. O ranking
contém todos os pares, em escore decrescente. Empates exatos são resolvidos
por ID crescente, sem arredondamento ou tolerância. Pares com respostas iguais
permanecem distintos: a avaliação usa o ID esperado e não concede crédito a
outro ID apenas por compartilhar a resposta.

Quando o melhor escore não supera o limiar, o ID retornado é -1 e a mensagem é
`Desculpe, não entendi. Pode reformular?`. O melhor escore e o ranking completo
são mantidos mesmo na rejeição para permitir MRR e análise do limiar.

### 4.3. Limiar de rejeição

A regra é estrita: a resposta só é aceita quando `escore > limiar`; igualdade
é rejeição. Valores não finitos ou fora de [0, 1] são recusados. Com limiar
zero, uma consulta com vetor nulo continua rejeitada.

Foram avaliados 21 limiares, de 0,00 a 1,00 em passos de 0,05, somente na
validação. O objetivo é a média entre acertos de ranking aceitos no escopo e
rejeições corretas fora do escopo. Empates favorecem o menor limiar para
reduzir falsas rejeições.

| Limiar | Acertos aceitos no escopo | Rejeição correta | Falsa rejeição | Objetivo |
| -----: | ------------------------: | ---------------: | -------------: | -------: |
|   0,50 |                      2/12 |              2/8 |           3/12 |   0,2083 |
|   0,65 |                      1/12 |              6/8 |          10/12 |   0,4167 |
|   0,70 |                      1/12 |              8/8 |          10/12 |   0,5417 |
|   0,75 |                      1/12 |              8/8 |          10/12 |   0,5417 |
|   1,00 |                      0/12 |              8/8 |          12/12 |   0,5000 |

O limiar padrão escolhido foi 0,70. A Acurácia@1 da validação foi 2/12 e o
MRR, 0,4421. Apenas duas consultas ficaram acima de 0,70: uma estava correta
e a outra recuperou o ID 45 no lugar do 46. O limiar é conservador, não
corrige erros semânticos e não garante que toda resposta aceita esteja
correta. Uma validação maior exigiria nova calibração, sem usar o teste atual.

### 4.4. Entradas e limitações

A interface aceita strings. Entradas vazias, espaços, pontuação, emojis e
vocabulário desconhecido produzem resposta controlada e ranking completo.
Textos em outros idiomas podem ser rejeitados ou encontrar termos comuns; não
há detector de idioma. Consultas acima de 10.000 caracteres são rejeitadas
antes da tokenização, sem truncamento.

O terminal exibe ID, escore com três casas e resposta, e encerra com `sair`,
Ctrl+C ou fim de arquivo. O arredondamento é apenas visual; cálculos e CSVs
mantêm a precisão disponível.

## 5. v2 - Recuperação com embeddings densos

### 5.1. Modelo escolhido

A v2 usa `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
revisão `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, com execução explícita em
CPU. Nos experimentos registrados, a versão de `sentence-transformers` foi
5.7.0. O modelo foi sugerido no enunciado, suporta português e é menor que a
alternativa avaliada.

A alternativa foi `sentence-transformers/distiluse-base-multilingual-cased-v2`,
revisão `bfe45d0732ca50787611c0fe107ba278c7f3f889`. A comparação foi limitada a
esses dois modelos compatíveis com português e CPU.

### 5.2. Cache dos embeddings

Os embeddings das 50 perguntas são calculados uma vez e salvos em arquivos
NumPy compactados sob `.cache/embeddings/`. Consultas calculam apenas o próprio
embedding. A chave SHA-256 inclui todos os campos da base, os textos
efetivamente indexados, identificador e revisão do modelo, versão da
biblioteca e opções de normalização.

Mudanças nesses elementos geram outro artefato. O cache é criado
automaticamente e não é versionado. Na leitura, metadados e dimensões são
verificados; um arquivo incompatível ou corrompido causa erro explícito. A
gravação usa arquivo temporário e substituição atômica.

### 5.3. Normalização e comparação entre modelos

Os quatro experimentos abaixo usaram apenas a validação. Cada configuração
recebeu seu próprio limiar, escolhido pelo mesmo objetivo da v1.

| Modelo    | Texto       | Limiar | Acurácia@1 |    MRR | Rejeição correta | Falsa rejeição | Objetivo |
| --------- | ----------- | -----: | ---------: | -----: | ---------------: | -------------: | -------: |
| MiniLM    | original    |   0,80 |     0,8333 | 0,9028 |           1,0000 |         0,3333 |   0,8333 |
| MiniLM    | normalizado |   0,90 |     0,3333 | 0,5087 |           1,0000 |         0,6667 |   0,6250 |
| DistilUSE | original    |   0,65 |     0,8333 | 0,8799 |           0,7500 |         0,0000 |   0,7917 |
| DistilUSE | normalizado |   0,75 |     0,3333 | 0,4640 |           0,8750 |         0,5000 |   0,5625 |

Foi escolhido o MiniLM com texto original por apresentar o maior objetivo e
MRR, além de rejeitar as 8 perguntas fora do escopo. O DistilUSE original
aceitou mais consultas válidas, mas rejeitou somente 6/8 casos fora do escopo.
A validação contém apenas 20 itens; a escolha é uma evidência local, não uma
conclusão geral sobre as arquiteturas.

Aplicar o pipeline completo da v0 reduziu a Acurácia@1 de ambos os modelos de
0,8333 para 0,3333. No MiniLM, o MRR caiu de 0,9028 para 0,5087. A configuração
final mantém o texto natural nos dois lados da recuperação. A hipótese é que
Transformers treinados com frases naturais utilizam palavras funcionais,
flexões e contexto, enquanto o RSLP e a remoção de stopwords eliminam essas
pistas. Esse contraste mede a normalização completa. A ablação de uma única
etapa, para isolar seu efeito, está na Seção 6.6.

### 5.4. Limiar de rejeição

Na validação, o limiar 0,80 maximizou o objetivo em 0,8333. Ele aceitou 8/12
acertos de ranking, rejeitou 8/8 casos fora do escopo e rejeitou falsamente
4/12 perguntas válidas. Em 0,65, a falsa rejeição era zero, mas apenas 6/8
casos fora do escopo eram rejeitados. A decisão favorece não responder fora
da base e pode exigir nova calibração se os dados mudarem.

## 6. Avaliação experimental

### 6.1. Metodologia e métricas

Modelo, normalização e limiares foram definidos na validação antes da avaliação
final. O teste não é usado para ajustar parâmetros. O script valida as colunas,
perguntas preenchidas e IDs existentes e exige casos dentro e fora do escopo.

As métricas são:

- Acurácia@1: proporção das 30 perguntas do escopo cujo primeiro ID do ranking
  é o esperado, independentemente da rejeição;
- MRR: média do recíproco da posição do ID esperado no ranking completo;
- rejeição correta: proporção das 5 perguntas fora do escopo rejeitadas;
- falsa rejeição: proporção das 30 perguntas válidas rejeitadas pelo limiar.

### 6.2. Comparação entre v1 e v2

| Versão | Limiar |     Acurácia@1 |    MRR | Rejeição correta | Falsa rejeição |
| ------ | -----: | -------------: | -----: | ---------------: | -------------: |
| v1     |   0,70 | 0,6667 (20/30) | 0,7414 |     1,0000 (5/5) | 0,6667 (20/30) |
| v2     |   0,80 | 0,8000 (24/30) | 0,8628 |     1,0000 (5/5) | 0,6000 (18/30) |

Na v1, somente 10 das 30 consultas válidas receberam resposta, embora o
ranking acertasse 20. Todas as 10 respostas aceitas estavam corretas neste
teste. Isso não equivale a 66,67% de atendimento: esse valor descreve o
ranking antes da rejeição.

A v2 melhorou Acurácia@1 e MRR e reduziu a falsa rejeição, mas ainda recusou
18 das 30 perguntas válidas. O resultado evidencia a relação entre cobertura
e rejeição: uma representação semântica melhor não elimina a necessidade de
calibrar o limiar.

### 6.3. Erros e acertos observados na v1

| Pergunta                                                         | Esperado | Primeiro ID v1 | Escore | Saída    | Hipótese                                                                                             |
| ---------------------------------------------------------------- | -------: | -------------: | -----: | -------- | ---------------------------------------------------------------------------------------------------- |
| Quando será a segunda rodada da eleição deste ano de 2026?       |        2 |              2 | 0,4051 | Rejeição | A paráfrase perde `data`, `segundo` e `turno`; o ranking acerta, mas o escore fica abaixo do limiar. |
| Que funções públicas serão eleitas no primeiro turno?            |        4 |              1 | 0,6040 | Rejeição | `Funções públicas` não coincide com `cargos`, e os termos de turno favorecem outra pergunta.         |
| Como faço meu cadastro eleitoral pela primeira vez?              |       15 |             35 | 0,6448 | Rejeição | `Primeira vez` aproxima lexicalmente a consulta do primeiro acesso ao e-Título.                      |
| Uma certidão de nascimento serve para me identificar na eleição? |       31 |             33 | 0,4369 | Rejeição | `Certidão` aparece na resposta do ID 31, mas não na pergunta indexada.                               |
| Minha fotografia sumiu do e-titulo. O que faço?                  |       36 |             34 | 0,5598 | Rejeição | `Fotografia` e `sumiu` não coincidem com `foto` e `ausente`.                                         |

A consulta `Como faço pra pagra minha multa eleitoral?` foi aceita com ID 41
e escore 0,7620. O erro de digitação não foi corrigido; os demais termos foram
suficientes. Já `Como pagar o boleto da mensalidade da faculdade?` obteve
0,6476 no ID 41 e foi rejeitada pelo limiar 0,70, apesar da sobreposição de
vocabulário.

### 6.4. Casos em que a v2 acerta e a v1 erra

Foram observados 8 casos nesse sentido. Três exemplos são:

| Pergunta                                                         | Esperado |  v1 |  v2 | Hipótese                                                                                                      |
| ---------------------------------------------------------------- | -------: | --: | --: | ------------------------------------------------------------------------------------------------------------- |
| Que funções públicas serão eleitas no primeiro turno?            |        4 |   1 |   4 | O embedding aproxima `funções públicas` de `cargos`; o TF-IDF se apoia em `primeiro turno` e favorece a data. |
| Como faço meu cadastro eleitoral pela primeira vez?              |       15 |  35 |  15 | A v2 relaciona cadastro ao primeiro título; a v1 associa `primeira vez` ao primeiro acesso ao aplicativo.     |
| Uma certidão de nascimento serve para me identificar na eleição? |       31 |  33 |  31 | A v2 representa a intenção de documento inválido; a v1 encontra vocabulário eleitoral genérico.               |

### 6.5. Casos em que a v1 acerta e a v2 erra

Foram observados 4 casos nesse sentido. Três exemplos são:

| Pergunta                                                  | Esperado |  v1 |  v2 | Hipótese                                                                                                                             |
| --------------------------------------------------------- | -------: | --: | --: | ------------------------------------------------------------------------------------------------------------------------------------ |
| Quais requisitos preciso cumprir pra transferir o título? |       19 |  19 |  18 | Os IDs 18 e 19 são semanticamente próximos; a v1 preserva `requisitos/condições`, enquanto a v2 favorece a ação geral de transferir. |
| Quais documentos com foto eu posso levar no dia de votar? |       28 |  28 |  30 | `Foto` aproxima a v2 da pergunta específica sobre documento vencido; o TF-IDF mantém o foco na lista de documentos.                  |
| Minha identidade está vencida. Ela ainda vale para votar? |       30 |  30 |  12 | A v1 casa `vencida` e `votar`; o embedding perde a condição documental e aproxima a pergunta de elegibilidade.                       |

Os resultados mostram que embeddings lidam melhor com sinonímia e
reformulações, enquanto TF-IDF pode ser superior quando uma pista lexical rara
ou uma distinção fina entre perguntas próximas é decisiva. Os IDs 18 e 19
permanecem pares distintos; o MRR usa o ID esperado, sem fundi-los.

### 6.6. Ablação isolada do stemming na v1

O comando `python avaliacao.py --ablacao-v1` compara a configuração completa
da v0 com outra em que somente `aplicar_stemming=False`. A base, as demais
etapas, o vetorizador, o cosseno e o limiar 0,70 são mantidos. O TF-IDF é
ajustado novamente às perguntas da base em cada condição, pois os tokens
mudam. O experimento lê exclusivamente `data/validacao.csv` (12 perguntas do
escopo e 8 fora dele) e grava `resultados/ablacao_v1.csv`.

| Stemming RSLP | Limiar | Acurácia@1 | MRR | Rejeição correta | Falsa rejeição |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ligado | 0,70 | 0,1667 (2/12) | 0,4421 | 1,0000 (8/8) | 0,8333 (10/12) |
| Desligado | 0,70 | 0,3333 (4/12) | 0,4594 | 1,0000 (8/8) | 0,8333 (10/12) |

Desligar o stemming melhorou o ranking em duas consultas: `Qual a sequência
dos candidatos na urna?` passou do ID 49 para o esperado 6; `Posso votar
levando só o título em papel?` passou do ID 42 para o esperado 29. Uma hipótese
é que preservar as formas das palavras mantém distinções lexicais que o RSLP
reduz. Isso não demonstra que o stemming seja sempre prejudicial: a validação
é pequena e o ganho de MRR foi de apenas 0,0173.

As duas correções continuaram abaixo do limiar (escores 0,4545 e 0,4721).
Assim, melhorar o primeiro item do ranking não alterou as taxas de rejeição.
O limiar foi mantido fixo para que a comparação medisse apenas a mudança na
representação; uma recalibração simultânea confundiria os dois efeitos.

Esta ablação complementar foi executada após a avaliação original. Ela não
foi usada para alterar o pipeline padrão ou selecionar novamente parâmetros:
as métricas de teste da Seção 6.2 continuam correspondendo à configuração
original com stemming. Uma futura mudança de configuração exige nova
calibração em validação e uma avaliação independente, conforme a Seção 2.2.

## 7. Reprodutibilidade, testes e limitações

O projeto requer Python 3.10 ou superior e execução em CPU. As dependências de
produção são NLTK, NumPy, scikit-learn e sentence-transformers; PyTorch é
instalado como dependência desta última. O bônus usa o SDK `openai` como
cliente da API compatível do OpenRouter e `python-dotenv` para ler a
configuração local do `.env`. Recursos `punkt_tab`, `stopwords` e
`rslp` são baixados explicitamente. O leitor de PDF usado durante a consulta
ao enunciado não integra as dependências do aplicativo.

Na etapa da v1, o `requirements.txt` foi convertido de UTF-16 para UTF-8. Ele
mantém NLTK 3.10.4 e faixas compatíveis de NumPy e scikit-learn; a v2 acrescenta
sentence-transformers. Não foram adicionados pandas, pytest ou dependências
sem uso no aplicativo. As tabelas em `resultados/` são geradas pelos scripts
de calibração e avaliação, e não calculadas manualmente.

O ambiente da avaliação final usou Python 3.13.9, NLTK 3.10.4, NumPy 2.5.4,
scikit-learn 1.9.1, sentence-transformers 5.7.0 e PyTorch 2.15.0+cpu. Outras
combinações aceitas pelo `requirements.txt` podem apresentar pequenas
diferenças numéricas.

O chatbot e a avaliação não usam aleatoriedade. Testes com vetores aleatórios
fixam a semente 42, e o desempate por ID torna o ranking determinístico. Na
etapa da v1, havia 16 testes automatizados. Após a v2, a suíte passou a ter 22
testes. Com o bônus e a verificação da ablação, a suíte tem 29 testes,
incluindo interface, ranking, limiar, persistência e invalidação de cache,
entradas especiais, configuração de CPU e tratamento de falhas da API.
Os testes da v2 usam embeddings simulados, e os do RAG usam um cliente
simulado; eles não substituem a execução com o modelo real e a API.

Os resultados pequenos em `resultados/` são versionados para inspeção e
reprodução. Ambientes virtuais, bytecode, recursos locais do NLTK, caches de
embeddings e arquivos temporários não são versionados.

As principais limitações são o tamanho dos conjuntos de validação e teste, o
viés possível na elaboração das perguntas, a remoção de negação na
normalização padrão, a indexação exclusiva das perguntas e a dificuldade de
distinguir pares semanticamente muito próximos. Os resultados devem ser
interpretados dentro desse conjunto experimental.

## 8. Avaliação experimental do bônus RAG

### 8.1. Procedimento

O bônus encadeia a recuperação densa da v2 com uma chamada ao OpenRouter. A
pergunta e a resposta recuperada são enviadas ao modelo
`nvidia/nemotron-3.5-lightning:free`, instruído a reformular o conteúdo sem
acrescentar fatos. Se a v2 rejeita a pergunta, o LLM não é acionado.

A coleta de 14 de novembro de 2026 preservou base, embeddings, limiar 0,80 e
prompt. Foram feitas 11 execuções no CLI: uma foi rejeitada pela v2 antes do
LLM e dez produziram chamadas reais ao OpenRouter. Todos os casos aceitos
recuperaram a pergunta idêntica da base, com escore 1,000, para separar a
análise generativa de possíveis erros de recuperação. O registro integral está
em `resultados/avaliacao_rag.md`.

### 8.2. Três melhorias selecionadas

A seleção considera organização, fluidez e preservação das informações da
base. O ganho de apresentação é avaliado separadamente da correção factual:
uma resposta mais legível que altera um procedimento é classificada como
distorção. As três melhorias têm intensidades distintas:

| Caso | Ganho observável | Informações preservadas | Intensidade |
| --- | --- | --- | --- |
| ID 16 | Lista em lugar de um período longo, facilitando conferir os documentos | Documentos, condições e selfie no atendimento virtual | Clara |
| ID 30 | Referência explícita à votação e redação mais fluida sobre a validade | Permissão restrita aos documentos oficiais com foto aceitos pela Justiça Eleitoral | Modesta |
| ID 35 | Sequência dividida em duas frases, facilitando acompanhar as ações | Baixar, informar dados exatos, responder perguntas e criar senha | Modesta |

Os textos originais e as saídas integrais constam em
`resultados/avaliacao_rag.md`. A classificação dos casos 30 e 35 é qualitativa;
não foi demonstrado ganho de compreensão por usuários ou avaliadores
independentes.

**Documentos para o primeiro título (ID 16).** A resposta original reúne os
documentos em um período longo. O LLM os apresentou como lista e manteve o
documento de identificação, o vínculo com o município, o débito quando houver,
a quitação militar para conscritos e a selfie no atendimento virtual. A
melhoria é de organização e localização visual, sem mudança de conteúdo.

**Documento vencido (ID 30).** A resposta manteve a confirmação de que os
documentos oficiais com foto aceitos pela Justiça Eleitoral podem ser usados
mesmo vencidos. A reformulação explicitou "para votação" e substituiu "data de
validade vencida" por "data de validade tenha expirado". Trata-se de melhoria
modesta de adequação e fluidez, sem novo fato ou condição.

**Primeiro acesso ao e-Título (ID 35).** A sequência original aparece em uma
única frase. O LLM contextualizou a abertura e separou as etapas em duas
frases: baixar o aplicativo e informar os dados, depois responder às perguntas
de segurança e criar uma senha. Todas as etapas foram preservadas, com leitura
mais direta.

### 8.3. Distorção selecionada

No caso da foto ausente no e-Título (ID 36), a v2 recuperou corretamente a
resposta da base. O original orienta a pessoa a "refazer a conferência
biométrica"; o LLM escreveu "refazer o cadastro biométrico" e condicionou essa
ação a "Caso não esteja" cadastrada e validada. Cadastro e conferência não são
o mesmo procedimento, e a condição foi introduzida pelo modelo. Portanto, a
distorção é generativa, não um erro de recuperação.

Também foi observado um caso adicional no ID 19: a retirada da expressão "Em
regra" transformou condições qualificadas para transferência em uma afirmação
absoluta. Esse caso foi mantido no registro completo, mas o ID 36 foi escolhido
como evidência principal por conter alteração procedimental mais explícita.

### 8.4. Limitações

A avaliação é qualitativa e pequena, com uma geração por pergunta. O endpoint
gratuito tem disponibilidade variável, e novas execuções podem produzir textos
diferentes. Nem toda reformulação foi melhor: duas foram quase idênticas ao
original, uma ficou apenas mais longa, uma trocou "pedido" por "autorização",
outra alterou "solicitada" para "retirada" e uma incluiu comentário
metalinguístico. A observação de melhorias não elimina o risco de distorção,
mesmo com instrução explícita para usar exclusivamente a resposta recuperada.
