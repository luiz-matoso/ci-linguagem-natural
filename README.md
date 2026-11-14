# Trabalho Prático - Linguagem Natural

Assistente de FAQ sobre **Eleições 2026**, com 50 pares de pergunta e resposta.
A v0 normaliza textos em português. A v1 recupera respostas com TF-IDF e a
v2 usa embeddings densos pré-treinados. Ambas compartilham a similaridade de
cossenos implementada em NumPy, o contrato de resposta e o ranking completo.

## Instalação

Use Python **3.10 ou superior**, em CPU. Na raiz do projeto:

```bash
python3 -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m nltk.downloader -d .venv/nltk_data punkt_tab stopwords rslp
```

Os recursos do NLTK ficam dentro do ambiente virtual. A instalação precisa
de internet; o chatbot e a avaliação executam localmente depois disso.

Na primeira execução da v2, a `sentence-transformers` baixa o modelo fixado no
código. Os pesos ficam no cache local do Hugging Face; os embeddings da base
ficam em `.cache/embeddings/` e são regenerados se base, modelo, revisão,
versão da biblioteca ou pré-processamento mudar.

## Conversar com a v1 ou v2

```bash
python chat.py --versao v1
python chat.py --versao v2
```

Digite uma pergunta e pressione Enter. Use `sair`, Ctrl+C ou fim de arquivo
para encerrar. Exemplo:

```text
> Qual é a data do primeiro turno das Eleições 2026?
[v1] (id=1, escore=1.000) O primeiro turno das Eleições 2026 será realizado em 4 de outubro de 2026.
> Qual é a senha do wifi da biblioteca?
[v1] (id=-1, escore=0.000) Desculpe, não entendi. Pode reformular?
```

Os limiares padrão, escolhidos separadamente na validação, são `0.70` para v1
e `0.80` para v2. Para experimentar outro limiar ou arquivo de base:

```bash
python chat.py --versao v1 --limiar 0.50 --base data/base.csv
python chat.py --versao v2 --limiar 0.75 --base data/base.csv
```

Só há resposta quando o escore é **maior** que o limiar. Entradas acima de
10.000 caracteres são rejeitadas com uma orientação para encurtar a pergunta.

## Usar pelo código

```python
from src.chatbot_tfidf import ChatbotTFIDF
from src.chatbot_denso import ChatbotDenso

assistente = ChatbotTFIDF()
# ou: assistente = ChatbotDenso()
resultado = assistente.responder("Quantos dígitos tem o número para senador?")
print(resultado.resposta)  # Resposta da base ou mensagem de rejeição.
print(resultado.id)        # ID do par; -1 se rejeitado.
print(resultado.escore)    # Melhor similaridade encontrada.
print(resultado.ranking)  # Todos os pares (id, escore), em ordem decrescente.
```

As etapas da v0 continuam configuráveis, por exemplo:
`ChatbotTFIDF(opcoes_normalizacao={"aplicar_stemming": False})`.
Essa mudança altera a representação e exige uma nova calibração para uso real.

## Validar e avaliar

```bash
python -m unittest discover -s tests -v
python avaliacao.py --experimentos-v2
python avaliacao.py --calibrar --versao v2
python avaliacao.py
```

A comparação de configurações e a calibração leem somente
`data/validacao.csv` (12 perguntas do escopo e 8 de rejeição). O primeiro
comando compara dois modelos com e sem a normalização da v0 e grava
`resultados/experimentos_v2.csv`. A calibração testa limiares de 0 a 1 com
passo 0,05 e grava `resultados/calibracao_v2.csv`.

A avaliação usa `data/teste.csv` (30 paráfrases e 5 perguntas fora do escopo),
imprime as quatro métricas e grava automaticamente as linhas v1 e v2 em
`resultados/comparacao.csv`. O teste não deve ser usado para escolher modelo,
normalização ou limiar.

Resultados com os limiares fixados: v1 obteve Acurácia@1 **66,67%**, MRR
**0,7414**, rejeição correta **100%** e falsa rejeição **66,67%**; v2 obteve
**80,00%**, **0,8628**, **100%** e **60,00%**, respectivamente. Acurácia@1 e
MRR medem o ranking antes da rejeição. As decisões e a análise comparativa
estão em [docs/decisoes.md](docs/decisoes.md).

## Organização

- `src/normalizacao.py`: pipeline configurável da v0.
- `src/base.py`: leitura e validação do CSV de conhecimento.
- `src/chatbot_tfidf.py`: v1, cosseno e contrato de resposta.
- `src/chatbot_denso.py`: v2, embeddings densos e cache invalidável.
- `src/__init__.py`: definição do pacote Python.
- `chat.py`: interface de terminal.
- `avaliacao.py`: calibração e métricas.
- `data/`: base, validação e teste separados.
- `tests/`: testes automatizados da v1 e da v2 com `unittest`.
- `resultados/`: resultados gerados pela avaliação.
- `docs/decisoes.md`: decisões consolidadas da v0, v1 e v2, experimentos e análise de erros.

O bônus opcional permanece fora do escopo desta etapa.

## Fontes da base

Os endereços abaixo já constam na coluna `fonte` de `data/base.csv`, herdada
da v0. As respostas são recuperadas desse arquivo, sem atualização pela internet.

- [Calendário das Eleições 2026](https://www.tse.jus.br/eleicoes/cde-2026).
- [Ordem de votação na urna eletrônica](https://www.tse.jus.br/comunicacao/noticias/2026/Setembro/faltam-25-dias-saiba-qual-e-a-ordem-de-votacao-na-urna-eletronica).
- [Carta de serviços ao eleitor](https://www.tse.jus.br/servicos-eleitorais/servicos/carta-de-servicos-ao-eleitor).
- [Perguntas frequentes sobre título eleitoral](https://www.tse.jus.br/servicos-eleitorais/titulo-eleitoral/titulo-eleitoral-faq).
- [Documentos para votar](https://www.tse.jus.br/comunicacao/noticias/2026/Abril/eleitor-em-dia-saiba-quais-documentos-sao-validos-para-votar-nas-eleicoes-2026).
- [Perguntas frequentes sobre e-Título](https://www.tse.jus.br/servicos-eleitorais/e-titulo-perguntas-frequentes).
- [Prazos para justificar ausência](https://www.tse.jus.br/comunicacao/noticias/2026/Outubro/manual-do-eleitor-veja-os-prazos-para-justificar-ausencia-nas-eleicoes-2026).
- [Voto em trânsito](https://www.tse.jus.br/eleicoes/eleicoes-2026-content/voto-em-transito-perguntas-e-respostas).
- [Acessibilidade na votação](https://www.tse.jus.br/comunicacao/noticias/2026/Agosto/manual-do-eleitor-acessibilidade-amplia-autonomia-de-pessoas-com-deficiencia-nas-eleicoes-2026).
- [Divulgação de candidaturas e contas eleitorais](https://www.tse.jus.br/administracao/painel/divulgacao-de-candidaturas-e-contas-eleitorais).
- [Regras na cabine de votação](https://www.tse.jus.br/comunicacao/noticias/2026/Outubro/falta-1-dia-veja-o-que-pode-e-o-que-nao-pode-no-1o-turno-das-eleicoes-2026).
