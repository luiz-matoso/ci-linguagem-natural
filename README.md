# Trabalho Prático - Linguagem Natural

Assistente de FAQ sobre **Eleições 2026**, com 50 pares de pergunta e resposta.
A v0 normaliza textos em português. A v1 recupera respostas com TF-IDF,
similaridade de cossenos implementada em NumPy e limiar de rejeição.

## Instalação

Use Python **3.10 ou superior**, em CPU. Na raiz do projeto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader -d .venv/nltk_data punkt_tab stopwords rslp
```

Os recursos do NLTK ficam dentro do ambiente virtual. A instalação precisa
de internet; o chatbot e a avaliação executam localmente depois disso.

## Conversar com a v1

```bash
python chat.py --versao v1
```

Digite uma pergunta e pressione Enter. Use `sair`, Ctrl+C ou fim de arquivo
para encerrar. Exemplo:

```text
> Qual é a data do primeiro turno das Eleições 2026?
[v1] (id=1, escore=1.000) O primeiro turno das Eleições 2026 será realizado em 4 de outubro de 2026.
> Qual é a senha do wifi da biblioteca?
[v1] (id=-1, escore=0.000) Desculpe, não entendi. Pode reformular?
```

O limiar padrão é `0.70`, escolhido em validação. Para experimentar outro
limiar ou arquivo de base:

```bash
python chat.py --versao v1 --limiar 0.50 --base data/base.csv
```

Só há resposta quando o escore é **maior** que o limiar. Entradas acima de
10.000 caracteres são rejeitadas com uma orientação para encurtar a pergunta.

## Usar pelo código

```python
from chatbot_tfidf import ChatbotTFIDF

assistente = ChatbotTFIDF()
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
python avaliacao.py --calibrar
python avaliacao.py
```

A calibração lê somente `data/validacao.csv` (12 perguntas do escopo e 8 de
rejeição), testa limiares de 0 a 1 com passo 0,05 e grava
`resultados/calibracao.csv`. Ela informa o limiar escolhido sem editar o código.

A avaliação usa `data/teste.csv` (30 paráfrases e 5 perguntas fora do escopo),
imprime as quatro métricas e grava `resultados/comparacao.csv`, atualmente com
uma linha para a v1. `python avaliacao.py --limiar 0.50` permite avaliar outro
limiar, mas o teste não deve ser usado para escolhê-lo.

Resultado com o limiar padrão: Acurácia@1 **66,67%**, MRR **0,7414**,
rejeição correta **100%** e falsa rejeição **66,67%**. Acurácia@1 e MRR medem
o ranking antes da rejeição. As limitações e os exemplos de erro estão em
[dedisoes.md](dedisoes.md).

## Organização

- `normalizacao.py`: pipeline configurável da v0.
- `base.py`: leitura e validação do CSV de conhecimento.
- `chatbot_tfidf.py`: v1, cosseno e contrato de resposta.
- `chat.py`: interface de terminal.
- `avaliacao.py`: calibração e métricas.
- `data/`: base, validação e teste separados.
- `tests/test_v1.py`: testes automatizados com `unittest`.
- `resultados/`: resultados gerados pela avaliação.
- `docs/decisoes.md`: decisões e exemplos originais da v0.
- `dedisoes.md`: decisões detalhadas desta implementação da v1.

A v2 com embeddings e os bônus ficam para a próxima etapa do trabalho.

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
