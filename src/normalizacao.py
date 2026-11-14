import unicodedata

from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

def _remover_acentos(texto):
    # Separa os acentos das letras antes de descartar as marcas combinantes.
    nfd = unicodedata.normalize("NFD", texto)
    return "".join(
        caractere
        for caractere in nfd
        if unicodedata.category(caractere) != "Mn"
    )

def normalizar(
    texto,
    tokenizar=True,
    minusculas=True,
    remover_pontuacao=True,
    remover_acentos=True,
    remover_stopwords=True,
    aplicar_stemming=True,
) -> list[str]:
    # Cada etapa continua independente para permitir experimentos de ablação.
    tokens = (
        word_tokenize(texto, language="portuguese")
        if tokenizar
        else texto.split()
    )

    if minusculas:
        tokens = [token.lower() for token in tokens]

    if remover_pontuacao:
        tokens = [token for token in tokens if any(c.isalnum() for c in token)]

    if remover_acentos:
        tokens = [_remover_acentos(token) for token in tokens]

    if remover_stopwords:
        stop = set(stopwords.words("portuguese"))
        # A lista precisa usar a mesma representação dos tokens sem acentos.
        if remover_acentos:
            stop = {_remover_acentos(palavra) for palavra in stop}
        tokens = [token for token in tokens if token not in stop]

    if aplicar_stemming:
        # O RSLP aproxima flexões em português; o resultado é um radical, não uma palavra.
        stemmer = RSLPStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    return tokens
