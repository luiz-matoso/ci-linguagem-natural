"""v1: recuperação de perguntas com TF-IDF e cosseno calculado em NumPy."""

from dataclasses import dataclass
from functools import partial

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from base import BASE_PADRAO, carregar_base
from normalizacao import normalizar


LIMIAR_PADRAO = 0.70  # Escolhido em data/validacao.csv, sem consultar o teste.
MAX_CARACTERES = 10_000
MENSAGEM_REJEICAO = "Desculpe, não entendi. Pode reformular?"


@dataclass
class Resultado:
    resposta: str
    id: int
    escore: float
    ranking: list[tuple[int, float]]


def similaridade_cossenos(consulta, documentos) -> np.ndarray:
    """Calcula q·d / (||q|| ||d||) para cada linha de documentos."""
    consulta = np.asarray(consulta, dtype=float)
    documentos = np.asarray(documentos, dtype=float)
    if (consulta.ndim != 1 or documentos.ndim != 2
            or documentos.shape[1] != consulta.shape[0]):
        raise ValueError("Esperados um vetor e uma matriz com o mesmo vocabulário.")

    produtos = documentos @ consulta
    denominadores = np.linalg.norm(documentos, axis=1) * np.linalg.norm(consulta)
    # Perguntas sem termos conhecidos viram vetores nulos e recebem escore zero.
    escores = np.divide(
        produtos, denominadores, out=np.zeros_like(produtos),
        where=denominadores != 0,
    )
    return np.clip(escores, -1.0, 1.0)


class ChatbotTFIDF:
    def __init__(self, caminho_base=BASE_PADRAO, limiar=LIMIAR_PADRAO,
                 opcoes_normalizacao=None):
        if not 0 <= limiar <= 1:
            raise ValueError("O limiar deve estar entre 0 e 1.")
        self.limiar = float(limiar)
        self.base = carregar_base(caminho_base)
        self.pares_por_id = {par.id: par for par in self.base}

        # O mesmo pipeline da v0 processa a base e todas as consultas.
        # O vocabulário e o IDF são ajustados somente nas perguntas da base.
        self.vetorizador = TfidfVectorizer(
            analyzer=partial(normalizar, **(opcoes_normalizacao or {})),
            lowercase=False, token_pattern=None,
        )
        try:
            self.vetores_base = self.vetorizador.fit_transform(
                [par.pergunta for par in self.base]
            ).toarray()
        except ValueError as erro:
            raise ValueError("A base não produziu um vocabulário TF-IDF válido.") from erro

    def responder(self, texto: str) -> Resultado:
        # Recusar entradas muito longas antes da tokenização limita o trabalho.
        if len(texto) > MAX_CARACTERES or not texto.strip():
            escores = np.zeros(len(self.base))
        else:
            consulta = self.vetorizador.transform([texto]).toarray()[0]
            escores = similaridade_cossenos(consulta, self.vetores_base)

        # O ID crescente desempata sem depender da ordem das linhas do CSV.
        ranking = sorted(
            [(par.id, float(escore)) for par, escore in zip(self.base, escores)],
            key=lambda item: (-item[1], item[0]),
        )
        identificador, escore = ranking[0]
        # A rejeição mantém o ranking e o melhor escore para avaliação e MRR.
        if escore <= self.limiar:
            return Resultado(MENSAGEM_REJEICAO, -1, escore, ranking)
        return Resultado(
            self.pares_por_id[identificador].resposta, identificador, escore, ranking,
        )
