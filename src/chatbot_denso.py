"""v2: recuperação de perguntas com sentence embeddings e cosseno em NumPy."""

import hashlib
import json
import math
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .base import BASE_PADRAO, carregar_base
from .chatbot_tfidf import (
    MAX_CARACTERES, MENSAGEM_REJEICAO, Resultado, similaridade_cossenos,
)
from .normalizacao import normalizar


MODELO_PADRAO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_ALTERNATIVO = "sentence-transformers/distiluse-base-multilingual-cased-v2"
REVISOES_MODELOS = {
    MODELO_PADRAO: "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    MODELO_ALTERNATIVO: "bfe45d0732ca50787611c0fe107ba278c7f3f889",
}
REVISAO_MODELO_PADRAO = REVISOES_MODELOS[MODELO_PADRAO]
LIMIAR_DENSO_PADRAO = 0.80  # Escolhido somente em data/validacao.csv.
CACHE_PADRAO = Path(__file__).resolve().parent.parent / ".cache" / "embeddings"


def _versao_pacote() -> str:
    try:
        return version("sentence-transformers")
    except PackageNotFoundError:
        return "desconhecida"


def _revisao_resolvida(modelo, solicitada: str) -> str:
    """Obtém o commit do Hugging Face quando exposto pelo modelo carregado."""
    try:
        primeiro = modelo._first_module()
        config = primeiro.auto_model.config
        commit = getattr(config, "_commit_hash", None)
        if commit:
            return str(commit)
        caminho = Path(str(primeiro.auto_model.name_or_path))
        partes = caminho.parts
        if "snapshots" in partes:
            return partes[partes.index("snapshots") + 1]
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return str(getattr(modelo, "revisao", solicitada))


class ChatbotDenso:
    """Assistente denso compatível com a interface da v1.

    ``modelo`` é injetável para testes offline. Em uso normal, o modelo é
    carregado pela sentence-transformers explicitamente em CPU.
    """

    def __init__(
        self,
        caminho_base=BASE_PADRAO,
        limiar=LIMIAR_DENSO_PADRAO,
        modelo_nome=MODELO_PADRAO,
        revisao_modelo=REVISAO_MODELO_PADRAO,
        normalizar_antes=False,
        opcoes_normalizacao=None,
        diretorio_cache=CACHE_PADRAO,
        modelo=None,
    ):
        if not math.isfinite(limiar) or not -1 <= limiar <= 1:
            raise ValueError("O limiar denso deve estar entre -1 e 1.")
        self.limiar = float(limiar)
        self.modelo_nome = str(modelo_nome)
        self.revisao_solicitada = str(revisao_modelo)
        self.normalizar_antes = bool(normalizar_antes)
        self.opcoes_normalizacao = dict(opcoes_normalizacao or {})
        self.diretorio_cache = Path(diretorio_cache)
        self.base = carregar_base(caminho_base)
        self.pares_por_id = {par.id: par for par in self.base}

        if modelo is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as erro:
                raise RuntimeError(
                    "Instale sentence-transformers conforme o requirements.txt."
                ) from erro
            modelo = SentenceTransformer(
                self.modelo_nome, revision=self.revisao_solicitada, device="cpu"
            )
        self.modelo = modelo
        self.dispositivo = str(getattr(modelo, "device", "cpu"))
        self.revisao_resolvida = _revisao_resolvida(modelo, self.revisao_solicitada)
        self.versao_sentence_transformers = _versao_pacote()

        textos = [self._preparar_texto(par.pergunta) for par in self.base]
        self.chave_cache = self._calcular_chave_cache(textos)
        self.caminho_cache = self.diretorio_cache / f"base-{self.chave_cache}.npz"
        self.cache_reutilizado = self.caminho_cache.exists()
        if self.cache_reutilizado:
            self.vetores_base = self._ler_cache()
        else:
            self.vetores_base = self._codificar(textos)
            self._salvar_cache(self.vetores_base)

    def _preparar_texto(self, texto: str) -> str:
        if not self.normalizar_antes:
            return texto.strip()
        return " ".join(normalizar(texto, **self.opcoes_normalizacao))

    def _metadados_cache(self, textos) -> dict:
        pares = [
            {"id": par.id, "pergunta": par.pergunta, "resposta": par.resposta,
             "fonte": par.fonte}
            for par in self.base
        ]
        return {
            "formato": 1,
            "base": pares,
            "textos_indexados": textos,
            "modelo": self.modelo_nome,
            "revisao": self.revisao_resolvida,
            "sentence_transformers": self.versao_sentence_transformers,
            "normalizar_antes": self.normalizar_antes,
            "opcoes_normalizacao": self.opcoes_normalizacao,
        }

    def _calcular_chave_cache(self, textos) -> str:
        serializado = json.dumps(
            self._metadados_cache(textos), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serializado).hexdigest()[:20]

    def _codificar(self, textos) -> np.ndarray:
        vetores = self.modelo.encode(
            list(textos), batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False,
        )
        vetores = np.asarray(vetores, dtype=np.float32)
        if vetores.ndim != 2 or vetores.shape[0] != len(textos) or vetores.shape[1] == 0:
            raise ValueError("O modelo retornou embeddings com dimensões inválidas.")
        if not np.isfinite(vetores).all():
            raise ValueError("O modelo retornou embeddings não finitos.")
        return vetores

    def _ler_cache(self) -> np.ndarray:
        try:
            with np.load(self.caminho_cache, allow_pickle=False) as arquivo:
                vetores = np.asarray(arquivo["vetores"], dtype=np.float32)
                metadados = str(arquivo["metadados"].item())
            esperado = json.dumps(
                self._metadados_cache(
                    [self._preparar_texto(par.pergunta) for par in self.base]
                ), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            if metadados != esperado:
                raise ValueError("Metadados do cache não correspondem à configuração.")
            if vetores.ndim != 2 or vetores.shape[0] != len(self.base):
                raise ValueError("Dimensões inválidas no cache de embeddings.")
            return vetores
        except (OSError, KeyError, ValueError) as erro:
            raise ValueError(f"Cache de embeddings inválido: {self.caminho_cache}") from erro

    def _salvar_cache(self, vetores: np.ndarray) -> None:
        self.diretorio_cache.mkdir(parents=True, exist_ok=True)
        textos = [self._preparar_texto(par.pergunta) for par in self.base]
        metadados = json.dumps(
            self._metadados_cache(textos), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        )
        temporario = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".tmp", dir=self.diretorio_cache, delete=False
            ) as arquivo:
                temporario = Path(arquivo.name)
                np.savez_compressed(arquivo, vetores=vetores, metadados=metadados)
            os.replace(temporario, self.caminho_cache)
        finally:
            if temporario is not None and temporario.exists():
                temporario.unlink()

    def responder(self, texto: str) -> Resultado:
        if not isinstance(texto, str):
            raise TypeError("A pergunta deve ser uma string.")
        if (len(texto) > MAX_CARACTERES or not texto.strip()
                or not any(caractere.isalnum() for caractere in texto)):
            escores = np.zeros(len(self.base), dtype=float)
        else:
            preparado = self._preparar_texto(texto)
            if not preparado:
                escores = np.zeros(len(self.base), dtype=float)
            else:
                consulta = self._codificar([preparado])[0]
                escores = similaridade_cossenos(consulta, self.vetores_base)

        ranking = sorted(
            [(par.id, float(escore)) for par, escore in zip(self.base, escores)],
            key=lambda item: (-item[1], item[0]),
        )
        identificador, escore = ranking[0]
        if escore <= self.limiar:
            return Resultado(MENSAGEM_REJEICAO, -1, escore, ranking)
        return Resultado(
            self.pares_por_id[identificador].resposta, identificador, escore, ranking,
        )
