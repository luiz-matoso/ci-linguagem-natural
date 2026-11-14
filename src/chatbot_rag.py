"""Bônus: reformulação por LLM da resposta recuperada pela v2."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .base import BASE_PADRAO
from .chatbot_denso import ChatbotDenso, LIMIAR_DENSO_PADRAO
from .chatbot_tfidf import Resultado


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELO_RAG_PADRAO = "nvidia/nemotron-3.5-lightning:free"
INSTRUCAO_RAG = """Você é um assistente de perguntas frequentes.

Reformule a resposta original de maneira clara e natural em português,
considerando a pergunta do usuário. Utilize exclusivamente as informações da
resposta original. Não acrescente regras, números, prazos ou outros fatos.
Responda somente com a reformulação."""

MENSAGEM_CHAVE_AUSENTE = (
    "OPENROUTER_API_KEY não configurada. A resposta original foi preservada."
)
MENSAGEM_FALHA_API = (
    "Não foi possível reformular a resposta pela API. "
    "A resposta original foi preservada."
)
MENSAGEM_RESPOSTA_VAZIA = (
    "O LLM retornou uma resposta vazia. A resposta original foi preservada."
)


@dataclass(frozen=True)
class ResultadoRAG:
    recuperacao: Resultado
    resposta_original: str
    resposta_reformulada: str | None
    erro: str | None = None


class _ChaveAusente(RuntimeError):
    pass


load_dotenv()


class ChatbotRAG:
    """Encadeia a recuperação da v2 com uma única reformulação por LLM."""

    def __init__(
        self,
        caminho_base=BASE_PADRAO,
        limiar=LIMIAR_DENSO_PADRAO,
        modelo_llm=None,
        assistente_v2=None,
        cliente_llm=None,
    ):
        self.modelo_llm = str(
            modelo_llm
            or os.environ.get("OPENROUTER_MODEL", "").strip()
            or MODELO_RAG_PADRAO
        )
        self.assistente_v2 = assistente_v2 or ChatbotDenso(
            caminho_base=caminho_base, limiar=limiar
        )
        self._cliente_llm = cliente_llm

    def _obter_cliente(self):
        if self._cliente_llm is not None:
            return self._cliente_llm

        chave = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not chave:
            raise _ChaveAusente(MENSAGEM_CHAVE_AUSENTE)

        try:
            from openai import OpenAI
        except ImportError as erro:
            raise RuntimeError(
                "Instale o SDK openai conforme o requirements.txt."
            ) from erro

        self._cliente_llm = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=chave,
            max_retries=0,
            timeout=30.0,
        )
        return self._cliente_llm

    def responder(self, pergunta: str) -> ResultadoRAG:
        recuperacao = self.assistente_v2.responder(pergunta)
        original = recuperacao.resposta

        # A rejeição da v2 encerra o fluxo antes de criar o cliente ou chamar a API.
        if recuperacao.id == -1:
            return ResultadoRAG(recuperacao, original, None)

        try:
            cliente = self._obter_cliente()
            resposta = cliente.chat.completions.create(
                model=self.modelo_llm,
                messages=[
                    {"role": "system", "content": INSTRUCAO_RAG},
                    {
                        "role": "user",
                        "content": (
                            f"Pergunta original:\n{pergunta}\n\n"
                            f"Resposta original recuperada:\n{original}"
                        ),
                    },
                ],
                max_completion_tokens=200,
                reasoning_effort="none",
            )
        except _ChaveAusente as erro:
            return ResultadoRAG(recuperacao, original, None, str(erro))
        except Exception:
            return ResultadoRAG(recuperacao, original, None, MENSAGEM_FALHA_API)

        try:
            conteudo = resposta.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            conteudo = None
        reformulada = (conteudo or "").strip()
        if not reformulada:
            return ResultadoRAG(
                recuperacao, original, None, MENSAGEM_RESPOSTA_VAZIA
            )
        return ResultadoRAG(recuperacao, original, reformulada)
