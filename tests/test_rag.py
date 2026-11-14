"""Testes offline da integração RAG, sem chamadas pagas à API."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.chatbot_rag import (
    ChatbotRAG, MENSAGEM_CHAVE_AUSENTE, MENSAGEM_FALHA_API,
    MENSAGEM_RESPOSTA_VAZIA, OPENROUTER_BASE_URL,
)
from src.chatbot_tfidf import MENSAGEM_REJEICAO, Resultado


class AssistenteV2Falso:
    def __init__(self, resultado):
        self.resultado = resultado
        self.perguntas = []

    def responder(self, pergunta):
        self.perguntas.append(pergunta)
        return self.resultado


class ConclusoesFalsas:
    def __init__(self, texto="Resposta reformulada.", erro=None):
        self.texto = texto
        self.erro = erro
        self.chamadas = []

    def create(self, **argumentos):
        self.chamadas.append(argumentos)
        if self.erro is not None:
            raise self.erro
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.texto)
        )])


class ChatFalso:
    def __init__(self, **opcoes):
        self.completions = ConclusoesFalsas(**opcoes)


class ClienteFalso:
    def __init__(self, **opcoes):
        self.chat = ChatFalso(**opcoes)


def resultado_aceito():
    return Resultado("Resposta original da base.", 7, 0.91, [(7, 0.91)])


def resultado_rejeitado():
    return Resultado(MENSAGEM_REJEICAO, -1, 0.20, [(7, 0.20)])


class TestChatbotRAG(unittest.TestCase):
    def test_envia_pergunta_e_resposta_original_e_retorna_reformulacao(self):
        v2 = AssistenteV2Falso(resultado_aceito())
        cliente = ClienteFalso(texto="Uma reformulação clara.")
        rag = ChatbotRAG(assistente_v2=v2, cliente_llm=cliente)

        resultado = rag.responder("Pergunta do usuário?")

        self.assertEqual(resultado.resposta_original, "Resposta original da base.")
        self.assertEqual(resultado.resposta_reformulada, "Uma reformulação clara.")
        self.assertIsNone(resultado.erro)
        self.assertEqual(v2.perguntas, ["Pergunta do usuário?"])
        self.assertEqual(len(cliente.chat.completions.chamadas), 1)
        argumentos = cliente.chat.completions.chamadas[0]
        self.assertEqual(
            argumentos["model"], "nvidia/nemotron-3.5-lightning:free"
        )
        self.assertEqual(argumentos["max_completion_tokens"], 200)
        self.assertEqual(argumentos["reasoning_effort"], "none")
        self.assertIn("exclusivamente", argumentos["messages"][0]["content"])
        dados = argumentos["messages"][1]["content"]
        self.assertIn("Pergunta do usuário?", dados)
        self.assertIn("Resposta original da base.", dados)

    def test_rejeicao_nao_chama_api(self):
        cliente = ClienteFalso()
        rag = ChatbotRAG(
            assistente_v2=AssistenteV2Falso(resultado_rejeitado()),
            cliente_llm=cliente,
        )

        resultado = rag.responder("Pergunta fora do escopo")

        self.assertEqual(resultado.recuperacao.id, -1)
        self.assertEqual(resultado.resposta_original, MENSAGEM_REJEICAO)
        self.assertIsNone(resultado.resposta_reformulada)
        self.assertIsNone(resultado.erro)
        self.assertEqual(cliente.chat.completions.chamadas, [])

    def test_configura_chave_modelo_e_endpoint_openrouter(self):
        ambiente = {
            "OPENROUTER_API_KEY": "chave-de-teste",
            "OPENROUTER_MODEL": "provedor/modelo:free",
        }
        with patch.dict(os.environ, ambiente, clear=True), patch(
            "openai.OpenAI"
        ) as construtor:
            rag = ChatbotRAG(
                assistente_v2=AssistenteV2Falso(resultado_aceito())
            )
            rag._obter_cliente()

        self.assertEqual(rag.modelo_llm, "provedor/modelo:free")
        construtor.assert_called_once_with(
            base_url=OPENROUTER_BASE_URL,
            api_key="chave-de-teste",
            max_retries=0,
            timeout=30.0,
        )

    def test_chave_ausente_preserva_resposta_original(self):
        rag = ChatbotRAG(assistente_v2=AssistenteV2Falso(resultado_aceito()))
        with patch.dict(os.environ, {}, clear=True):
            resultado = rag.responder("Pergunta")

        self.assertEqual(resultado.resposta_original, "Resposta original da base.")
        self.assertIsNone(resultado.resposta_reformulada)
        self.assertEqual(resultado.erro, MENSAGEM_CHAVE_AUSENTE)

    def test_falha_do_provedor_nao_derruba_programa(self):
        cliente = ClienteFalso(erro=RuntimeError("falha simulada"))
        rag = ChatbotRAG(
            assistente_v2=AssistenteV2Falso(resultado_aceito()),
            cliente_llm=cliente,
        )

        resultado = rag.responder("Pergunta")

        self.assertEqual(resultado.resposta_original, "Resposta original da base.")
        self.assertIsNone(resultado.resposta_reformulada)
        self.assertEqual(resultado.erro, MENSAGEM_FALHA_API)

    def test_resposta_vazia_preserva_original(self):
        cliente = ClienteFalso(texto="  ")
        rag = ChatbotRAG(
            assistente_v2=AssistenteV2Falso(resultado_aceito()),
            cliente_llm=cliente,
        )

        resultado = rag.responder("Pergunta")

        self.assertIsNone(resultado.resposta_reformulada)
        self.assertEqual(resultado.erro, MENSAGEM_RESPOSTA_VAZIA)


if __name__ == "__main__":
    unittest.main()
