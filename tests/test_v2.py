"""Testes offline da v2: contrato, cache, invalidação e robustez."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.chatbot_denso import ChatbotDenso
from src.chatbot_tfidf import MAX_CARACTERES, MENSAGEM_REJEICAO


class ModeloFalso:
    device = "cpu"
    revisao = "revisao-teste-1"

    def __init__(self):
        self.chamadas = 0

    def encode(self, textos, **_):
        self.chamadas += 1
        vetores = []
        for texto in textos:
            texto = texto.lower()
            vetores.append([
                len(texto), texto.count("voto"), texto.count("turno"),
                texto.count("titulo"), sum(c.isdigit() for c in texto) + 1,
            ])
        return np.asarray(vetores, dtype=np.float32)


def escrever_base(caminho, pergunta2="Quando é o segundo turno?"):
    with caminho.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["id", "pergunta", "resposta", "fonte"])
        escritor.writerows([
            [1, "Como funciona o voto?", "Resposta 1", "fonte"],
            [2, pergunta2, "Resposta 2", "fonte"],
        ])


class TestChatbotDenso(unittest.TestCase):
    def criar(self, pasta, modelo=None, **opcoes):
        base = Path(pasta) / "base.csv"
        if not base.exists():
            escrever_base(base)
        limiar = opcoes.pop("limiar", 0.0)
        return ChatbotDenso(
            caminho_base=base, limiar=limiar, modelo=modelo or ModeloFalso(),
            modelo_nome=opcoes.pop("modelo_nome", "modelo/teste"),
            diretorio_cache=Path(pasta) / "cache", **opcoes,
        )

    def test_interface_ranking_completo_ordenado_e_cpu(self):
        with tempfile.TemporaryDirectory() as pasta:
            assistente = self.criar(pasta)
            resultado = assistente.responder("voto")
            self.assertIsInstance(resultado.resposta, str)
            self.assertIsInstance(resultado.id, int)
            self.assertIsInstance(resultado.escore, float)
            self.assertEqual(len(resultado.ranking), 2)
            self.assertEqual(resultado.ranking,
                             sorted(resultado.ranking, key=lambda x: (-x[1], x[0])))
            self.assertEqual(assistente.dispositivo, "cpu")

    def test_limiar_rejeita_em_igualdade_sem_perder_ranking(self):
        with tempfile.TemporaryDirectory() as pasta:
            assistente = self.criar(pasta)
            aceito = assistente.responder("voto")
            assistente.limiar = aceito.escore
            rejeitado = assistente.responder("voto")
            self.assertEqual(rejeitado.id, -1)
            self.assertEqual(rejeitado.resposta, MENSAGEM_REJEICAO)
            self.assertEqual(rejeitado.ranking, aceito.ranking)

    def test_cache_persiste_e_e_reutilizado(self):
        with tempfile.TemporaryDirectory() as pasta:
            primeiro_modelo = ModeloFalso()
            primeiro = self.criar(pasta, primeiro_modelo)
            self.assertFalse(primeiro.cache_reutilizado)
            self.assertEqual(primeiro_modelo.chamadas, 1)
            segundo_modelo = ModeloFalso()
            segundo = self.criar(pasta, segundo_modelo)
            self.assertTrue(segundo.cache_reutilizado)
            self.assertEqual(segundo_modelo.chamadas, 0)
            np.testing.assert_array_equal(primeiro.vetores_base, segundo.vetores_base)

    def test_cache_invalida_com_base_modelo_e_preprocessamento(self):
        with tempfile.TemporaryDirectory() as pasta:
            inicial = self.criar(pasta)
            outro_modelo = self.criar(pasta, modelo_nome="modelo/outro")
            normalizado = self.criar(pasta, normalizar_antes=True)
            base = Path(pasta) / "base.csv"
            escrever_base(base, "Qual a data do segundo turno?")
            base_alterada = self.criar(pasta)
            chaves = {
                inicial.chave_cache, outro_modelo.chave_cache,
                normalizado.chave_cache, base_alterada.chave_cache,
            }
            self.assertEqual(len(chaves), 4)
            self.assertTrue(all(not a.cache_reutilizado for a in
                                [inicial, outro_modelo, normalizado, base_alterada]))

    def test_entradas_especiais_nao_travam(self):
        with tempfile.TemporaryDirectory() as pasta:
            assistente = self.criar(pasta)
            for texto in ["", "!!!", "😀🚀", "x" * (MAX_CARACTERES + 1),
                          "Where do I vote?", "選挙について教えて"]:
                with self.subTest(texto=texto[:20]):
                    resultado = assistente.responder(texto)
                    self.assertTrue(np.isfinite(resultado.escore))
                    self.assertEqual(len(resultado.ranking), 2)
            self.assertEqual(assistente.responder("!!!").id, -1)

    def test_limiar_invalido(self):
        with tempfile.TemporaryDirectory() as pasta:
            for limiar in [-1.1, 1.1, float("nan"), float("inf")]:
                with self.subTest(limiar=limiar), self.assertRaises(ValueError):
                    self.criar(pasta, limiar=limiar)


if __name__ == "__main__":
    unittest.main()
