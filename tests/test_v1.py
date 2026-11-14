"""Testes do contrato da v1, da matemática e da avaliação."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from avaliacao import (
    calcular_metricas, calibrar_limiar, carregar_perguntas, obter_resultados,
    comparar_stemming_v1,
)
from src.base import carregar_base
from src.chatbot_tfidf import (
    ChatbotTFIDF, MAX_CARACTERES, MENSAGEM_REJEICAO, Resultado, similaridade_cossenos,
)


class TestCosseno(unittest.TestCase):
    def test_confere_com_sklearn_inclusive_vetores_nulos(self):
        gerador = np.random.default_rng(42)
        documentos = gerador.normal(size=(15, 8))
        documentos[0] = 0
        for consulta in [gerador.normal(size=8), np.zeros(8)]:
            with self.subTest(consulta=consulta):
                np.testing.assert_allclose(
                    similaridade_cossenos(consulta, documentos),
                    cosine_similarity(consulta.reshape(1, -1), documentos)[0],
                    atol=1e-12, rtol=1e-12,
                )

    def test_dimensoes_incompativeis(self):
        with self.assertRaises(ValueError):
            similaridade_cossenos([1, 2], [[1, 2, 3]])


class TestAssistente(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistente = ChatbotTFIDF()

    def test_perguntas_originais_recuperam_seus_ids(self):
        for par in self.assistente.base:
            with self.subTest(id=par.id):
                resultado = self.assistente.responder(par.pergunta)
                self.assertEqual(resultado.id, par.id)
                self.assertEqual(resultado.resposta, par.resposta)
                self.assertAlmostEqual(resultado.escore, 1.0)

    def test_ranking_completo_e_ordenado(self):
        resultado = self.assistente.responder("Quando será o segundo turno?")
        self.assertEqual({identificador for identificador, _ in resultado.ranking},
                         {par.id for par in self.assistente.base})
        self.assertEqual(resultado.ranking,
                         sorted(resultado.ranking, key=lambda item: (-item[1], item[0])))

    def test_cosseno_da_base_tfidf_confere_com_sklearn(self):
        consulta = self.assistente.vetorizador.transform(["Como tirar meu título?"]).toarray()[0]
        np.testing.assert_allclose(
            similaridade_cossenos(consulta, self.assistente.vetores_base),
            cosine_similarity([consulta], self.assistente.vetores_base)[0],
            atol=1e-12, rtol=1e-12,
        )

    def test_entradas_sem_informacao_sao_rejeitadas_com_ranking(self):
        for pergunta in ["", "  \n ", "!!! ??? ...", "😀🚀", "zxqwvblorp",
                         "a " * MAX_CARACTERES]:
            with self.subTest(pergunta=pergunta[:30]):
                resultado = self.assistente.responder(pergunta)
                self.assertEqual(resultado.id, -1)
                self.assertEqual(resultado.resposta, MENSAGEM_REJEICAO)
                self.assertEqual(resultado.escore, 0.0)
                self.assertEqual(len(resultado.ranking), len(self.assistente.base))

    def test_outro_idioma_e_texto_no_limite_nao_travam(self):
        for pergunta in ["Where can I vote?", "選挙について教えて", "voto " * 2000]:
            resultado = self.assistente.responder(pergunta)
            self.assertTrue(np.isfinite(resultado.escore))
            self.assertEqual(len(resultado.ranking), len(self.assistente.base))

    def test_consulta_nao_altera_vocabulario_ou_idf(self):
        vocabulario = dict(self.assistente.vetorizador.vocabulary_)
        idf = self.assistente.vetorizador.idf_.copy()
        self.assistente.responder("palavrainventadaquenaoestanafaq")
        self.assertEqual(self.assistente.vetorizador.vocabulary_, vocabulario)
        np.testing.assert_array_equal(self.assistente.vetorizador.idf_, idf)

    def test_limiar_invalido(self):
        for limiar in [-0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(limiar=limiar), self.assertRaises(ValueError):
                ChatbotTFIDF(limiar=limiar)

    def test_empate_respostas_duplicadas_e_limiar_estrito(self):
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "base.csv"
            with caminho.open("w", encoding="utf-8-sig", newline="") as arquivo:
                escritor = csv.writer(arquivo)
                escritor.writerow(["id", "pergunta", "resposta", "fonte"])
                escritor.writerows([
                    [8, "Como votar?", "Resposta compartilhada", "fonte"],
                    [2, "Como votar?", "Resposta compartilhada", "fonte"],
                ])
            assistente = ChatbotTFIDF(caminho, limiar=0)
            resultado = assistente.responder("Como votar?")
            self.assertEqual(resultado.id, 2)
            self.assertEqual([item[0] for item in resultado.ranking], [2, 8])
            assistente.limiar = resultado.escore
            rejeitado = assistente.responder("Como votar?")
            self.assertEqual(rejeitado.id, -1)
            self.assertEqual(rejeitado.ranking, resultado.ranking)
            self.assertEqual(assistente.responder("zxqwvblorp").id, -1)

    def test_base_invalida(self):
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "base.csv"
            for conteudo in [
                "pergunta,resposta\na,b\n",
                "id,pergunta,resposta,fonte\n",
                "id,pergunta,resposta,fonte\n1,pergunta,,fonte\n",
                "id,pergunta,resposta,fonte\n1,p,r,f\n1,p,r,f\n",
            ]:
                with self.subTest(conteudo=conteudo):
                    caminho.write_text(conteudo, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        carregar_base(caminho)

    def test_normalizacao_configuravel_e_vocabulario_vazio(self):
        assistente = ChatbotTFIDF(opcoes_normalizacao={"aplicar_stemming": False})
        self.assertIn("eleicoes", assistente.vetorizador.vocabulary_)
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "base.csv"
            caminho.write_text("id,pergunta,resposta,fonte\n1,!!!,Resposta,Fonte\n",
                               encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "vocabulário"):
                ChatbotTFIDF(caminho)


class TestAvaliacao(unittest.TestCase):
    def test_ablacao_usa_so_validacao_e_preserva_limiar(self):
        # A ablação deve funcionar sem ter acesso ao conjunto de teste.
        raiz = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            (destino / "data").mkdir()
            (destino / "data" / "validacao.csv").write_bytes(
                (raiz / "data" / "validacao.csv").read_bytes()
            )
            with patch("avaliacao.RAIZ", destino), patch(
                "avaliacao.calibrar_limiar",
                side_effect=AssertionError("A ablação deve manter o limiar fixo."),
            ):
                linhas = comparar_stemming_v1()
            with (destino / "resultados" / "ablacao_v1.csv").open() as arquivo:
                gravadas = list(csv.DictReader(arquivo))
            self.assertEqual(len(gravadas), 2)
            self.assertEqual([r["aplicar_stemming"] for r in gravadas], ["True", "False"])
            self.assertTrue(all(r["limiar"] == "0.7" for r in gravadas))
            # Confere as métricas exportadas por uma avaliação direta independente.
            perguntas = carregar_perguntas(destino / "data" / "validacao.csv")
            for linha in linhas:
                bot = ChatbotTFIDF(
                    limiar=0.7,
                    opcoes_normalizacao={"aplicar_stemming": linha["aplicar_stemming"]},
                )
                esperado = calcular_metricas(obter_resultados(bot, perguntas), 0.7)
                for metrica, valor in esperado.items():
                    self.assertAlmostEqual(linha[metrica], valor)

    def test_metricas_separam_ranking_e_rejeicao(self):
        resultados = [
            (1, Resultado("rejeição", -1, 0.3, [(1, 0.3), (2, 0.2)])),
            (2, Resultado("resposta", 1, 0.8, [(1, 0.8), (2, 0.7)])),
            (-1, Resultado("rejeição", -1, 0.1, [(1, 0.1), (2, 0.0)])),
        ]
        self.assertEqual(calcular_metricas(resultados, 0.4), {
            "acuracia_1": 0.5, "mrr": 0.75,
            "rejeicao_correta": 1.0, "falsa_rejeicao": 0.5,
        })

    def test_calibracao_escolhe_menor_limiar_empatado(self):
        resultados = [
            (1, Resultado("resposta", 1, 0.8, [(1, 0.8)])),
            (-1, Resultado("resposta", 1, 0.2, [(1, 0.2)])),
        ]
        limiar, candidatos = calibrar_limiar(resultados)
        self.assertEqual(limiar, 0.2)
        self.assertEqual(len(candidatos), 21)

    def test_conjuntos_separados_e_teste_com_tamanho_exigido(self):
        raiz = Path(__file__).resolve().parents[1]
        validacao = carregar_perguntas(raiz / "data" / "validacao.csv")
        teste = carregar_perguntas(raiz / "data" / "teste.csv")
        base = carregar_base()
        self.assertEqual(sum(esperado != -1 for _, esperado in teste), 30)
        self.assertEqual(sum(esperado == -1 for _, esperado in teste), 5)
        self.assertTrue({p for p, _ in teste}.isdisjoint(p for p, _ in validacao))
        self.assertTrue({p for p, _ in teste + validacao}.isdisjoint(p.pergunta for p in base))

    def test_avaliacao_recusa_id_inexistente(self):
        with self.assertRaisesRegex(ValueError, "ID inexistente"):
            obter_resultados(ChatbotTFIDF(), [("pergunta", 999)])


if __name__ == "__main__":
    unittest.main()
