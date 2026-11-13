"""Calibração do limiar em validação e avaliação reprodutível da v1."""

import argparse
import csv
from pathlib import Path

from chatbot_tfidf import ChatbotTFIDF, LIMIAR_PADRAO


RAIZ = Path(__file__).resolve().parent


def carregar_perguntas(caminho) -> list[tuple[str, int]]:
    with Path(caminho).open(encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        if not {"pergunta", "id_esperado"}.issubset(leitor.fieldnames or []):
            raise ValueError("Esperadas as colunas pergunta e id_esperado.")
        perguntas = []
        for linha in leitor:
            if not (linha["pergunta"] or "").strip():
                raise ValueError("O conjunto contém uma pergunta vazia.")
            perguntas.append((linha["pergunta"], int(linha["id_esperado"])))
    if not perguntas:
        raise ValueError("O conjunto de avaliação está vazio.")
    return perguntas


def obter_resultados(assistente, perguntas):
    ids_base = {par.id for par in assistente.base}
    if any(esperado != -1 and esperado not in ids_base for _, esperado in perguntas):
        raise ValueError("O conjunto referencia um ID inexistente na base.")
    return [(esperado, assistente.responder(texto)) for texto, esperado in perguntas]


def calcular_metricas(resultados, limiar):
    dentro = [(esperado, resultado) for esperado, resultado in resultados if esperado != -1]
    fora = [resultado for esperado, resultado in resultados if esperado == -1]
    if not dentro or not fora:
        raise ValueError("A avaliação exige perguntas dentro e fora do escopo.")

    # Acurácia@1 e MRR avaliam o ranking, independentemente da rejeição.
    acuracia = sum(r.ranking[0][0] == esperado for esperado, r in dentro) / len(dentro)
    reciprocos = []
    for esperado, resultado in dentro:
        reciprocos.append(next(
            (1 / posicao for posicao, (identificador, _) in enumerate(resultado.ranking, 1)
             if identificador == esperado), 0.0,
        ))
    return {
        "acuracia_1": acuracia,
        "mrr": sum(reciprocos) / len(dentro),
        "rejeicao_correta": sum(r.escore <= limiar for r in fora) / len(fora),
        "falsa_rejeicao": sum(r.escore <= limiar for _, r in dentro) / len(dentro),
    }


def calibrar_limiar(resultados):
    dentro = [(esperado, r) for esperado, r in resultados if esperado != -1]
    candidatos = []
    # A grade e o critério são fixados antes de observar o conjunto de teste.
    for passo in range(21):
        limiar = passo / 20
        metricas = calcular_metricas(resultados, limiar)
        acerto = sum(r.ranking[0][0] == esperado and r.escore > limiar
                     for esperado, r in dentro) / len(dentro)
        candidatos.append({
            "limiar": limiar,
            "acerto_aceito_no_escopo": acerto,
            "objetivo": (acerto + metricas["rejeicao_correta"]) / 2,
            **metricas,
        })
    # Empates no objetivo favorecem o menor limiar, para reduzir falsas rejeições.
    melhor = max(candidatos, key=lambda linha: (linha["objetivo"], -linha["limiar"]))
    return melhor["limiar"], candidatos


def salvar_csv(caminho, linhas):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrar", action="store_true",
                        help="Usa somente data/validacao.csv para escolher o limiar.")
    parser.add_argument("--limiar", type=float, default=LIMIAR_PADRAO)
    args = parser.parse_args()
    try:
        assistente = ChatbotTFIDF(limiar=args.limiar)
        if args.calibrar:
            # Este caminho não abre data/teste.csv.
            perguntas = carregar_perguntas(RAIZ / "data" / "validacao.csv")
            resultados = obter_resultados(assistente, perguntas)
            limiar, linhas = calibrar_limiar(resultados)
            salvar_csv(RAIZ / "resultados" / "calibracao.csv", linhas)
            print(f"Limiar escolhido na validação: {limiar:.2f}")
        else:
            perguntas = carregar_perguntas(RAIZ / "data" / "teste.csv")
            resultados = obter_resultados(assistente, perguntas)
            metricas = calcular_metricas(resultados, args.limiar)
            salvar_csv(RAIZ / "resultados" / "comparacao.csv", [{
                "versao": "v1", "limiar": args.limiar, **metricas,
            }])
            for nome, valor in metricas.items():
                print(f"{nome}: {valor:.4f}")
    except (OSError, ValueError, LookupError) as erro:
        parser.exit(1, f"Não foi possível avaliar: {erro}\n")


if __name__ == "__main__":
    main()
