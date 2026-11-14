"""Calibração em validação e comparação reprodutível entre v1 e v2."""

import argparse
import csv
from pathlib import Path

from src.chatbot_denso import (
    ChatbotDenso, LIMIAR_DENSO_PADRAO, MODELO_ALTERNATIVO, MODELO_PADRAO,
    REVISAO_MODELO_PADRAO, REVISOES_MODELOS,
)
from src.chatbot_tfidf import ChatbotTFIDF, LIMIAR_PADRAO


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
        escritor = csv.DictWriter(
            arquivo, fieldnames=list(linhas[0]), lineterminator="\n"
        )
        escritor.writeheader()
        escritor.writerows(linhas)


def criar_assistente(versao, limiar, modelo_v2=MODELO_PADRAO,
                     normalizar_v2=False):
    if versao == "v1":
        return ChatbotTFIDF(limiar=limiar)
    return ChatbotDenso(
        limiar=limiar, modelo_nome=modelo_v2,
        revisao_modelo=REVISAO_MODELO_PADRAO,
        normalizar_antes=normalizar_v2,
    )


def comparar_configuracoes_v2():
    """Compara duas arquiteturas e a ablação de normalização só na validação."""
    perguntas = carregar_perguntas(RAIZ / "data" / "validacao.csv")
    linhas = []
    for modelo in [MODELO_PADRAO, MODELO_ALTERNATIVO]:
        for normalizar_antes in [False, True]:
            assistente = ChatbotDenso(
                limiar=0.0, modelo_nome=modelo,
                revisao_modelo=REVISOES_MODELOS[modelo],
                normalizar_antes=normalizar_antes,
            )
            resultados = obter_resultados(assistente, perguntas)
            limiar, _ = calibrar_limiar(resultados)
            metricas = calcular_metricas(resultados, limiar)
            dentro = [(e, r) for e, r in resultados if e != -1]
            acerto_aceito = sum(
                r.ranking[0][0] == e and r.escore > limiar for e, r in dentro
            ) / len(dentro)
            linhas.append({
                "modelo": modelo,
                "revisao_resolvida": assistente.revisao_resolvida,
                "sentence_transformers": assistente.versao_sentence_transformers,
                "normalizar_antes": normalizar_antes,
                "limiar": limiar,
                "acerto_aceito_no_escopo": acerto_aceito,
                "objetivo": (acerto_aceito + metricas["rejeicao_correta"]) / 2,
                **metricas,
            })
    salvar_csv(RAIZ / "resultados" / "experimentos_v2.csv", linhas)
    return linhas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrar", action="store_true",
                        help="Usa somente data/validacao.csv para escolher o limiar.")
    parser.add_argument("--experimentos-v2", action="store_true",
                        help="Compara modelos e normalização somente na validação.")
    parser.add_argument("--versao", choices=["v1", "v2", "todas"], default="todas")
    parser.add_argument("--limiar-v1", type=float, default=LIMIAR_PADRAO)
    parser.add_argument("--limiar-v2", type=float, default=LIMIAR_DENSO_PADRAO)
    args = parser.parse_args()
    try:
        if args.experimentos_v2:
            for linha in comparar_configuracoes_v2():
                print(
                    f"{linha['modelo']} normalizar={linha['normalizar_antes']} "
                    f"limiar={linha['limiar']:.2f} objetivo={linha['objetivo']:.4f}"
                )
            return
        if args.calibrar:
            # Este caminho não abre data/teste.csv.
            perguntas = carregar_perguntas(RAIZ / "data" / "validacao.csv")
            versoes = ["v1", "v2"] if args.versao == "todas" else [args.versao]
            for versao in versoes:
                inicial = args.limiar_v1 if versao == "v1" else args.limiar_v2
                assistente = criar_assistente(versao, inicial)
                resultados = obter_resultados(assistente, perguntas)
                limiar, linhas = calibrar_limiar(resultados)
                nome = "calibracao.csv" if versao == "v1" else "calibracao_v2.csv"
                salvar_csv(RAIZ / "resultados" / nome, linhas)
                print(f"{versao}: limiar escolhido na validação: {limiar:.2f}")
        else:
            perguntas = carregar_perguntas(RAIZ / "data" / "teste.csv")
            versoes = ["v1", "v2"] if args.versao == "todas" else [args.versao]
            linhas = []
            for versao in versoes:
                limiar = args.limiar_v1 if versao == "v1" else args.limiar_v2
                assistente = criar_assistente(versao, limiar)
                resultados = obter_resultados(assistente, perguntas)
                metricas = calcular_metricas(resultados, limiar)
                linhas.append({"versao": versao, "limiar": limiar, **metricas})
                print(versao)
                for nome, valor in metricas.items():
                    print(f"  {nome}: {valor:.4f}")
            salvar_csv(RAIZ / "resultados" / "comparacao.csv", linhas)
    except (OSError, RuntimeError, ValueError, LookupError) as erro:
        parser.exit(1, f"Não foi possível avaliar: {erro}\n")


if __name__ == "__main__":
    main()
