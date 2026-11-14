"""Interface de terminal do assistente de FAQ."""

import argparse

from src.base import BASE_PADRAO
from src.chatbot_denso import ChatbotDenso, LIMIAR_DENSO_PADRAO
from src.chatbot_rag import ChatbotRAG
from src.chatbot_tfidf import ChatbotTFIDF, LIMIAR_PADRAO, MAX_CARACTERES


def main():
    parser = argparse.ArgumentParser(description="Assistente de FAQ - Eleições 2026")
    parser.add_argument("--versao", choices=["v1", "v2", "rag"], default="v1")
    parser.add_argument("--limiar", type=float, default=None)
    parser.add_argument("--base", default=BASE_PADRAO)
    args = parser.parse_args()
    try:
        if args.versao == "v1":
            limiar = LIMIAR_PADRAO if args.limiar is None else args.limiar
            assistente = ChatbotTFIDF(args.base, limiar)
        elif args.versao == "v2":
            limiar = LIMIAR_DENSO_PADRAO if args.limiar is None else args.limiar
            assistente = ChatbotDenso(args.base, limiar)
        else:
            limiar = LIMIAR_DENSO_PADRAO if args.limiar is None else args.limiar
            assistente = ChatbotRAG(args.base, limiar)
    except (OSError, RuntimeError, ValueError, LookupError) as erro:
        parser.exit(1, f"Não foi possível iniciar o assistente: {erro}\n")

    print("FAQ Eleições 2026. Digite 'sair' para encerrar.")
    while True:
        try:
            pergunta = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if pergunta.strip().lower() == "sair":
            break
        resultado = assistente.responder(pergunta)
        if args.versao == "rag":
            recuperacao = resultado.recuperacao
            print(f"[rag/v2] (id={recuperacao.id}, "
                  f"escore={recuperacao.escore:.3f})")
            print(f"Original: {resultado.resposta_original}")
            if resultado.resposta_reformulada is not None:
                print(f"Reformulada: {resultado.resposta_reformulada}")
            elif resultado.erro is not None:
                print(f"Reformulação indisponível: {resultado.erro}")
            else:
                print("LLM não acionado: a pergunta foi rejeitada pela v2.")
        else:
            print(f"[{args.versao}] (id={resultado.id}, "
                  f"escore={resultado.escore:.3f}) {resultado.resposta}")
        if len(pergunta) > MAX_CARACTERES:
            print(f"Use uma pergunta de até {MAX_CARACTERES} caracteres.")


if __name__ == "__main__":
    main()
