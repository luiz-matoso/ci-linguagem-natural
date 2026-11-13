"""Interface de terminal do assistente de FAQ."""

import argparse

from base import BASE_PADRAO
from chatbot_tfidf import ChatbotTFIDF, LIMIAR_PADRAO, MAX_CARACTERES


def main():
    parser = argparse.ArgumentParser(description="Assistente de FAQ - Eleições 2026")
    parser.add_argument("--versao", choices=["v1"], default="v1")
    parser.add_argument("--limiar", type=float, default=LIMIAR_PADRAO)
    parser.add_argument("--base", default=BASE_PADRAO)
    args = parser.parse_args()
    try:
        assistente = ChatbotTFIDF(args.base, args.limiar)
    except (OSError, ValueError, LookupError) as erro:
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
        print(f"[v1] (id={resultado.id}, escore={resultado.escore:.3f}) "
              f"{resultado.resposta}")
        if len(pergunta) > MAX_CARACTERES:
            print(f"Use uma pergunta de até {MAX_CARACTERES} caracteres.")


if __name__ == "__main__":
    main()
