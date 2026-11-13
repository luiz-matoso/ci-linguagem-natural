"""Leitura da base de conhecimento compartilhada pelas versões do assistente."""

import csv
from dataclasses import dataclass
from pathlib import Path


BASE_PADRAO = Path(__file__).resolve().parent / "data" / "base.csv"


@dataclass(frozen=True)
class ParFAQ:
    id: int
    pergunta: str
    resposta: str
    fonte: str


def carregar_base(caminho=BASE_PADRAO) -> list[ParFAQ]:
    pares = []
    ids = set()
    # utf-8-sig aceita tanto o CSV original com BOM quanto arquivos sem BOM.
    with Path(caminho).open(encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        campos = {"id", "pergunta", "resposta", "fonte"}
        if not campos.issubset(leitor.fieldnames or []):
            raise ValueError("A base deve conter id, pergunta, resposta e fonte.")
        for linha in leitor:
            if any(not (linha[campo] or "").strip() for campo in campos):
                raise ValueError("A base contém um campo obrigatório vazio.")
            identificador = int(linha["id"])
            if identificador <= 0 or identificador in ids:
                raise ValueError("Os IDs da base devem ser positivos e únicos.")
            ids.add(identificador)
            pares.append(ParFAQ(
                identificador, linha["pergunta"].strip(),
                linha["resposta"].strip(), linha["fonte"].strip(),
            ))
    if not pares:
        raise ValueError("A base de conhecimento está vazia.")
    return pares
