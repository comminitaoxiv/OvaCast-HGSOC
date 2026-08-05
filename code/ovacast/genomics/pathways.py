import csv
import json
from pathlib import Path

from ovacast.records.types import Pathway


def load_gmt(path: Path, source: str) -> tuple[Pathway, ...]:
    pathways = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 12:
                continue
            identifier = row[0]
            name = row[1]
            genes = tuple(dict.fromkeys(g.strip().upper() for g in row[2:] if g.strip()))
            if len(genes) >= 10:
                pathways.append(Pathway(identifier, name, genes, source))
    return tuple(pathways)


def load_json(path: Path) -> tuple[Pathway, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        Pathway(str(item["identifier"]), str(item["name"]), tuple(item["genes"]), item["source"])
        for item in payload
        if len(item["genes"]) >= 10
    )


def save_json(pathways: tuple[Pathway, ...], path: Path) -> None:
    payload = [
        {"identifier": p.identifier, "name": p.name, "genes": list(p.genes), "source": p.source}
        for p in pathways
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def select_non_overlapping(pathways: tuple[Pathway, ...], limit: int = 330) -> tuple[Pathway, ...]:
    ranked = sorted(pathways, key=lambda p: (-len(p.genes), p.identifier))
    selected = []
    assigned = set()
    for pathway in ranked:
        remaining = tuple(g for g in pathway.genes if g not in assigned)
        if len(remaining) < 10:
            continue
        selected.append(Pathway(pathway.identifier, pathway.name, remaining, pathway.source))
        assigned.update(remaining)
        if len(selected) == limit:
            break
    return tuple(selected)
