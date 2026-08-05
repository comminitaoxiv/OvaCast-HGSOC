import argparse
from pathlib import Path

from ovacast.genomics.pathways import load_gmt, save_json, select_non_overlapping


def main() -> None:
    parser = argparse.ArgumentParser(prog="ovacast-prepare")
    parser.add_argument("--kegg", type=Path, required=True)
    parser.add_argument("--reactome", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    combined = load_gmt(arguments.kegg, "KEGG") + load_gmt(arguments.reactome, "Reactome")
    save_json(select_non_overlapping(combined), arguments.output)


if __name__ == "__main__":
    main()
