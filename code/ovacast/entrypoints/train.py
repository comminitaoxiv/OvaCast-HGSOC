import argparse
import logging
from pathlib import Path

import yaml


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="ovacast-train")
    value.add_argument("--config", type=Path, default=Path("settings/main.yaml"))
    value.add_argument("--cohort", type=Path, required=True)
    value.add_argument("--pathways", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    arguments = parser().parse_args()
    configuration = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(
        "Configuration loaded for %s with %d training phases",
        configuration["model"]["identifier"],
        len(configuration["training"]["phases"]),
    )


if __name__ == "__main__":
    main()
