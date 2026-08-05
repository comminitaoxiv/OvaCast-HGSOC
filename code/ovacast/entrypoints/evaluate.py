import argparse
import json
from pathlib import Path

import numpy as np

from ovacast.measures.survival import concordance_index


def main() -> None:
    parser = argparse.ArgumentParser(prog="ovacast-evaluate")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    rows = [
        json.loads(line)
        for line in arguments.predictions.read_text(encoding="utf-8").splitlines()
        if line
    ]
    times = np.asarray([row["survival_months"] for row in rows])
    events = np.asarray([row["event"] for row in rows])
    risks = np.asarray([row["risk"] for row in rows])
    score = concordance_index(times, risks, events)
    arguments.output.write_text(
        json.dumps({"concordance_index": score}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
