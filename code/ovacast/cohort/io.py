import csv
import json
from pathlib import Path

from ovacast.records.types import ClinicalProfile, PatientRecord, RadiologyProfile


def read_expression(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or reader.fieldnames[0] != "gene":
            raise ValueError("expression table must begin with a gene column")
        samples = {sample: {} for sample in reader.fieldnames[1:]}
        for row in reader:
            gene = row["gene"].strip().upper()
            for sample in samples:
                value = row.get(sample, "")
                if value not in (None, "", "NA"):
                    samples[sample][gene] = float(value)
    return samples


def read_records(path: Path, expression: dict[str, dict[str, float]]) -> list[PatientRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for item in payload:
        patient_id = str(item["patient_id"])
        clinical_data = item["clinical"]
        clinical = ClinicalProfile(
            float(clinical_data["age"]),
            str(clinical_data["stage"]),
            str(clinical_data["grade"]),
            str(clinical_data["debulking"]),
        )
        radiology_data = item.get("radiology")
        radiology = None
        if radiology_data is not None:
            radiology = RadiologyProfile(
                *(
                    str(radiology_data[key])
                    for key in (
                        "ovarian_mass",
                        "peritoneal_spread",
                        "mesenteric_infiltration",
                        "implants",
                        "effusion_ascites",
                        "lymphadenopathy",
                        "distant_metastases",
                    )
                )
            )
        records.append(
            PatientRecord(
                patient_id,
                expression[patient_id],
                float(item["survival_months"]),
                int(item["event"]),
                clinical,
                str(item.get("pathology", "")),
                radiology,
                tuple(item.get("mutations", ())),
                tuple(item.get("copy_number", ())),
                item.get("subtype"),
                item.get("platinum_response"),
            )
        )
    return records
