import re

from ovacast.records.types import ClinicalProfile, PatientRecord, RadiologyProfile


IMAGE_ABSENT = "[IMAGE MODALITY ABSENT(CT): CT imaging unavailable. All predictions are based on genomic and textual (narrative) data only.]"
TEXT_ABSENT = "[TEXT MODALITY ABSENT(PATHOLOGY): Pathology report unavailable.]"
TASK_SURVIVAL = (
    "Predict the 5-year overall survival risk and provide a rationale for this patient with HGSOC."
)
TASK_SUBTYPE = "Classify the CLOVAR transcriptional subtype for this patient with HGSOC."
TASK_PLATINUM = "Predict platinum sensitivity for this patient with HGSOC."


def clinical_token(profile: ClinicalProfile) -> str:
    return f"Age: {profile.age:.0f}. FIGO Stage: {profile.stage}. Histological Grade: {profile.grade}. Debulking: {profile.debulking}."


def radiology_token(profile: RadiologyProfile | None) -> str:
    if profile is None:
        return IMAGE_ABSENT
    fields = (
        ("Primary ovarian mass", profile.ovarian_mass),
        ("Peritoneal spread", profile.peritoneal_spread),
        ("Mesenteric infiltration", profile.mesenteric_infiltration),
        ("Other implant sites", profile.implants),
        ("Effusion or ascites", profile.effusion_ascites),
        ("Lymphadenopathy", profile.lymphadenopathy),
        ("Distant metastases", profile.distant_metastases),
    )
    return " ".join(f"{name}: {value}." for name, value in fields)


def pathology_token(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if compact else TEXT_ABSENT


def explanation_reference(record: PatientRecord, genomic_tokens: tuple[str, ...]) -> str:
    evidence = " ".join(genomic_tokens[:4])
    imaging = radiology_token(record.radiology)
    event_phrase = "observed adverse outcome" if record.event else "censored follow-up"
    return f"Risk assessment: evaluate five-year overall survival. Key genomic findings: {evidence} Clinical and imaging evidence: {clinical_token(record.clinical)} {imaging} Prognostic statement: integrate these findings with {event_phrase} at {record.survival_months:.1f} months."


def assemble(
    record: PatientRecord, genomic_tokens: tuple[str, ...], task: str = TASK_SURVIVAL
) -> str:
    sections = (
        task,
        clinical_token(record.clinical),
        pathology_token(record.pathology),
        radiology_token(record.radiology),
        *genomic_tokens,
    )
    return "\n".join(sections)
