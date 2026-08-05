import torch

from ovacast.records.types import NormalizationState


class FrozenGeneNormalizer:
    def __init__(self) -> None:
        self.state: NormalizationState | None = None

    def fit(self, samples: list[dict[str, float]], genes: tuple[str, ...]) -> NormalizationState:
        if not samples:
            raise ValueError("training samples are empty")
        matrix = torch.tensor([[sample.get(g, float("nan")) for g in genes] for sample in samples])
        means = torch.nanmean(matrix, dim=0)
        centered = matrix - means
        count = torch.sum(~torch.isnan(matrix), dim=0).clamp_min(2)
        variances = torch.nansum(centered.square(), dim=0) / (count - 1)
        deviations = torch.sqrt(variances).clamp_min(1e-8)
        self.state = NormalizationState(genes, means, deviations)
        return self.state

    def transform(self, sample: dict[str, float]) -> dict[str, float]:
        if self.state is None:
            raise RuntimeError("normalizer has not been fitted")
        values = torch.tensor([sample.get(g, float("nan")) for g in self.state.genes])
        values = torch.where(torch.isnan(values), self.state.means, values)
        zscores = (values - self.state.means) / self.state.standard_deviations
        return dict(zip(self.state.genes, zscores.tolist(), strict=True))

    def state_dict(self) -> dict[str, object]:
        if self.state is None:
            raise RuntimeError("normalizer has not been fitted")
        return {
            "genes": self.state.genes,
            "means": self.state.means,
            "standard_deviations": self.state.standard_deviations,
        }

    def load_state_dict(self, payload: dict[str, object]) -> None:
        genes = tuple(str(x) for x in payload["genes"])
        means = torch.as_tensor(payload["means"])
        deviations = torch.as_tensor(payload["standard_deviations"])
        self.state = NormalizationState(genes, means, deviations)
