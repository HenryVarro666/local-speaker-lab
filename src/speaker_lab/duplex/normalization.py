"""Construct a weight-one control without changing selected audio or targets."""

from copy import deepcopy


def validate_loss_contrast(weighted: dict, uniform: dict) -> None:
    left, right = weighted["entries"], uniform["entries"]
    keys = [r["audio"] for r in left]
    if not left or len(set(keys)) != len(keys):
        raise ValueError("Empty or duplicate audio keys")
    if [(r["audio"], r["target"]) for r in left] != [
        (r["audio"], r["target"]) for r in right
    ]:
        raise ValueError("Control changed selected audio or targets")
    if not any(r["spans"] for r in left) or any(r["spans"] for r in right):
        raise ValueError(
            "Contrast requires nonempty weighted spans and empty uniform spans"
        )


def uniform_sidecar(weighted: dict) -> dict:
    result = deepcopy(weighted)
    for row in result["entries"]:
        row["spans"] = []
    validate_loss_contrast(weighted, result)
    return result
