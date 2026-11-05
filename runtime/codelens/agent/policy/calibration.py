from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class CalibrationSample:
    score: float
    label: int


@dataclass(frozen=True)
class CalibrationResult:
    slope: float
    intercept: float
    sample_count: int
    positive_count: int
    nll: float
    brier: float
    ece: float


def _sigmoid(value: float) -> float:
    value = min(35.0, max(-35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def calibration_metrics(
    samples: Iterable[CalibrationSample], *, slope: float, intercept: float, bins: int = 10
) -> tuple[float, float, float]:
    rows = list(samples)
    if not rows:
        return 0.0, 0.0, 0.0
    nll = brier = 0.0
    buckets: List[List[tuple[float, int]]] = [[] for _ in range(max(1, bins))]
    for row in rows:
        probability = _sigmoid(slope * row.score + intercept)
        probability = min(1.0 - 1e-12, max(1e-12, probability))
        label = 1 if row.label else 0
        nll -= label * math.log(probability) + (1 - label) * math.log(1.0 - probability)
        brier += (probability - label) ** 2
        bucket_index = min(len(buckets) - 1, int(probability * len(buckets)))
        buckets[bucket_index].append((probability, label))
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        ece += len(bucket) / len(rows) * abs(confidence - accuracy)
    return nll / len(rows), brier / len(rows), ece


def fit_platt_scaling(
    samples: Iterable[CalibrationSample],
    *,
    initial_slope: float = 1.0,
    initial_intercept: float = 0.0,
    l2: float = 1e-3,
    max_iterations: int = 100,
) -> CalibrationResult:
    """Fit a two-parameter logistic calibrator with damped Newton updates."""
    rows = list(samples)
    if not rows:
        raise ValueError("At least one calibration sample is required")
    positives = sum(1 if row.label else 0 for row in rows)
    if positives in {0, len(rows)}:
        raise ValueError("Calibration requires both positive and negative samples")
    slope = float(initial_slope)
    intercept = float(initial_intercept)
    for _ in range(max(1, max_iterations)):
        grad_slope = l2 * slope
        grad_intercept = 0.0
        h_ss = l2
        h_si = 0.0
        h_ii = 1e-9
        for row in rows:
            x = float(row.score)
            y = 1.0 if row.label else 0.0
            p = _sigmoid(slope * x + intercept)
            error = p - y
            variance = max(1e-9, p * (1.0 - p))
            grad_slope += error * x
            grad_intercept += error
            h_ss += variance * x * x
            h_si += variance * x
            h_ii += variance
        determinant = h_ss * h_ii - h_si * h_si
        if abs(determinant) < 1e-12:
            break
        delta_slope = (h_ii * grad_slope - h_si * grad_intercept) / determinant
        delta_intercept = (-h_si * grad_slope + h_ss * grad_intercept) / determinant
        damping = 0.5
        slope -= damping * delta_slope
        intercept -= damping * delta_intercept
        if max(abs(delta_slope), abs(delta_intercept)) < 1e-7:
            break
    nll, brier, ece = calibration_metrics(rows, slope=slope, intercept=intercept)
    return CalibrationResult(
        slope=slope,
        intercept=intercept,
        sample_count=len(rows),
        positive_count=positives,
        nll=nll,
        brier=brier,
        ece=ece,
    )
