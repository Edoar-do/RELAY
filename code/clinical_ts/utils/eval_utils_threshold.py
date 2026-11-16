"""Threshold-based (operating-point) metrics for multi-label classification.

Operating points, chosen on the evaluated predictions themselves (one threshold per class):
  sens_at_spec<x>   sensitivity at the smallest score threshold t whose specificity(score > t) exceeds x
  spec_at_sens<x>   specificity at the largest score threshold t whose sensitivity(score > t) still exceeds x
                    (no such t: everything predicted positive, i.e. specificity 0 and sensitivity 1)
A class without positives or without negatives is undefined (NaN) and left out of the macro average (nanmean).

Fixed thresholds (e.g. chosen on the validation fold with --export-thresholds, applied to another set with
--fixed-thresholds): sensitivity, specificity, balanced accuracy (sens + spec) / 2 and PPV. PPV is computed from
sensitivity and specificity at the prevalence of the full evaluated sample (Bayes), so it stays at that prevalence
in every bootstrap resample.

All *_flat functions return a flat numpy array (fixed order given by the matching *_names function), which is the
format empirical_bootstrap expects.
"""

__all__ = ['threshold_at_spec', 'operating_point_names', 'operating_point_flat', 'fixed_threshold_names',
           'fixed_threshold_flat', 'spec_key']

import numpy as np


def spec_key(target):
    """0.9 -> 'spec90', 0.95 -> 'spec95'."""
    return f"spec{round(100 * float(target)):d}"


def _sens_key(target):
    return f"sens{round(100 * float(target)):d}"


def _curve(s, y):
    """Distinct score cuts u (ascending) and sens/spec of the rule 'positive = score > u'. None if undefined."""
    P, N = int(y.sum()), int((~y).sum())
    if P == 0 or N == 0:
        return None
    u = np.unique(s)
    cp = np.searchsorted(np.sort(s[y]), u, side="right")    # positives with score <= cut
    cn = np.searchsorted(np.sort(s[~y]), u, side="right")   # negatives with score <= cut
    return u, (P - cp) / P, cn / N


def threshold_at_spec(s, y, target):
    """Smallest distinct score t with specificity(score > t) > target (the largest score if none). NaN if undefined."""
    c = _curve(np.asarray(s), np.asarray(y, dtype=bool))
    if c is None:
        return np.nan
    u, _, spec = c
    return float(u[min(int((spec <= target).sum()), len(u) - 1)])


def _sens_at_spec(s, y, target):
    c = _curve(s, y)
    if c is None:
        return np.nan, np.nan
    _, sens, spec = c
    j = min(int((spec <= target).sum()), len(spec) - 1)
    return sens[j], spec[j]


def _spec_at_sens(s, y, target):
    c = _curve(s, y)
    if c is None:
        return np.nan, np.nan
    _, sens, spec = c
    j = int((sens > target).sum()) - 1
    return (0.0, 1.0) if j < 0 else (spec[j], sens[j])


def operating_point_names(classes, sens_at_spec=(0.90, 0.95), spec_at_sens=(0.80, 0.90)):
    """Metric names in the order of operating_point_flat: per operating point the macro, the macro of the quantity
    held fixed ('_achieved') and one value per class."""
    names = []
    for kind, targets in (("sens_at_", sens_at_spec), ("spec_at_", spec_at_sens)):
        for t in targets:
            op = kind + (spec_key(t) if kind == "sens_at_" else _sens_key(t))
            names += [op + "_macro", op + "_achieved_macro"] + [f"{op}_{c}" for c in classes]
    return names


def operating_point_flat(targs, preds, classes, sens_at_spec=(0.90, 0.95), spec_at_sens=(0.80, 0.90)):
    y_all = np.asarray(targs) > 0.5
    preds = np.asarray(preds)
    out = []
    for fn, targets in ((_sens_at_spec, sens_at_spec), (_spec_at_sens, spec_at_sens)):
        for t in targets:
            vals = np.array([fn(preds[:, i], y_all[:, i], t) for i in range(len(classes))], dtype=np.float64)
            with np.errstate(all="ignore"):
                macro = np.nanmean(vals[:, 0]) if np.isfinite(vals[:, 0]).any() else np.nan
                ach = np.nanmean(vals[:, 1]) if np.isfinite(vals[:, 1]).any() else np.nan
            out += [macro, ach] + list(vals[:, 0])
    return np.array(out, dtype=np.float64)


def fixed_threshold_names(thresholds, classes):
    """thresholds: {spec key: {class: t}}. Only classes present in both are evaluated."""
    names = []
    for key, thr in thresholds.items():
        for c in classes:
            if c in thr:
                names += [f"fixed_{key}_{m}_{c}" for m in ("sens", "spec", "ba", "ppv")]
    return names


def fixed_threshold_flat(targs, preds, classes, thresholds, prevalence):
    """prevalence: {class: prevalence of the full evaluated sample}, used for the PPV (Bayes)."""
    y_all = np.asarray(targs) > 0.5
    preds = np.asarray(preds)
    out = []
    for key, thr in thresholds.items():
        for i, c in enumerate(classes):
            if c not in thr:
                continue
            y, p = y_all[:, i], preds[:, i] > thr[c]
            P, N = y.sum(), (~y).sum()
            sens = (p & y).sum() / P if P else np.nan
            spec = 1 - (p & ~y).sum() / N if N else np.nan
            pi = prevalence[c]
            with np.errstate(all="ignore"):
                ppv = sens * pi / (sens * pi + (1 - spec) * (1 - pi))
            out += [sens, spec, (sens + spec) / 2, ppv]
    return np.array(out, dtype=np.float64)
