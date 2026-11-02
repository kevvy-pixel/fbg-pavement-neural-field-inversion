"""V4-A: leakage-safe Prony/Burgers-style temporal head feasibility study."""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PHASES = np.asarray([5.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0, 210.0])
EARLY = PHASES <= 90.0
LATE = PHASES > 90.0
STRAIN_FACTOR = 1.0e6 / 0.78


def log(message: str):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_enlight(path: Path):
    timestamps, channels = [], [[] for _ in range(4)]
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        in_data = False
        for line in handle:
            if line.startswith("Timestamp\t"):
                in_data = True
                continue
            if not in_data:
                continue
            tokens = line.rstrip().split("\t")
            if len(tokens) < 6:
                continue
            try:
                stamp = datetime.strptime(tokens[0], "%Y/%m/%d %H:%M:%S.%f")
                counts = [int(tokens[i]) for i in range(1, 5)]
                values = np.asarray(tokens[5:], dtype=float)
            except (ValueError, TypeError):
                continue
            if values.size != sum(counts):
                continue
            timestamps.append(stamp)
            cursor = 0
            for channel, count in enumerate(counts):
                channels[channel].append(values[cursor:cursor + count])
                cursor += count
    if not timestamps:
        raise ValueError(f"No samples parsed from {path}")
    seconds = np.asarray([(x - timestamps[0]).total_seconds() for x in timestamps])
    return seconds, channels


def fixed_channel(rows, count: int):
    result = np.full((len(rows), count), np.nan)
    for i, row in enumerate(rows):
        if len(row) >= count:
            result[i] = row[:count]
    return result


def track_variable_channel(rows, reference):
    reference = np.asarray(reference, dtype=float)
    result = np.full((len(rows), len(reference)), np.nan)
    combinations = {}
    target = reference.copy()
    for i, row in enumerate(rows):
        observed = np.asarray(row, dtype=float)
        n = len(observed)
        if n < len(reference):
            continue
        if n not in combinations:
            combinations[n] = np.asarray(
                list(itertools.combinations(range(n), len(reference))), dtype=int
            )
        candidates = observed[combinations[n]]
        residual = candidates - target[None, :]
        residual -= np.median(residual, axis=1, keepdims=True)
        selected = candidates[np.argmin(np.sum(residual * residual, axis=1))]
        result[i] = selected
        target = 0.995 * target + 0.005 * selected
    return result


def one_second_medians(seconds, values):
    bins = np.floor(seconds).astype(int)
    unique = np.unique(bins)
    output = np.vstack([np.nanmedian(values[bins == b], axis=0) for b in unique])
    return unique.astype(float), output


def detect_onset(seconds, asphalt):
    time_1s, values_1s = one_second_medians(seconds, asphalt)
    noise_window = values_1s[(time_1s >= 60) & (time_1s <= 220)]
    noise = np.nanstd(noise_window - np.nanmedian(noise_window, axis=0), axis=0)
    positive = noise[noise > 0]
    noise = np.maximum(noise, np.nanmedian(positive) if len(positive) else 1e-4)
    best = None
    for candidate in np.arange(240.0, 381.0):
        before = np.nanmedian(values_1s[(time_1s >= candidate - 20) & (time_1s <= candidate - 3)], axis=0)
        after = np.nanmedian(values_1s[(time_1s >= candidate + 3) & (time_1s <= candidate + 20)], axis=0)
        standardized = np.abs(after - before) / noise
        finite = standardized[np.isfinite(standardized)]
        score = float(np.mean(np.sort(finite)[-8:])) if finite.size >= 8 else -np.inf
        if best is None or score > best[0]:
            best = (score, candidate)
    return float(best[1]), float(best[0])


def window_median(seconds, values, start, stop):
    mask = (seconds >= start) & (seconds <= stop)
    if np.sum(mask) == 0:
        raise ValueError(f"Empty time window {start}-{stop}")
    return np.nanmedian(values[mask], axis=0)


def load_case(data_root: Path, case_id: str, d2_reference):
    d1_t, d1_rows = parse_enlight(data_root / "Demod1" / f"{case_id}.txt")
    d1 = np.column_stack([
        fixed_channel(d1_rows[0], 8), fixed_channel(d1_rows[1], 6),
        fixed_channel(d1_rows[2], 8), fixed_channel(d1_rows[3], 6),
    ])
    onset, onset_score = detect_onset(d1_t, d1)
    d2_t, d2_rows = parse_enlight(data_root / "Demod2" / f"{case_id}.txt")
    d2 = np.column_stack([
        track_variable_channel(d2_rows[0], d2_reference),
        fixed_channel(d2_rows[1], 6), fixed_channel(d2_rows[2], 8),
    ])
    response = []
    for seconds, values in ((d1_t, d1), (d2_t, d2)):
        baseline = window_median(seconds, values, max(30.0, onset - 60.0), onset - 10.0)
        phases = np.vstack([
            window_median(seconds, values, onset + p - 1.0, onset + p + 1.0)
            for p in PHASES
        ]).T
        response.append((phases - baseline[:, None]) / baseline[:, None] * STRAIN_FACTOR)
    return np.vstack(response), onset, onset_score


def metadata_features(response_table: pd.DataFrame, cases):
    rows = []
    keys = ["单点中心", "单点偏心", "双点对称", "双点不对称", "纵中心线"]
    for case in cases:
        part = response_table[response_table.case_id == case]
        if part.empty:
            raise KeyError(f"Missing metadata for {case}")
        r = part.iloc[0]
        text = str(r.load_type) + case
        p1x = float(r.p1_x) if pd.notna(r.p1_x) else 60.0
        p1y = float(r.p1_y) if pd.notna(r.p1_y) else 25.0
        p2x = float(r.p2_x) if pd.notna(r.p2_x) else p1x
        p2y = float(r.p2_y) if pd.notna(r.p2_y) else p1y
        rows.append([
            float(r.load_kg) / 80.0, p1x / 120.0, p1y / 50.0,
            p2x / 120.0, p2y / 50.0, float(pd.notna(r.p2_x)),
            *[float(key in text) for key in keys],
        ])
    return np.asarray(rows, dtype=float)


def rbf_predict(x_train, y_train, x_test, gamma, alpha):
    d_train = np.sum((x_train[:, None, :] - x_train[None, :, :]) ** 2, axis=2)
    d_test = np.sum((x_test[:, None, :] - x_train[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-gamma * d_train)
    weights = np.linalg.solve(kernel + alpha * np.eye(len(x_train)), y_train)
    return np.exp(-gamma * d_test) @ weights


def temporal_basis(kind: str, phases, tau=None):
    t = np.asarray(phases, dtype=float)
    scaled = t / 210.0
    if kind == "linear":
        return np.column_stack([np.ones_like(t), scaled])
    if kind == "prony1":
        return np.column_stack([np.ones_like(t), 1.0 - np.exp(-t / float(tau)), scaled])
    if kind == "prony2":
        tau1, tau2 = tau
        return np.column_stack([
            np.ones_like(t), 1.0 - np.exp(-t / tau1),
            1.0 - np.exp(-t / tau2), scaled,
        ])
    raise ValueError(kind)


def encode_coefficients(y, basis, phase_mask):
    pinv = np.linalg.pinv(basis[phase_mask], rcond=1e-6)
    coeff = np.einsum("kt,cst->csk", pinv, y[:, :, phase_mask])
    return coeff.reshape(len(y), -1)


def decode_coefficients(coeff, n_sensors, basis):
    reshaped = coeff.reshape(len(coeff), n_sensors, basis.shape[1])
    return np.einsum("csk,tk->cst", reshaped, basis)


def rmse(actual, predicted, sensor_slice=slice(None), phase_mask=slice(None)):
    error = actual[:, sensor_slice, phase_mask] - predicted[:, sensor_slice, phase_mask]
    return float(np.sqrt(np.nanmean(error * error)))


def evaluate_candidate(x_train, y_train, x_val, y_val, spec, gamma, alpha):
    family, kind, fit_mask, tau = spec
    if family == "direct":
        pred = rbf_predict(x_train, y_train.reshape(len(y_train), -1), x_val, gamma, alpha)
        return pred.reshape(y_val.shape)
    basis = temporal_basis(kind, PHASES, tau)
    coeff = encode_coefficients(y_train, basis, fit_mask)
    pred_coeff = rbf_predict(x_train, coeff, x_val, gamma, alpha)
    return decode_coefficients(pred_coeff, y_train.shape[1], basis)


def tune_model(x, y, fold_ids, outer_fold, spec, gammas, alphas):
    outer_train = fold_ids != outer_fold
    inner_folds = sorted(set(fold_ids[outer_train]))
    best = None
    for gamma in gammas:
        for alpha in alphas:
            errors = []
            for inner in inner_folds:
                inner_val = fold_ids == inner
                inner_train = outer_train & ~inner_val
                prediction = evaluate_candidate(
                    x[inner_train], y[inner_train], x[inner_val], y[inner_val],
                    spec, gamma, alpha,
                )
                mask = LATE if spec[0] == "structured_early" else np.ones(len(PHASES), dtype=bool)
                errors.append(rmse(y[inner_val], prediction, phase_mask=mask))
            score = float(np.mean(errors))
            if best is None or score < best[0]:
                best = (score, gamma, alpha)
    return best


def build_specs(smoke=False):
    tau1 = [40.0] if smoke else [10.0, 20.0, 40.0, 80.0, 160.0]
    tau2 = [(20.0, 120.0)] if smoke else [(10.0, 60.0), (20.0, 120.0), (40.0, 160.0)]
    full = np.ones(len(PHASES), dtype=bool)
    specs = [("direct", "direct", full, None), ("structured_early", "linear", EARLY, None)]
    specs += [("structured_full", "prony1", full, t) for t in tau1]
    specs += [("structured_full", "prony2", full, t) for t in tau2]
    specs += [("structured_early", "prony1", EARLY, t) for t in tau1]
    specs += [("structured_early", "prony2", EARLY, t) for t in tau2]
    return specs


def model_label(spec):
    family, kind, _, tau = spec
    suffix = "early" if family.endswith("early") else "full"
    if family == "direct":
        return "direct_full"
    if tau is None:
        return f"{kind}_{suffix}"
    tau_text = "-".join(str(int(x)) for x in tau) if isinstance(tau, tuple) else str(int(tau))
    return f"{kind}_{suffix}_tau{tau_text}"


def model_family_label(spec):
    family, kind, _, _ = spec
    suffix = "early" if family.endswith("early") else "full"
    return "direct_full" if family == "direct" else f"{kind}_{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path(__file__).with_name("fixed_case_split.csv"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "RUNNING").write_text(datetime.now().isoformat(), encoding="utf-8")

    split = pd.read_csv(args.split)
    development = split[split.v4_role == "development"].copy()
    cases = development.case_id.tolist()
    fold_ids = development.v4_dev_fold.astype(int).to_numpy()
    cache = args.output / "development_dynamic_dataset_v4a.npz"
    if cache.exists() and not args.rebuild_cache:
        log(f"Loading development-only cache: {cache}")
        loaded = np.load(cache, allow_pickle=False)
        if loaded["cases"].astype(str).tolist() != cases:
            raise RuntimeError("Cached case order does not match frozen split")
        y = loaded["response_ue"]
        onsets, onset_scores = loaded["onsets_s"], loaded["onset_scores"]
    else:
        log("Extracting 17 development cases; fixed test raw files will not be opened")
        baseline = pd.read_csv(args.data_root / "step1,2" / "Unified_Baseline_Table_Demod2.csv")
        d2_reference = baseline[baseline.Channel == 1].sort_values("SensorIndex").UnifiedBaseline_nm.to_numpy()
        responses, onsets, onset_scores = [], [], []
        for index, case in enumerate(cases, 1):
            log(f"Data {index:02d}/{len(cases)}: {case}")
            response, onset, score = load_case(args.data_root, case, d2_reference)
            responses.append(response)
            onsets.append(onset)
            onset_scores.append(score)
        y = np.stack(responses)
        onsets, onset_scores = np.asarray(onsets), np.asarray(onset_scores)
        np.savez_compressed(
            cache, cases=np.asarray(cases), phases_s=PHASES, response_ue=y,
            onsets_s=onsets, onset_scores=onset_scores,
        )
    if y.shape != (17, 50, len(PHASES)) or not np.all(np.isfinite(y)):
        raise RuntimeError(f"Invalid development tensor: shape={y.shape}, finite={np.all(np.isfinite(y))}")

    metadata = pd.read_csv(args.data_root / "Response_Matrix_Table.csv")
    x = metadata_features(metadata, cases)
    pd.DataFrame({
        "case_id": cases, "dev_fold": fold_ids, "onset_s": onsets,
        "onset_score": onset_scores,
        "max_abs_response_ue": np.max(np.abs(y), axis=(1, 2)),
    }).to_csv(args.output / "data_audit.csv", index=False, encoding="utf-8-sig")
    log(f"Dataset ready: {y.shape}; onset range {onsets.min():.0f}-{onsets.max():.0f} s")

    gammas = [0.3] if args.smoke else [0.1, 0.3, 1.0, 3.0]
    alphas = [0.1] if args.smoke else [1e-4, 1e-2, 1e-1, 1.0]
    specs = build_specs(args.smoke)
    outer_folds = [1] if args.smoke else sorted(set(fold_ids))
    rows, prediction_rows, selection_rows = [], [], []
    for outer in outer_folds:
        train, valid = fold_ids != outer, fold_ids == outer
        log(f"Outer fold {outer}: train={np.sum(train)}, validation={np.sum(valid)}")
        selected = {}
        for spec in specs:
            inner_score, gamma, alpha = tune_model(x, y, fold_ids, outer, spec, gammas, alphas)
            label, key = model_label(spec), (spec[0], spec[1])
            selection_rows.append({
                "outer_fold": outer, "candidate": label, "family": spec[0],
                "inner_rmse_ue": inner_score, "gamma": gamma, "alpha": alpha,
            })
            if key not in selected or inner_score < selected[key][0]:
                selected[key] = (inner_score, spec, gamma, alpha)
        for inner_score, spec, gamma, alpha in selected.values():
            label = model_label(spec)
            family_label = model_family_label(spec)
            prediction = evaluate_candidate(x[train], y[train], x[valid], y[valid], spec, gamma, alpha)
            scoring = LATE if spec[0] == "structured_early" else np.ones(len(PHASES), dtype=bool)
            result = {
                "outer_fold": outer, "model_family": family_label, "model": label,
                "inner_rmse_ue": inner_score,
                "gamma": gamma, "alpha": alpha,
                "overall_rmse_ue": rmse(y[valid], prediction, phase_mask=scoring),
                "asphalt_rmse_ue": rmse(y[valid], prediction, slice(0, 28), scoring),
                "soil_rmse_ue": rmse(y[valid], prediction, slice(28, 50), scoring),
                "early_rmse_ue": rmse(y[valid], prediction, phase_mask=EARLY),
                "late_rmse_ue": rmse(y[valid], prediction, phase_mask=LATE),
            }
            rows.append(result)
            log(f"  {label}: late={result['late_rmse_ue']:.3f} ue, overall={result['overall_rmse_ue']:.3f} ue")
            for local, case_index in enumerate(np.where(valid)[0]):
                for sensor in range(50):
                    for phase_index, phase in enumerate(PHASES):
                        prediction_rows.append({
                            "outer_fold": outer, "model_family": family_label, "model": label,
                            "case_id": cases[case_index],
                            "layer": "Asphalt" if sensor < 28 else "Soil",
                            "sensor_index": sensor + 1, "phase_s": phase,
                            "actual_ue": y[case_index, sensor, phase_index],
                            "predicted_ue": prediction[local, sensor, phase_index],
                        })
        pd.DataFrame(rows).to_csv(args.output / "fold_metrics_partial.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(rows)
    selections = pd.DataFrame(selection_rows)
    predictions = pd.DataFrame(prediction_rows)
    prefix = "smoke_" if args.smoke else ""
    metrics.to_csv(args.output / f"{prefix}fold_metrics.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(args.output / f"{prefix}model_selection.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(args.output / f"{prefix}dev_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary = metrics.groupby("model_family", as_index=False).agg(
        folds=("outer_fold", "nunique"), overall_rmse_ue=("overall_rmse_ue", "mean"),
        asphalt_rmse_ue=("asphalt_rmse_ue", "mean"), soil_rmse_ue=("soil_rmse_ue", "mean"),
        early_rmse_ue=("early_rmse_ue", "mean"), late_rmse_ue=("late_rmse_ue", "mean"),
    ).sort_values("late_rmse_ue")
    summary.to_csv(args.output / f"{prefix}summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "scope": "development-only smoke" if args.smoke else "development-only four-fold OOF",
        "fixed_test_opened": False, "development_cases": len(cases), "fixed_test_cases": 5,
        "phases_s": PHASES.tolist(), "best_by_late_rmse": summary.iloc[0].to_dict(),
    }
    (args.output / f"{prefix}summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.smoke:
        (args.output / "DONE").write_text(datetime.now().isoformat(), encoding="utf-8")
        (args.output / "RUNNING").unlink(missing_ok=True)
    log("Completed successfully")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
