"""修正数据上的零目标观测Direct RBF多输出场模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
GAMMAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
SPECIMEN_LENGTH_CM = 120.0
SPECIMEN_WIDTH_CM = 50.0
WAVELENGTH_SENSITIVITY_NM_PER_UE = 1.2e-3


def load_corrected_dataset(path: Path):
    raw = pd.read_csv(path)
    required = {
        "case_id", "sensor_id", "layer", "direction", "delta_lambda_final_nm",
        "load_kg", "load_type", "p1_x", "p1_y", "p2_x", "p2_y",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"修正数据缺少字段：{', '.join(missing)}")
    asphalt = raw[raw.layer.eq("Asphalt")].copy()
    if asphalt.case_id.nunique() != 22 or len(asphalt) != 616:
        raise ValueError(
            f"期望22个工况、616条沥青记录，实际为"
            f"{asphalt.case_id.nunique()}个工况、{len(asphalt)}条记录"
        )
    field = (
        asphalt.pivot(index="case_id", columns="sensor_id", values="delta_lambda_final_nm")
        .sort_index()
        / WAVELENGTH_SENSITIVITY_NM_PER_UE
    )
    metadata = raw.groupby("case_id").first().loc[field.index]
    return raw, field, metadata


def build_features(metadata: pd.DataFrame, load_types: list[str]) -> np.ndarray:
    p1x = metadata.p1_x.to_numpy(float) / SPECIMEN_LENGTH_CM
    p1y = metadata.p1_y.to_numpy(float) / SPECIMEN_WIDTH_CM
    p2x = metadata.p2_x.fillna(metadata.p1_x).to_numpy(float) / SPECIMEN_LENGTH_CM
    p2y = metadata.p2_y.fillna(metadata.p1_y).to_numpy(float) / SPECIMEN_WIDTH_CM
    has_second = metadata.p2_x.notna().to_numpy().astype(float)
    load = metadata.load_kg.to_numpy(float) / 40.0
    separation = np.hypot(p2x - p1x, p2y - p1y) * has_second
    edge1 = np.minimum.reduce([p1x, 1 - p1x, p1y, 1 - p1y])
    edge2 = np.minimum.reduce([p2x, 1 - p2x, p2y, 1 - p2y]) * has_second
    one_hot = np.column_stack(
        [(metadata.load_type.to_numpy() == value).astype(float) for value in load_types]
    )
    return np.column_stack(
        [
            load, p1x, p1y, p2x, p2y, has_second, separation, edge1, edge2,
            load * p1x, load * p1y, load * has_second, one_hot,
        ]
    )


def standardize(x_train: np.ndarray, x_test: np.ndarray):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (x_train - mean) / scale, (x_test - mean) / scale, mean, scale


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float) -> np.ndarray:
    squared_distance = ((x1[:, None, :] - x2[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * squared_distance)


def fit_rbf(x: np.ndarray, y: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    kernel = rbf_kernel(x, x, gamma)
    return np.linalg.solve(kernel + alpha * np.eye(len(x)), y)


def predict_rbf(
    x_train: np.ndarray, weights: np.ndarray, x_test: np.ndarray, gamma: float
) -> np.ndarray:
    return rbf_kernel(x_test, x_train, gamma) @ weights


def tune_parameters(x: np.ndarray, y: np.ndarray):
    best = None
    for gamma in GAMMAS:
        for alpha in ALPHAS:
            squared_error = 0.0
            count = 0
            for held_out in range(len(y)):
                train_mask = np.arange(len(y)) != held_out
                x_train, x_valid, _, _ = standardize(x[train_mask], x[[held_out]])
                weights = fit_rbf(x_train, y[train_mask], gamma, alpha)
                prediction = predict_rbf(x_train, weights, x_valid, gamma)[0]
                squared_error += float(np.sum((prediction - y[held_out]) ** 2))
                count += y.shape[1]
            rmse = float(np.sqrt(squared_error / count))
            if best is None or rmse < best[0]:
                best = (rmse, gamma, alpha)
    return best


def read_fold_map(path: Path, cases: np.ndarray) -> np.ndarray:
    table = pd.read_csv(path).set_index("case_id")
    missing = sorted(set(cases).difference(table.index))
    if missing:
        raise ValueError(f"固定分折缺少工况：{missing}")
    folds = table.loc[cases, "fold"].to_numpy(int)
    if set(folds) != {1, 2, 3, 4, 5}:
        raise ValueError("固定分折必须包含1--5折")
    return folds


def cross_validate(data_path: Path, fold_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _, field, metadata = load_corrected_dataset(data_path)
    cases = field.index.to_numpy()
    sensors = field.columns.to_numpy()
    load_types = sorted(metadata.load_type.unique().tolist())
    x = build_features(metadata, load_types)
    y = field.to_numpy()
    folds = read_fold_map(fold_path, cases)
    directions = np.asarray(
        ["Longitudinal" if sensor.startswith(("D1_CH1", "D1_CH3")) else "Transverse" for sensor in sensors]
    )
    predictions = []
    tuning = []
    for fold in range(1, 6):
        train = folds != fold
        test = folds == fold
        inner_rmse, gamma, alpha = tune_parameters(x[train], y[train])
        x_train, x_test, _, _ = standardize(x[train], x[test])
        weights = fit_rbf(x_train, y[train], gamma, alpha)
        predicted = predict_rbf(x_train, weights, x_test, gamma)
        tuning.append(
            {"fold": fold, "inner_rmse_ue": inner_rmse, "gamma": gamma, "alpha": alpha}
        )
        for local_index, case_index in enumerate(np.flatnonzero(test)):
            for sensor_index, sensor in enumerate(sensors):
                predictions.append(
                    {
                        "fold": fold,
                        "case_id": cases[case_index],
                        "sensor_id": sensor,
                        "direction": directions[sensor_index],
                        "actual_ue": y[case_index, sensor_index],
                        "predicted_ue": predicted[local_index, sensor_index],
                    }
                )
    prediction_table = pd.DataFrame(predictions)
    prediction_table.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tuning).to_csv(output / "nested_tuning.csv", index=False, encoding="utf-8-sig")
    summary = []
    for direction in ("all", "Longitudinal", "Transverse"):
        part = prediction_table if direction == "all" else prediction_table[prediction_table.direction.eq(direction)]
        error = part.predicted_ue - part.actual_ue
        summary.append(
            {
                "direction": direction,
                "n": len(part),
                "rmse_ue": float(np.sqrt(np.mean(error**2))),
                "mae_ue": float(np.mean(np.abs(error))),
            }
        )
    summary_table = pd.DataFrame(summary)
    summary_table.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary_table.to_string(index=False))


def fit_final(data_path: Path, model_path: Path) -> None:
    _, field, metadata = load_corrected_dataset(data_path)
    load_types = sorted(metadata.load_type.unique().tolist())
    x = build_features(metadata, load_types)
    y = field.to_numpy()
    inner_rmse, gamma, alpha = tune_parameters(x, y)
    x_standardized, _, mean, scale = standardize(x, x[:1])
    weights = fit_rbf(x_standardized, y, gamma, alpha)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        train_x=x_standardized,
        weights=weights,
        feature_mean=mean,
        feature_scale=scale,
        gamma=np.asarray(gamma),
        alpha=np.asarray(alpha),
        inner_rmse_ue=np.asarray(inner_rmse),
        load_types=np.asarray(load_types),
        # 显式保存为 Unicode 数组，确保 allow_pickle=False 时仍可安全加载。
        sensor_ids=np.asarray(field.columns.astype(str).tolist(), dtype=str),
    )
    metadata_json = {
        "model": "Direct RBF多输出场模型",
        "task": "新工况零沥青目标观测",
        "training_cases": int(len(field)),
        "outputs": int(field.shape[1]),
        "gamma": gamma,
        "alpha": alpha,
        "inner_rmse_ue": inner_rmse,
        "target": "delta_lambda_final_nm / 0.0012",
    }
    model_path.with_suffix(".json").write_text(
        json.dumps(metadata_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata_json, ensure_ascii=False, indent=2))


def predict(model_path: Path, metadata_path: Path, output_path: Path) -> None:
    model = np.load(model_path, allow_pickle=False)
    metadata = pd.read_csv(metadata_path)
    required = {"case_id", "load_kg", "load_type", "p1_x", "p1_y", "p2_x", "p2_y"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"预测元数据缺少字段：{', '.join(missing)}")
    load_types = model["load_types"].astype(str).tolist()
    x = build_features(metadata, load_types)
    x = (x - model["feature_mean"]) / model["feature_scale"]
    prediction = predict_rbf(model["train_x"], model["weights"], x, float(model["gamma"]))
    rows = []
    for case_index, case_id in enumerate(metadata.case_id.astype(str)):
        for sensor_index, sensor_id in enumerate(model["sensor_ids"].astype(str)):
            rows.append(
                {
                    "case_id": case_id,
                    "sensor_id": sensor_id,
                    "predicted_ue": prediction[case_index, sensor_index],
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cv = subparsers.add_parser("cross-validate", help="执行固定五折嵌套验证")
    cv.add_argument("data", type=Path)
    cv.add_argument("--folds", type=Path, default=Path(__file__).with_name("fixed_case_folds.csv"))
    cv.add_argument("--output", type=Path, default=Path("results/direct_rbf_cv"))
    fit = subparsers.add_parser("fit", help="使用全部22工况训练并保存部署模型")
    fit.add_argument("data", type=Path)
    fit.add_argument("--model", type=Path, default=Path("models/direct_rbf_corrected_v2.npz"))
    inference = subparsers.add_parser("predict", help="对新工况元数据预测28点沥青响应")
    inference.add_argument("model", type=Path)
    inference.add_argument("metadata", type=Path)
    inference.add_argument("--output", type=Path, default=Path("predictions.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "cross-validate":
        cross_validate(args.data, args.folds, args.output)
    elif args.command == "fit":
        fit_final(args.data, args.model)
    else:
        predict(args.model, args.metadata, args.output)


if __name__ == "__main__":
    main()
