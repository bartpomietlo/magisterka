#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video features and train a binary classifier."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("kod/dataset"),
        help="Root with ai/auth folders (default: kod/dataset).",
    )
    parser.add_argument(
        "--ai-folder",
        type=str,
        default="ai_baseline",
        help="Subfolder with AI videos (label=1).",
    )
    parser.add_argument(
        "--auth-folder",
        type=str,
        default="authentic",
        help="Subfolder with authentic videos (label=0).",
    )
    parser.add_argument(
        "--fallback-auth-folder",
        type=str,
        default="adv_fp_trap",
        help="Fallback subfolder if auth-folder does not exist.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=8,
        help="Minimum sampled frames per video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Maximum sampled frames per video.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=12,
        help="Target sampled frames per video (clamped to min/max).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Where to save outputs (default: current directory).",
    )
    return parser.parse_args()


def pick_frame_count(total_frames: int, min_frames: int, max_frames: int, target: int) -> int:
    target = max(min_frames, min(target, max_frames))
    if total_frames <= 0:
        return target
    return min(max_frames, max(min_frames, min(total_frames, target)))


def sample_indices(total_frames: int, n_samples: int) -> np.ndarray:
    if total_frames <= 1:
        return np.zeros(n_samples, dtype=int)
    return np.linspace(0, total_frames - 1, num=n_samples, dtype=int)


def build_embedder(device: torch.device) -> tuple[torch.nn.Module, transforms.Compose]:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.eval()
    model.to(device)
    preprocess = weights.transforms()
    return model, preprocess


def frame_embedding(model: torch.nn.Module, preprocess: transforms.Compose, frame_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess(frame_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        x = model.features(tensor)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
    return x.squeeze(0).detach().cpu().numpy().astype(np.float32)


def mean_optical_flow_magnitude(frames_bgr: list[np.ndarray]) -> float:
    if len(frames_bgr) < 2:
        return 0.0
    mags: list[float] = []
    prev_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    for cur_bgr in frames_bgr[1:]:
        gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(float(np.mean(mag)))
        prev_gray = gray
    return float(np.mean(mags)) if mags else 0.0


def optional_audio_features(_video_path: Path) -> dict[str, float]:
    # Audio jest opcjonalne. Domyślnie zwracamy 0, żeby nie blokować pipeline.
    return {"audio_mfcc_mean": 0.0, "audio_mfcc_std": 0.0}


def extract_video_features(
    video_path: Path,
    label: int,
    model: torch.nn.Module,
    preprocess: transforms.Compose,
    device: torch.device,
    min_frames: int,
    max_frames: int,
    target_frames: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie mozna otworzyc pliku: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_samples = pick_frame_count(total_frames, min_frames, max_frames, target_frames)
    indices = sample_indices(total_frames, n_samples)

    sampled_frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        sampled_frames.append(frame)
    cap.release()

    if not sampled_frames:
        raise RuntimeError(f"Brak odczytanych klatek: {video_path}")

    embeddings = np.stack(
        [frame_embedding(model, preprocess, frm, device) for frm in sampled_frames], axis=0
    )
    emb_mean = embeddings.mean(axis=0)
    emb_std = embeddings.std(axis=0)
    emb_min = embeddings.min(axis=0)
    emb_max = embeddings.max(axis=0)
    temporal_var = float(np.mean(np.var(embeddings, axis=0)))
    mean_flow = mean_optical_flow_magnitude(sampled_frames)
    audio_feats = optional_audio_features(video_path)

    row: dict[str, Any] = {
        "filename": video_path.name,
        "label": label,
        "n_sampled_frames": len(sampled_frames),
        "mean_optical_flow_magnitude": mean_flow,
        "temporal_embedding_variance": temporal_var,
        **audio_feats,
    }

    for i, v in enumerate(emb_mean):
        row[f"emb_mean_{i:04d}"] = float(v)
    for i, v in enumerate(emb_std):
        row[f"emb_std_{i:04d}"] = float(v)
    for i, v in enumerate(emb_min):
        row[f"emb_min_{i:04d}"] = float(v)
    for i, v in enumerate(emb_max):
        row[f"emb_max_{i:04d}"] = float(v)
    return row


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.resolve()
    ai_dir = dataset_root / args.ai_folder
    auth_dir = dataset_root / args.auth_folder
    if not auth_dir.exists():
        fallback = dataset_root / args.fallback_auth_folder
        if fallback.exists():
            print(
                f"[INFO] Folder '{args.auth_folder}' nie istnieje, uzywam fallback: "
                f"'{args.fallback_auth_folder}'."
            )
            auth_dir = fallback
        else:
            raise FileNotFoundError(
                f"Brak folderu autentycznych: {auth_dir} i fallback: {fallback}"
            )

    ai_files = sorted(ai_dir.glob("*.mp4"))
    auth_files = sorted(auth_dir.glob("*.mp4"))
    if not ai_files or not auth_files:
        raise RuntimeError(
            f"Brak plikow .mp4. AI={len(ai_files)} ({ai_dir}), AUTH={len(auth_files)} ({auth_dir})"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device}")
    model, preprocess = build_embedder(device)

    all_entries: list[tuple[Path, int, str]] = (
        [(p, 1, "ai_baseline") for p in ai_files]
        + [(p, 0, "authentic") for p in auth_files]
    )
    print(f"[INIT] Videos to process: {len(all_entries)} (AI={len(ai_files)}, AUTH={len(auth_files)})")

    rows: list[dict[str, Any]] = []
    total = len(all_entries)
    for idx, (video_path, label, split_name) in enumerate(all_entries, 1):
        print(f"[VIDEO] ({idx}/{total}) {split_name}/{video_path.name}")
        try:
            row = extract_video_features(
                video_path=video_path,
                label=label,
                model=model,
                preprocess=preprocess,
                device=device,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
                target_frames=args.target_frames,
            )
            row["split"] = split_name
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Pomijam {video_path.name}: {exc}")

    if not rows:
        raise RuntimeError("Nie udało się wyciągnąć cech z żadnego pliku.")

    feature_fieldnames = list(rows[0].keys())
    features_csv = out_dir / "features.csv"
    with features_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=feature_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OUT] {features_csv}")

    feature_cols = [
        c for c in feature_fieldnames if c not in {"filename", "label", "split"}
    ]
    X = np.array(
        [
            [float(row.get(col, 0.0) if row.get(col, 0.0) is not None else 0.0) for col in feature_cols]
            for row in rows
        ],
        dtype=np.float32,
    )
    y = np.array([int(row["label"]) for row in rows], dtype=np.int32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics: list[dict[str, Any]] = []
    oof_pred = np.zeros_like(y)
    oof_prob = np.zeros(len(y), dtype=np.float32)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=42 + fold,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        clf.fit(X[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X[te_idx])[:, 1]
        preds = (probs >= 0.5).astype(np.int32)

        oof_pred[te_idx] = preds
        oof_prob[te_idx] = probs

        acc = accuracy_score(y[te_idx], preds)
        prec = precision_score(y[te_idx], preds, zero_division=0)
        rec = recall_score(y[te_idx], preds, zero_division=0)
        f1 = f1_score(y[te_idx], preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y[te_idx], preds, labels=[0, 1]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        fold_metrics.append(
            {
                "fold": fold,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "fpr": fpr,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )
        print(
            f"[CV] fold={fold} acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} "
            f"f1={f1:.4f} fpr={fpr:.4f}"
        )

    mean_row = {
        "fold": "mean",
        "accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "precision": float(np.mean([m["precision"] for m in fold_metrics])),
        "recall": float(np.mean([m["recall"] for m in fold_metrics])),
        "f1": float(np.mean([m["f1"] for m in fold_metrics])),
        "fpr": float(np.mean([m["fpr"] for m in fold_metrics])),
        "tp": float(np.mean([m["tp"] for m in fold_metrics])),
        "fp": float(np.mean([m["fp"] for m in fold_metrics])),
        "tn": float(np.mean([m["tn"] for m in fold_metrics])),
        "fn": float(np.mean([m["fn"] for m in fold_metrics])),
    }
    metrics_with_mean = fold_metrics + [mean_row]
    cv_metrics_csv = out_dir / "cv_metrics.csv"
    with cv_metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "accuracy", "precision", "recall", "f1", "fpr", "tp", "fp", "tn", "fn"],
        )
        writer.writeheader()
        writer.writerows(metrics_with_mean)
    print(f"[OUT] {cv_metrics_csv}")

    final_model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    final_model.fit(X, y)

    model_pkl = out_dir / "model.pkl"
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_cols,
            "label_mapping": {"authentic": 0, "ai_generated": 1},
        },
        model_pkl,
    )
    print(f"[OUT] {model_pkl}")

    pred_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        conf = float(oof_prob[i])
        pred_rows.append(
            {
                "filename": row["filename"],
                "true_label": int(y[i]),
                "predicted_label": int(oof_pred[i]),
                "confidence_score": conf,
                "uncertain": conf < 0.6,
            }
        )
    preds_csv = out_dir / "per_video_predictions.csv"
    with preds_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "true_label", "predicted_label", "confidence_score", "uncertain"],
        )
        writer.writeheader()
        writer.writerows(pred_rows)
    print(f"[OUT] {preds_csv}")

    cm = confusion_matrix(y, oof_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center", color="black")
    plt.xticks([0, 1], ["authentic(0)", "ai(1)"])
    plt.yticks([0, 1], ["authentic(0)", "ai(1)"])
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Confusion Matrix (OOF, 5-fold)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_png = out_dir / "confusion_matrix.png"
    plt.savefig(cm_png, dpi=180)
    plt.close()
    print(f"[OUT] {cm_png}")

    importances = final_model.feature_importances_
    top_k = min(25, len(feature_cols))
    top_idx = np.argsort(importances)[-top_k:][::-1]
    top_features = [feature_cols[i] for i in top_idx]
    top_values = importances[top_idx]

    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_values, color="#4C72B0")
    plt.yticks(y_pos, top_features)
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_k} Feature Importances (RandomForest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    fi_png = out_dir / "feature_importances.png"
    plt.savefig(fi_png, dpi=180)
    plt.close()
    print(f"[OUT] {fi_png}")

    print("\n[SUMMARY] 5-fold metrics:")
    for row in metrics_with_mean:
        fold_name = row["fold"]
        print(
            f"  fold={fold_name} acc={float(row['accuracy']):.4f} "
            f"prec={float(row['precision']):.4f} rec={float(row['recall']):.4f} "
            f"f1={float(row['f1']):.4f} fpr={float(row['fpr']):.4f}"
        )

    uncertain_n = sum(1 for r in pred_rows if bool(r["uncertain"]))
    print(f"[SUMMARY] Uncertain (<0.6 confidence): {uncertain_n}/{len(pred_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
