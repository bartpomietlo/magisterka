#!/usr/bin/env python3
# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: ACTIVE

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
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AI-style CLIP embedding classifier for AI vs authentic videos."
    )
    parser.add_argument("--ai-dir", type=Path, default=Path("kod/dataset/ai_baseline"))
    parser.add_argument("--auth-dir", type=Path, default=Path("kod/dataset/adv_fp_trap"))
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--embeddings-pkl", type=Path, default=Path("clip_embeddings.pkl"))
    parser.add_argument("--cv-results-csv", type=Path, default=Path("clip_classifier_cv_results.csv"))
    parser.add_argument("--model-pkl", type=Path, default=Path("clip_classifier.pkl"))
    parser.add_argument("--plot-png", type=Path, default=Path("clip_embedding_plot.png"))
    parser.add_argument("--grok-scores-csv", type=Path, default=Path("clip_grok_scores.csv"))
    return parser.parse_args()


def sample_frame_indices(total_frames: int, n_samples: int) -> np.ndarray:
    if total_frames <= 1:
        return np.zeros(n_samples, dtype=int)
    return np.linspace(0, total_frames - 1, num=n_samples, dtype=int)


def extract_video_embedding(
    video_path: Path,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    n_frames: int,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("video_open_failed")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(total_frames, n_frames)
    images: list[Image.Image] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb))
    cap.release()
    if not images:
        raise RuntimeError("no_frames")

    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    arr = feats.detach().cpu().numpy().astype(np.float32)
    emb = np.concatenate([arr.mean(axis=0), arr.std(axis=0)], axis=0)
    return emb.astype(np.float32)


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr}


def evaluate_model_cv(
    model_name: str,
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows: list[dict[str, Any]] = []
    oof_prob = np.zeros(len(y), dtype=np.float32)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        clf = model_factory()
        clf.fit(X[tr_idx], y[tr_idx])
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X[te_idx])[:, 1]
        else:
            dec = clf.decision_function(X[te_idx])
            prob = 1.0 / (1.0 + np.exp(-dec))
        pred = (prob >= 0.5).astype(np.int32)
        oof_prob[te_idx] = prob
        m = fold_metrics(y[te_idx], pred)
        row = {"model": model_name, "fold": fold, **m}
        rows.append(row)
        print(
            f"[CV] {model_name} fold={fold} acc={m['accuracy']:.4f} "
            f"prec={m['precision']:.4f} rec={m['recall']:.4f} "
            f"f1={m['f1']:.4f} fpr={m['fpr']:.4f}"
        )

    mean_row = {
        "model": model_name,
        "fold": "mean",
        "accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "precision": float(np.mean([r["precision"] for r in rows])),
        "recall": float(np.mean([r["recall"] for r in rows])),
        "f1": float(np.mean([r["f1"] for r in rows])),
        "fpr": float(np.mean([r["fpr"] for r in rows])),
    }
    rows.append(mean_row)
    return rows, oof_prob


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    best_thr = 0.5
    best_f1 = -1.0
    best_metrics = {}
    for thr in np.linspace(0.10, 0.90, 81):
        pred = (probs >= thr).astype(np.int32)
        m = fold_metrics(y_true, pred)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(thr)
            best_metrics = m
    return best_thr, best_metrics


def main() -> int:
    args = parse_args()
    ai_files = sorted(args.ai_dir.glob("*.mp4"))
    auth_files = sorted(args.auth_dir.glob("*.mp4"))
    if not ai_files or not auth_files:
        print(f"[ERR] Brak plikow: ai={len(ai_files)} auth={len(auth_files)}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] device={device}")
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model)
    model.to(device)
    model.eval()

    dataset: list[dict[str, Any]] = []
    total = len(ai_files) + len(auth_files)
    idx = 0

    for p in ai_files:
        idx += 1
        print(f"[AI ] ({idx}/{total}) {p.name}")
        try:
            emb = extract_video_embedding(p, processor, model, device, args.frames_per_video)
            dataset.append(
                {
                    "filename": p.name,
                    "label": 1,
                    "split": "ai_baseline",
                    "is_grok": int("grok-video-" in p.name.lower()),
                    "embedding": emb,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] skip {p.name}: {exc}")

    for p in auth_files:
        idx += 1
        print(f"[AUT] ({idx}/{total}) {p.name}")
        try:
            emb = extract_video_embedding(p, processor, model, device, args.frames_per_video)
            dataset.append(
                {
                    "filename": p.name,
                    "label": 0,
                    "split": "adv_fp_trap",
                    "is_grok": 0,
                    "embedding": emb,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] skip {p.name}: {exc}")

    if len(dataset) < 10:
        print(f"[ERR] Za malo rekordow po ekstrakcji: {len(dataset)}")
        return 2

    # Save embeddings.
    emb_payload = {
        item["filename"]: {
            "embedding": item["embedding"],
            "label": item["label"],
            "split": item["split"],
            "is_grok": item["is_grok"],
        }
        for item in dataset
    }
    joblib.dump(emb_payload, args.embeddings_pkl)
    print(f"[OUT] {args.embeddings_pkl.resolve()}")

    X = np.stack([item["embedding"] for item in dataset], axis=0).astype(np.float32)
    y = np.array([int(item["label"]) for item in dataset], dtype=np.int32)
    filenames = [item["filename"] for item in dataset]
    is_grok = np.array([int(item["is_grok"]) for item in dataset], dtype=np.int32)

    lr_rows, lr_oof = evaluate_model_cv(
        "logistic_regression",
        lambda: LogisticRegression(
            C=1.0,
            penalty="l2",
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        X,
        y,
    )
    lr_mean_f1 = float([r["f1"] for r in lr_rows if r["fold"] == "mean"][0])

    all_rows = list(lr_rows)
    best_name = "logistic_regression"
    best_oof = lr_oof
    best_mean_f1 = lr_mean_f1

    if lr_mean_f1 <= 0.70:
        svm_rows, svm_oof = evaluate_model_cv(
            "svm_rbf",
            lambda: SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced"),
            X,
            y,
        )
        all_rows.extend(svm_rows)
        svm_mean_f1 = float([r["f1"] for r in svm_rows if r["fold"] == "mean"][0])
        if svm_mean_f1 > best_mean_f1:
            best_name = "svm_rbf"
            best_oof = svm_oof
            best_mean_f1 = svm_mean_f1

    # Save CV table.
    with args.cv_results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "fold", "accuracy", "precision", "recall", "f1", "fpr"],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[OUT] {args.cv_results_csv.resolve()}")

    best_thr, best_thr_metrics = find_best_threshold(y, best_oof)
    print(
        f"[THR] model={best_name} threshold={best_thr:.3f} "
        f"(acc={best_thr_metrics['accuracy']:.4f} f1={best_thr_metrics['f1']:.4f} fpr={best_thr_metrics['fpr']:.4f})"
    )

    # Train final chosen model on full data.
    if best_name == "logistic_regression":
        final_model = LogisticRegression(
            C=1.0,
            penalty="l2",
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )
    else:
        final_model = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced")
    final_model.fit(X, y)

    # Top-20 dims (dla LR).
    top_dims: list[int] = []
    coef: np.ndarray | None = None
    if hasattr(final_model, "coef_"):
        coef = np.array(final_model.coef_).reshape(-1).astype(np.float32)
        top_dims = np.argsort(np.abs(coef))[-20:][::-1].tolist()
        print("[LR] Top-20 dims:", top_dims)

    joblib.dump(
        {
            "model": final_model,
            "model_name": best_name,
            "threshold": float(best_thr),
            "feature_dim": int(X.shape[1]),
            "coef": coef,
            "top_dims": top_dims,
            "clip_model": args.clip_model,
            "n_frames": int(args.frames_per_video),
        },
        args.model_pkl,
    )
    print(f"[OUT] {args.model_pkl.resolve()}")

    # PCA plot.
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    ai_mask = y == 1
    au_mask = y == 0
    plt.scatter(coords[au_mask, 0], coords[au_mask, 1], c="#1f77b4", label="authentic", alpha=0.7)
    plt.scatter(coords[ai_mask, 0], coords[ai_mask, 1], c="#d62728", label="ai", alpha=0.7)
    grok_idx = np.where(is_grok == 1)[0]
    if grok_idx.size:
        plt.scatter(
            coords[grok_idx, 0],
            coords[grok_idx, 1],
            marker="*",
            s=160,
            c="gold",
            edgecolors="black",
            linewidths=0.8,
            label="grok",
        )
    plt.title("CLIP embedding PCA (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_png, dpi=180)
    plt.close()
    print(f"[OUT] {args.plot_png.resolve()}")

    # Grok confidence.
    if hasattr(final_model, "predict_proba"):
        probs_full = final_model.predict_proba(X)[:, 1]
    else:
        dec = final_model.decision_function(X)
        probs_full = 1.0 / (1.0 + np.exp(-dec))
    with args.grok_scores_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "clip_ai_prob", "predicted_label", "threshold"])
        for i, name in enumerate(filenames):
            if "grok-video-" in name.lower():
                prob = float(probs_full[i])
                pred = int(prob >= best_thr)
                writer.writerow([name, f"{prob:.6f}", pred, f"{best_thr:.6f}"])
                print(f"[GROK] {name}: prob={prob:.4f}, pred={pred}")
    print(f"[OUT] {args.grok_scores_csv.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
