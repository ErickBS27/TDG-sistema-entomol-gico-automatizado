"""
Evaluación por detección (IoU matching) y curvas ROC/PR para Cross-Validation de YOLOv11.

- Lee config_cv.yaml para ubicar dataset y resultados por fold
- Para cada fold: carga el best.pt, infiere en el split de validación y hace matching por IoU
- Genera arrays por-detección (score, TP/FP) y calcula:
    - ROC (per-detection) con sklearn
    - PR correcta para detección (precision/recall barriendo umbrales usando total_gt)
    - AP (micro) y mAP opcional por clase
- Exporta PNGs por fold y un promedio entre folds disponibles

Nota sobre ROC en detección: la ROC aquí es "per-detection" (clasifica cada predicción como TP/FP en función del score),
no considera TN del fondo. Es útil para ver la discriminación del score; la métrica estándar en detección sigue siendo PR/AP/mAP.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# ===================== Utilidades I/O =====================

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT.parent / "yaml" / "ontophagus_04h.yaml"


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_cv_layout(cfg: Dict) -> Dict:
    """Usa estrictamente las rutas definidas en config: output_cv_dir y results_dir."""
    # Por defecto usar la raíz del proyecto (carpeta padre de `scripts`)
    base_dir = Path(cfg.get("base_dir", ROOT.parent)).resolve()
    dataset_cv_dir = base_dir / cfg.get("output_cv_dir", "ontophagus_04h")
    results_cv_dir = base_dir / cfg.get("results_dir", "resultados_ontophagus_cv")

    # Tamaño de batch para inferencia (si no hay sección específica de validación, usar training.batch)
    infer_batch = int(cfg.get("validation", {}).get("batch", cfg.get("training", {}).get("batch", 16)))

    return {
        "base_dir": base_dir,
        "dataset_cv_dir": dataset_cv_dir,
        "results_cv_dir": results_cv_dir,
        "n_folds": int(cfg.get("n_folds", 5)),
        "imgsz": int(cfg.get("training", {}).get("imgsz", 640)),
        "device": cfg.get("training", {}).get("device", 0),
        "val_conf": float(cfg.get("validation", {}).get("conf", 0.001)),
        "val_iou": float(cfg.get("validation", {}).get("iou", 0.6)),
        "infer_batch": int(max(1, infer_batch)),
    }


# ===================== Utilidades de cajas =====================

def xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
    cx = xc * img_w
    cy = yc * img_h
    bw = w * img_w
    bh = h * img_h
    return [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    xa = max(boxA[0], boxB[0])
    ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2])
    yb = min(boxA[3], boxB[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def load_yolo_label_file(label_path: Path, img_w: int, img_h: int) -> List[Dict]:
    """Lee un .txt YOLO; soporta bbox estándar y polígonos (8 coords normalizadas)."""
    if not label_path.exists():
        return []
    objects = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(float(parts[0]))
            if len(parts) == 9:
                xs = [float(parts[i]) * img_w for i in range(1, 9, 2)]
                ys = [float(parts[i]) * img_h for i in range(2, 9, 2)]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                bbox = [x1, y1, x2, y2]
            else:
                xc, yc, w, h = map(float, parts[1:5])
                bbox = xywhn_to_xyxy(xc, yc, w, h, img_w, img_h)
            objects.append({"class": cls_id, "bbox": bbox})
    return objects


# ===================== Matching y evaluación =====================

def greedy_match_per_image(
    preds: List[Dict], gts: List[Dict], iou_thr: float
) -> Tuple[List[int], List[float]]:
    """Empareja greedy por confianza: devuelve y_true binario y scores alineados a predicciones.
    - preds: [{class:int, bbox:[x1,y1,x2,y2], conf:float}]
    - gts:   [{class:int, bbox:[x1,y1,x2,y2]}]
    """
    if not preds:
        return [], []
    # Ordenar por score desc
    order = np.argsort([-p["conf"] for p in preds])
    preds_sorted = [preds[i] for i in order]

    matched_gt = set()  # índices de GT ya usados
    y_true = []
    scores = []

    for p in preds_sorted:
        best_iou, best_idx = 0.0, -1
        for j, gt in enumerate(gts):
            if j in matched_gt:
                continue
            if int(gt["class"]) != int(p["class"]):
                continue
            iou = calculate_iou(p["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou, best_idx = iou, j

        if best_iou >= iou_thr and best_idx >= 0:
            matched_gt.add(best_idx)
            y_true.append(1)
        else:
            y_true.append(0)
        scores.append(float(p["conf"]))

    return y_true, scores


def compute_pr_from_detections(y_true_all: List[int], scores_all: List[float], total_gt: int,
                               num_points: int = 200) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calcula PR barriendo umbrales correctamente para detección (usa total_gt).
    Devuelve (recall, precision, AP_micro).
    """
    if len(scores_all) == 0:
        return np.array([0.0]), np.array([1.0]), 0.0

    # Ordenar por score desc
    order = np.argsort(-np.array(scores_all))
    y_true_sorted = np.array(y_true_all)[order]
    scores_sorted = np.array(scores_all)[order]

    # Umbrales a muestrear
    thr_values = np.linspace(scores_sorted.min(), scores_sorted.max(), num_points)
    precisions, recalls = [], []

    for thr in thr_values:
        mask = scores_sorted >= thr
        tp = int(y_true_sorted[mask].sum())
        fp = int(mask.sum()) - tp
        fn = max(0, int(total_gt) - tp)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    # Interpolación tipo 11-point es obsoleta; usar integración trapezoidal sobre P vs R ordenado por R
    recalls_np = np.array(recalls)
    precisions_np = np.array(precisions)

    # Asegurar monotonía no creciente de precision (suavizado)
    for i in range(len(precisions_np) - 2, -1, -1):
        precisions_np[i] = max(precisions_np[i], precisions_np[i + 1])

    # Ordenar por recall ascendente
    order_r = np.argsort(recalls_np)
    # Reemplazar trapz deprecado por trapezoid
    ap = np.trapezoid(precisions_np[order_r], recalls_np[order_r])

    return recalls_np, precisions_np, float(ap)


def evaluate_fold(
    fold_idx: int,
    weights_path: Path,
    val_images_dir: Path,
    val_labels_dir: Path,
    imgsz: int,
    device: str | int,
    conf_for_pred: float,
    iou_match: float,
    infer_batch: int,
) -> Dict:
    """Ejecuta inferencia en val y calcula ROC/PR por detección para el fold.
    Devuelve dict con arrays/metricas resumen.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"No existe best.pt para fold {fold_idx}: {weights_path}")

    # Carga modelo en GPU si disponible/solicitado
    model = YOLO(str(weights_path))

    # Preparar device (forzar GPU si no es 'cpu')
    dev_str: str
    if isinstance(device, (int,)):
        dev_str = f"cuda:{device}"
    elif isinstance(device, str):
        dev_str = "cpu" if device.lower() == "cpu" else (f"cuda:{device}" if device.isdigit() else device)
    elif isinstance(device, (list, tuple)) and len(device) > 0:
        # Si viene una lista de GPUs [0,1,...], usar la primera
        dev_str = f"cuda:{device[0]}"
    else:
        dev_str = "cuda:0"

    if dev_str != "cpu":
        model.to(dev_str)

    # Recolectores globales
    y_true_all: List[int] = []
    scores_all: List[float] = []
    total_gt = 0

    # Lista de imágenes (extensiones comunes)
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.JPG", "*.PNG"):
        img_paths.extend(sorted(val_images_dir.glob(ext)))

    if not img_paths:
        raise FileNotFoundError(f"No hay imágenes en {val_images_dir}")

    # Inferencia en lotes para no saturar GPU
    for i in range(0, len(img_paths), infer_batch):
        batch_paths = img_paths[i:i + infer_batch]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=imgsz,
            conf=conf_for_pred,
            iou=0.7,  # NMS IoU (independiente del IoU de matching)
            device=dev_str,
            verbose=False,
            stream=False,
            save=False,
            max_det=300,
        )

        # Recorrer resultados por imagen dentro del batch
        for res in results:
            img_path = Path(res.path)
            # Tamaño imagen
            if hasattr(res, "orig_shape") and res.orig_shape is not None:
                img_h, img_w = int(res.orig_shape[0]), int(res.orig_shape[1])
            else:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size

            # GT
            label_path = val_labels_dir / f"{img_path.stem}.txt"
            gts = load_yolo_label_file(label_path, img_w, img_h)
            total_gt += len(gts)

            # Predicciones
            preds = []
            if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.detach().cpu().numpy()
                cls = res.boxes.cls.detach().cpu().numpy().astype(int)
                confs = res.boxes.conf.detach().cpu().numpy()
                for b, c, s in zip(xyxy, cls, confs):
                    preds.append({"class": int(c), "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])], "conf": float(s)})

            y_img, s_img = greedy_match_per_image(preds, gts, iou_thr=iou_match)
            if y_img:
                y_true_all.extend(y_img)
                scores_all.extend(s_img)

        # Liberar memoria GPU entre lotes
        try:
            del results
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Algunas instalaciones soportan ipc_collect; si no existe, se ignora
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    # ROC per-detection con sklearn
    roc = {"fpr": [], "tpr": [], "auc": None}
    if len(scores_all) > 0 and any(y_true_all):
        fpr, tpr, _ = roc_curve(np.array(y_true_all), np.array(scores_all))
        roc["fpr"], roc["tpr"] = fpr.tolist(), tpr.tolist()
        roc["auc"] = float(auc(fpr, tpr))
    else:
        roc["fpr"], roc["tpr"], roc["auc"] = [0.0, 1.0], [0.0, 1.0], 0.0

    # PR correcta con total_gt
    rec, prec, ap = compute_pr_from_detections(y_true_all, scores_all, total_gt)

    return {
        "fold": fold_idx,
        "total_gt": int(total_gt),
        "total_preds": int(len(scores_all)),
        "tp_predictions": int(sum(y_true_all)),
        "roc": roc,
        "pr": {"recall": rec.tolist(), "precision": prec.tolist(), "ap": float(ap)},
        # Para combinar globalmente
        "_raw_y_true": y_true_all,
        "_raw_scores": scores_all,
    }


# ===================== Plotting =====================

def plot_roc(fold_metrics: List[Dict], out_png: Path):
    plt.figure(figsize=(6, 5))
    aucs = []
    for m in fold_metrics:
        fpr = np.array(m["roc"]["fpr"]) if m["roc"]["fpr"] else np.array([0.0, 1.0])
        tpr = np.array(m["roc"]["tpr"]) if m["roc"]["tpr"] else np.array([0.0, 1.0])
        plt.plot(fpr, tpr, alpha=0.5, label=f"Fold {m['fold']} (AUC={m['roc']['auc']:.3f})")
        if m["roc"]["auc"] is not None:
            aucs.append(m["roc"]["auc"])

    # Promedio en rejilla común
    grid = np.linspace(0, 1, 200)
    tprs = []
    for m in fold_metrics:
        fpr = np.array(m["roc"]["fpr"]) if m["roc"]["fpr"] else np.array([0.0, 1.0])
        tpr = np.array(m["roc"]["tpr"]) if m["roc"]["tpr"] else np.array([0.0, 1.0])
        tprs.append(np.interp(grid, fpr, tpr))
    if tprs:
        mean_tpr = np.mean(np.vstack(tprs), axis=0)
        mean_auc = auc(grid, mean_tpr)
        plt.plot(grid, mean_tpr, color="black", lw=2, label=f"Media (AUC={mean_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("FPR (per-detection)")
    plt.ylabel("TPR")
    plt.title("ROC por detección - Cross-Validation")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_pr(fold_metrics: List[Dict], out_png: Path):
    plt.figure(figsize=(6, 5))
    aps = []
    for m in fold_metrics:
        r = np.array(m["pr"]["recall"]) if m["pr"]["recall"] else np.array([0.0])
        p = np.array(m["pr"]["precision"]) if m["pr"]["precision"] else np.array([1.0])
        plt.plot(r, p, alpha=0.5, label=f"Fold {m['fold']} (AP={m['pr']['ap']:.3f})")
        aps.append(m["pr"]["ap"])

    # Promedio en rejilla común
    grid_r = np.linspace(0, 1, 200)
    ps = []
    for m in fold_metrics:
        r = np.array(m["pr"]["recall"]) if m["pr"]["recall"] else np.array([0.0])
        p = np.array(m["pr"]["precision"]) if m["pr"]["precision"] else np.array([1.0])
        ps.append(np.interp(grid_r, r, p))
    if ps:
        mean_p = np.mean(np.vstack(ps), axis=0)
        mean_ap = float(np.trapezoid(mean_p, grid_r))
        plt.plot(grid_r, mean_p, color="black", lw=2, label=f"Media (AP≈{mean_ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR por detección - Cross-Validation")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_global_roc(y_true_all: List[int], scores_all: List[float], out_png: Path):
    """Dibuja una sola curva ROC combinando todas las inferencias (todos los folds)."""
    plt.figure(figsize=(6, 5))
    if len(scores_all) > 0 and any(y_true_all):
        fpr, tpr, _ = roc_curve(np.array(y_true_all), np.array(scores_all))
        roc_auc = float(auc(fpr, tpr))
        plt.plot(fpr, tpr, color="C0", lw=2, label=f"ROC global (AUC={roc_auc:.3f})")
    else:
        roc_auc = 0.0
        plt.plot([0, 1], [0, 1], color="C0", lw=2, label="ROC global (AUC=0.000)")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("FPR (per-detection)")
    plt.ylabel("TPR")
    plt.title("ROC global por detección")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_global_pr(y_true_all: List[int], scores_all: List[float], total_gt: int, out_png: Path):
    """Dibuja una sola curva PR combinada (opcional, útil para referencia)."""
    r, p, ap = compute_pr_from_detections(y_true_all, scores_all, total_gt)
    plt.figure(figsize=(6, 5))
    plt.plot(r, p, color="C1", lw=2, label=f"PR global (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR global por detección")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


# ===================== Main =====================

def main():
    cfg = load_config(CONFIG_PATH)
    env = detect_cv_layout(cfg)

    dataset_cv_dir: Path = env["dataset_cv_dir"]
    results_cv_dir: Path = env["results_cv_dir"]
    n_folds: int = env["n_folds"]
    imgsz: int = env["imgsz"]
    device = env["device"]
    conf_pred: float = env["val_conf"]
    iou_match: float = env["val_iou"]
    infer_batch: int = env["infer_batch"]

    print("Configuración detectada:")
    print(f"- dataset_cv_dir: {dataset_cv_dir}")
    print(f"- results_cv_dir: {results_cv_dir}")
    print(f"- n_folds: {n_folds}")
    print(f"- imgsz: {imgsz}")
    print(f"- device: {device}  (usar GPU si disponible)")
    print(f"- conf_pred: {conf_pred}")
    print(f"- iou_match: {iou_match}")
    print(f"- infer_batch: {infer_batch}")

    folds_metrics: List[Dict] = []
    # Acumuladores globales para ROC/PR
    global_y_true: List[int] = []
    global_scores: List[float] = []
    global_total_gt: int = 0

    # Descubrir folds disponibles según pesos
    candidate_folds = sorted([d for d in results_cv_dir.glob("fold_*") if d.is_dir()])
    if not candidate_folds:
        # fallback: usar 1..n_folds aunque falten
        candidate_folds = [results_cv_dir / f"fold_{i}" for i in range(1, n_folds + 1)]

    for fold_dir in candidate_folds:
        try:
            fold_name = fold_dir.name
            fold_idx = int(fold_name.split("_")[-1]) if fold_name.split("_")[-1].isdigit() else None
            if fold_idx is None:
                continue

            # Buscar pesos en rutas comunes: primero la estructura con 'train/weights',
            # luego una ubicación más simple 'weights' (algunas ejecuciones usan esa estructura),
            # y finalmente como fallback buscar cualquier .pt en subcarpetas.
            candidate1 = fold_dir / "train" / "weights" / "best.pt"
            candidate2 = fold_dir / "weights" / "best.pt"
            weights_path = None
            if candidate1.exists():
                weights_path = candidate1
            elif candidate2.exists():
                weights_path = candidate2
            else:
                # Buscar cualquier .pt en subdirectorios 'weights' o en el fold
                found = list(fold_dir.rglob("*.pt"))
                if found:
                    # Preferir archivos dentro de una carpeta 'weights' si hay
                    weights_in_weights = [p for p in found if "weights" in [part.lower() for part in p.parts]]
                    weights_path = weights_in_weights[0] if weights_in_weights else found[0]

            if weights_path is None or not weights_path.exists():
                print(f"[AVISO] Sin pesos en fold {fold_idx} (buscadas: {candidate1} , {candidate2}) -> se omite fold {fold_idx}")
                continue
            else:
                print(f"[INFO] Usando pesos para fold {fold_idx}: {weights_path}")

            # data/labels del fold correspondiente
            data_dir = dataset_cv_dir / f"fold_{fold_idx}"
            # Soportar dos estructuras comunes:
            #  1) images/val  and labels/val  (ej. dataset_cv_dir/fold_X/images/val)
            #  2) val/images  and val/labels  (ej. dataset_cv_dir/fold_X/val/images)
            cand1_images = data_dir / "images" / "val"
            cand1_labels = data_dir / "labels" / "val"
            cand2_images = data_dir / "val" / "images"
            cand2_labels = data_dir / "val" / "labels"

            if cand1_images.exists():
                val_images_dir = cand1_images
                val_labels_dir = cand1_labels
            elif cand2_images.exists():
                val_images_dir = cand2_images
                val_labels_dir = cand2_labels
            else:
                # Fallback: buscar imágenes y labels en subcarpetas 'val' o 'images'
                # Esto permite máxima tolerancia a estructuras ligeramente distintas.
                possible_imgs = list(data_dir.rglob('*.jpg')) + list(data_dir.rglob('*.JPG'))
                if possible_imgs:
                    # Usar la carpeta que contiene la primera imagen encontrada
                    val_images_dir = possible_imgs[0].parent
                    # intentar localizar labels relativas
                    val_labels_dir = (val_images_dir.parent / 'labels') if (val_images_dir.parent / 'labels').exists() else (val_images_dir / '..' / 'labels')
                else:
                    val_images_dir = cand1_images  # dejar como está para que el mensaje de error muestre la ruta esperada
                    val_labels_dir = cand1_labels

            print(f"\nEvaluando fold {fold_idx} ...")
            m = evaluate_fold(
                fold_idx=fold_idx,
                weights_path=weights_path,
                val_images_dir=val_images_dir,
                val_labels_dir=val_labels_dir,
                imgsz=imgsz,
                device=device,
                conf_for_pred=conf_pred,
                iou_match=iou_match,
                infer_batch=infer_batch,
            )
            # Acumular global
            global_y_true.extend(m.get("_raw_y_true", []))
            global_scores.extend(m.get("_raw_scores", []))
            global_total_gt += int(m.get("total_gt", 0))

            # Guardar copia sin arrays crudos
            m_to_save = {k: v for k, v in m.items() if k not in {"_raw_y_true", "_raw_scores"}}
            folds_metrics.append(m_to_save)

            # Guardar métricas por fold en el directorio de resultados del CV
            out_dir = results_cv_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"fold_{fold_idx}_roc_pr.json", "w", encoding="utf-8") as f:
                json.dump(m_to_save, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[ERROR] Fold {fold_dir.name}: {e}")
            continue

    if not folds_metrics:
        print("No se evaluó ningún fold. Verifica rutas de pesos y dataset.")
        return

    # Curvas globales (una sola curva con todas las inferencias)
    out_dir = results_cv_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_global_roc(global_y_true, global_scores, out_dir / "cv_roc_global.png")
    plot_global_pr(global_y_true, global_scores, global_total_gt, out_dir / "cv_pr_global.png")

    # Resumen
    # Calcular métricas globales
    if len(global_scores) > 0 and any(global_y_true):
        fpr_g, tpr_g, _ = roc_curve(np.array(global_y_true), np.array(global_scores))
        auc_g = float(auc(fpr_g, tpr_g))
    else:
        auc_g = 0.0
    _, _, ap_g = compute_pr_from_detections(global_y_true, global_scores, global_total_gt)

    summary = {
        "folds": [m["fold"] for m in folds_metrics],
        "mean_ap": float(np.mean([m["pr"]["ap"] for m in folds_metrics])),
        "mean_auc": float(np.mean([m["roc"]["auc"] or 0.0 for m in folds_metrics])),
        "global": {
            "total_gt": int(global_total_gt),
            "total_preds": int(len(global_scores)),
            "auc": float(auc_g),
            "ap": float(ap_g),
        },
        "details": folds_metrics,
    }
    with open(out_dir / "cv_summary_roc_pr.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nListo. Salidas en:")
    print(f"- {out_dir / 'cv_roc_global.png'}")
    print(f"- {out_dir / 'cv_pr_global.png'}")
    print(f"- {out_dir / 'cv_summary_roc_pr.json'}")


if __name__ == "__main__":
    main()
