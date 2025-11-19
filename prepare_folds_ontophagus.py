"""
Script para preparar 5 folds de cross-validation con división 80% train / 20% val
Dataset: Escarabajos generales (clase 0) + Ontophagus_04h (clase 1)
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm

# Configuración de rutas
BASE_DIR = Path(r"D:\Proyecto escarabajos")
DATASET_GENERAL_IMAGES = BASE_DIR / "dataset original" / "images"
DATASET_GENERAL_LABELS = BASE_DIR / "dataset original" / "labels"
DATASET_ONTOPHAGUS_IMAGES = BASE_DIR / "dataset original" / "ontophagus_04h.jpg"
DATASET_ONTOPHAGUS_LABELS = BASE_DIR / "dataset original" / "ontophagus_04h.txt"

OUTPUT_DIR = BASE_DIR / "ontophagus_04h"
N_FOLDS = 5
TRAIN_SPLIT = 0.8  # 80% train, 20% val

def collect_dataset():
    """Recolecta todos los archivos de imágenes y labels con sus etiquetas de clase"""
    print("📁 Recolectando archivos del dataset...")
    
    dataset = []
    
    # Clase 0: Escarabajos generales
    print("  → Procesando escarabajos generales (clase 0)...")
    general_images = list(DATASET_GENERAL_IMAGES.glob("*.JPG")) + list(DATASET_GENERAL_IMAGES.glob("*.jpg"))
    for img_path in general_images:
        label_path = DATASET_GENERAL_LABELS / (img_path.stem + ".txt")
        if label_path.exists():
            dataset.append({
                'image': img_path,
                'label': label_path,
                'class': 0,
                'original_class': 0
            })
    
    # Clase 1: Ontophagus_04h
    print("  → Procesando ontophagus_04h (clase 1)...")
    onto_images = list(DATASET_ONTOPHAGUS_IMAGES.glob("*.JPG")) + list(DATASET_ONTOPHAGUS_IMAGES.glob("*.jpg"))
    for img_path in onto_images:
        label_path = DATASET_ONTOPHAGUS_LABELS / (img_path.stem + ".txt")
        if label_path.exists():
            dataset.append({
                'image': img_path,
                'label': label_path,
                'class': 1,
                'original_class': 1
            })
    
    print(f"✅ Total de archivos recolectados: {len(dataset)}")
    print(f"   - Escarabajos generales: {sum(1 for d in dataset if d['class'] == 0)}")
    print(f"   - Ontophagus_04h: {sum(1 for d in dataset if d['class'] == 1)}")
    
    return dataset

def update_label_class(label_path, new_class):
    """Actualiza la clase en el archivo de etiqueta"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    updated_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # Formato YOLO: class x_center y_center width height
            parts[0] = str(new_class)
            updated_lines.append(' '.join(parts))
    
    return '\n'.join(updated_lines)

def create_fold_structure(fold_num, train_indices, val_indices, dataset):
    """Crea la estructura de carpetas y copia archivos para un fold"""
    fold_dir = OUTPUT_DIR / f"fold_{fold_num}"
    
    # Crear estructura de directorios
    train_img_dir = fold_dir / "train" / "images"
    train_lbl_dir = fold_dir / "train" / "labels"
    val_img_dir = fold_dir / "val" / "images"
    val_lbl_dir = fold_dir / "val" / "labels"
    
    for directory in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Creando Fold {fold_num}...")
    print(f"   Train: {len(train_indices)} imágenes")
    print(f"   Val: {len(val_indices)} imágenes")
    
    # Copiar archivos de entrenamiento
    print("   → Copiando archivos de train...")
    for idx in tqdm(train_indices, desc=f"  Fold {fold_num} Train"):
        item = dataset[idx]
        
        # Copiar imagen
        dst_img = train_img_dir / item['image'].name
        shutil.copy2(item['image'], dst_img)
        
        # Copiar y actualizar label
        label_content = update_label_class(item['label'], item['class'])
        dst_lbl = train_lbl_dir / item['label'].name
        with open(dst_lbl, 'w') as f:
            f.write(label_content)
    
    # Copiar archivos de validación
    print("   → Copiando archivos de val...")
    for idx in tqdm(val_indices, desc=f"  Fold {fold_num} Val"):
        item = dataset[idx]
        
        # Copiar imagen
        dst_img = val_img_dir / item['image'].name
        shutil.copy2(item['image'], dst_img)
        
        # Copiar y actualizar label
        label_content = update_label_class(item['label'], item['class'])
        dst_lbl = val_lbl_dir / item['label'].name
        with open(dst_lbl, 'w') as f:
            f.write(label_content)
    
    print(f"✅ Fold {fold_num} completado")
    
    return fold_dir

def create_folds():
    """Crea los 5 folds con división 80/20"""
    print("\n" + "="*60)
    print("🔄 PREPARANDO 5-FOLD CROSS VALIDATION")
    print("="*60)
    
    # Recolectar dataset
    dataset = collect_dataset()
    
    # Limpiar directorio de salida si existe
    if OUTPUT_DIR.exists():
        print(f"\n🗑️  Limpiando directorio existente: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Crear índices para los folds
    indices = np.arange(len(dataset))
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    print(f"\n📊 Configuración:")
    print(f"   - Total de muestras: {len(dataset)}")
    print(f"   - Número de folds: {N_FOLDS}")
    print(f"   - División: {int(TRAIN_SPLIT*100)}% train / {int((1-TRAIN_SPLIT)*100)}% val")
    
    # Crear cada fold
    for fold_num, (train_idx, val_idx) in enumerate(kfold.split(indices), 1):
        create_fold_structure(fold_num, train_idx, val_idx, dataset)
    
    print("\n" + "="*60)
    print("✅ TODOS LOS FOLDS CREADOS EXITOSAMENTE")
    print("="*60)
    print(f"\n📁 Ubicación: {OUTPUT_DIR}")
    print("\n📝 Estructura creada:")
    print("   ontophagus_04h/")
    for i in range(1, N_FOLDS + 1):
        print(f"   ├── fold_{i}/")
        print(f"   │   ├── train/")
        print(f"   │   │   ├── images/")
        print(f"   │   │   └── labels/")
        print(f"   │   └── val/")
        print(f"   │       ├── images/")
        print(f"   │       └── labels/")
    
    # Estadísticas finales
    print("\n📊 Estadísticas por fold:")
    for i in range(1, N_FOLDS + 1):
        fold_dir = OUTPUT_DIR / f"fold_{i}"
        train_imgs = len(list((fold_dir / "train" / "images").glob("*.JPG")))
        val_imgs = len(list((fold_dir / "val" / "images").glob("*.JPG")))
        total = train_imgs + val_imgs
        print(f"   Fold {i}: Train={train_imgs} ({train_imgs/total*100:.1f}%), Val={val_imgs} ({val_imgs/total*100:.1f}%)")

if __name__ == "__main__":
    try:
        create_folds()
        print("\n🎉 Proceso completado con éxito!")
    except Exception as e:
        print(f"\n❌ Error durante el proceso: {str(e)}")
        import traceback
        traceback.print_exc()
