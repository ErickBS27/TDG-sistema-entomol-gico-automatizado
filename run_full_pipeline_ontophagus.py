"""run_full_pipeline_ontophagus.py

Script principal para ejecutar el pipeline de Cross-Validation.

Pasos realizados:
1. Preparar los 5 folds
2. Entrenar con YOLOv11x mediante cross-validation
3. Generar reportes y mostrar resumen
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


BASE_DIR = Path(r"D:\Proyecto escarabajos")
SCRIPTS_DIR = BASE_DIR / "scripts"


def configure_logging() -> None:
    """Configura el logging básico para el script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_script(script_name: str, description: str, timeout: Optional[int] = None) -> bool:
    """Ejecuta un script Python localizado en `SCRIPTS_DIR`.

    Retorna True si el script finaliza con código 0, False en caso contrario.
    """
    logging.info("%s", "=" * 72)
    logging.info("Iniciando: %s", description)

    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        logging.error("No se encontró el script: %s", script_path)
        return False

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True,
            timeout=timeout,
        )
        logging.info("%s - COMPLETADO", description)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("%s - FALLÓ (CalledProcessError): %s", description, e)
        return False
    except subprocess.TimeoutExpired as e:
        logging.error("%s - FALLÓ (Timeout): %s", description, e)
        return False
    except Exception as e:
        logging.exception("%s - FALLÓ (Excepción inesperada): %s", description, e)
        return False


def main() -> None:
    configure_logging()
    start_time = datetime.now()

    logging.info("Inicio del pipeline de Cross-Validation")
    logging.info("Directorio base: %s", BASE_DIR)
    logging.info("Hora de inicio: %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Paso 1: Preparar folds
    success = run_script(
        "prepare_folds_ontophagus.py",
        "PASO 1: Preparación de 5 folds con división 80/20",
    )

    if not success:
        logging.error("No se pudo completar la preparación de folds. Abortando.")
        return

    # Paso 2: Entrenamiento con cross-validation
    success = run_script(
        "train_cv_ontophagus.py",
        "PASO 2: Entrenamiento con Cross-Validation",
    )

    if not success:
        logging.warning(
            "El entrenamiento tuvo problemas. Los folds fueron creados correctamente, revisar logs de entrenamiento."
        )

    # Resumen final
    end_time = datetime.now()
    total_minutes = (end_time - start_time).total_seconds() / 60.0

    logging.info("%s", "=" * 72)
    logging.info("RESUMEN FINAL")
    logging.info("Tiempo total: %.2f minutos", total_minutes)
    logging.info("Resultados en: %s", BASE_DIR / 'resultados_ontophagus_cv')
    logging.info("Folds en: %s", BASE_DIR / 'ontophagus_04h')
    logging.info("Ejecución finalizada")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Pipeline interrumpido por el usuario (KeyboardInterrupt)")
        raise
    except Exception:
        logging.exception("Error no controlado durante la ejecución del pipeline")
        raise
