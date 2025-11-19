"""Exportar un modelo YOLO (ultralytics) a ONNX u otros formatos.

Uso recomendado (PowerShell):
	python "d:\Proyecto escarabajos\scripts\exportar modelo.py"
Opciones para personalizar:
	--model / -m    Ruta al archivo .pt (por defecto apunta al peso en fold_3)
	--format / -f   Formato de export (por defecto: onnx)
	--opset         Opset para ONNX (por defecto: 12)
	--simplify      Intentar simplificar (requiere onnxsim)
	--dynamic       Exportar con ejes dinámicos

El script intentará importar `ultralytics` y, si no está instalado, mostrará
la instrucción para instalarlo.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def main() -> None:
	parser = argparse.ArgumentParser(description="Exportar modelo Ultralytics YOLO a ONNX u otros formatos")
	parser.add_argument(
		"--model", "-m",
		default=r"D:\\Proyecto escarabajos\\resultados_cv_5fold\\fold_3\\train\\weights\\best.pt",
		help="Ruta al .pt entrenado o a la carpeta del fold (se buscará train/weights/best.pt)",
	)
	parser.add_argument("--format", "-f", default="onnx", help="Formato de export (por defecto: onnx)")
	parser.add_argument("--opset", type=int, default=12, help="ONNX opset (solo para onnx)")
	parser.add_argument("--simplify", action="store_true", help="Intentar simplificar el ONNX (requiere onnxsim)")
	parser.add_argument("--dynamic", action="store_true", help="Exportar con ejes dinámicos")
	parser.add_argument("--fp16", action="store_true", help="Convertir el ONNX resultante a FP16 para reducir tamaño (recomendado)")
	args = parser.parse_args()

	try:
		from ultralytics import YOLO
	except Exception:
		print("Error: no se pudo importar 'ultralytics'. Instálalo con:")
		print("    pip install ultralytics onnx onnxsim")
		sys.exit(1)

	model_path = Path(args.model)

	# Si el usuario pasó la carpeta del fold (por ejemplo ...\fold_3), buscar ubicaciones comunes
	if model_path.is_dir():
		candidates = [
			model_path / "train" / "weights" / "best.pt",
			model_path / "weights" / "best.pt",
			model_path / "train" / "weights" / "best.pt",
			model_path / "best.pt",
		]
		found = None
		for c in candidates:
			if c.exists():
				found = c
				break
		if found is None:
			print(f"No se encontró 'best.pt' dentro de la carpeta {model_path}. Buscado en: {candidates}")
			sys.exit(1)
		model_path = found

	if not model_path.exists():
		print(f"Error: modelo no encontrado en {model_path}")
		sys.exit(1)

	try:
		print(f"Cargando modelo: {model_path}")
		model = YOLO(str(model_path))
		print("Exportando...")
		model.export(format=args.format, opset=args.opset, simplify=args.simplify, dynamic=args.dynamic)
		print("Export terminado.")

		# Si exportamos a ONNX y el usuario pidió FP16, intentar convertir
		if args.format.lower() == "onnx":
			# localizar el archivo .onnx generado
			onnx_path = None
			candidates = []
			# candidato: mismo nombre con .onnx
			direct = model_path.with_suffix('.onnx')
			candidates.append(direct)
			# candidato: best.onnx en la carpeta del modelo
			candidates.append(model_path.parent / 'best.onnx')
			# buscar por patrón en la carpeta
			candidates.extend(list(model_path.parent.glob(model_path.stem + "*.onnx")))
			# tomar el primero que exista
			for c in candidates:
				if c and Path(c).exists():
					onnx_path = Path(c)
					break
			if onnx_path is None:
				# fallback: buscar cualquier .onnx modificado recientemente en la carpeta
				found = sorted(model_path.parent.glob('*.onnx'), key=lambda p: p.stat().st_mtime, reverse=True)
				if found:
					onnx_path = found[0]
			
			if onnx_path is None:
				print("Aviso: no se encontró el archivo .onnx resultante para convertir a FP16.")
			else:
				print(f"Archivo ONNX detectado: {onnx_path}")
				if args.fp16:
					print("Intentando convertir ONNX a FP16 (reducirá tamaño). Esto no cambia el entrenamiento.")
					try:
						# intentar importar; si falta, instalar y reintentar
						try:
							import onnx
							from onnxconverter_common import convert_float_to_float16
						except Exception:
							print("Paquete 'onnxconverter-common' no encontrado. Intentando instalarlo...")
							import subprocess
							subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxconverter-common"]) 
							# reintentar import después de instalar
							import importlib
							onnx = importlib.import_module('onnx')
							convert_float_to_float16 = importlib.import_module('onnxconverter_common').convert_float_to_float16
						# cargar y convertir
						print("Cargando ONNX para conversión FP16...")
						m = onnx.load(str(onnx_path))
						m16 = convert_float_to_float16(m, keep_io_types=True)
						out_path = onnx_path.with_name(onnx_path.stem + '.fp16.onnx')
						onnx.save(m16, str(out_path))
						print(f"ONNX FP16 guardado en: {out_path} (tamaño reducido)")
					except Exception:
						print("Error durante la conversión a FP16:")
						traceback.print_exc()
						print("Puedes intentar instalar 'onnxconverter-common' manualmente: pip install onnxconverter-common")
					
		print("Export listo.")
	except Exception:
		print("Se produjo un error durante la exportación:")
		traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()