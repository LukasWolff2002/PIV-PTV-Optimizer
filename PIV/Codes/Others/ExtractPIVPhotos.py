#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
from typing import List, Optional

# ========= CONFIG =========
Camara = "3"
Fecha = "2 mar"
CARPETA_BASE = f"Tomas/Cam {Camara}/Tomas {Fecha} Cam {Camara}"

DEST_ROOT    = f"TomasProcesadas/Cam{Camara}"
MASKS_ROOT   = f"Masks/Cam{Camara}"   # <<< salida del script de máscaras

NATURAL_SORT = True
OVERWRITE    = True

# Si True, borra DEST_ROOT/sub_name si ya existe y reprocesa.
# Además borra MASKS_ROOT/sub_name (máscaras asociadas) si existe.
DELETE_EXISTING = True

CONFIG = {
    # blocks:
    #   - int  => procesa hasta N bloques
    #   - None => procesa TODA la carpeta (todos los bloques completos posibles)
    "blocks": 40,
    "skip_inter": 0,
    "skip_final": 20,
    "block_size": 22
}

# ========= UTILIDADES =========

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def listar_archivos_ordenados(dir_path: str) -> List[str]:
    files = [
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]
    return sorted(files, key=natural_key if NATURAL_SORT else None)

def copiar_si_corresponde(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if not OVERWRITE and os.path.exists(dst):
        print(f"[SKIP] {dst}")
        return

    shutil.copy2(src, dst)
    print(f"[COPY] {os.path.basename(src)}")

def dest_path(sub_name: str) -> str:
    return os.path.join(DEST_ROOT, sub_name)

def masks_path(sub_name: str) -> str:
    return os.path.join(MASKS_ROOT, sub_name)

def borrar_si_existe(path: str, label: str):
    if os.path.isdir(path):
        print(f"[DEL] Eliminando {label}: {path}")
        shutil.rmtree(path)

# ========= PROCESAMIENTO =========

def procesar_subcarpeta(sub_dir: str, sub_name: str):

    dst_sub = dest_path(sub_name)
    msk_sub = masks_path(sub_name)

    # Si ya existe en destino:
    if os.path.isdir(dst_sub):
        if DELETE_EXISTING:
            print(f"[DEL] '{sub_name}' ya existía en destino. Eliminando para reprocesar...")

            # Borra imágenes procesadas
            borrar_si_existe(dst_sub, "procesadas")

            # Borra máscaras asociadas (si existen)
            borrar_si_existe(msk_sub, "máscaras")

        else:
            print(f"[SKIP] '{sub_name}' ya fue procesada anteriormente.")
            return

    archivos = listar_archivos_ordenados(sub_dir)
    total = len(archivos)

    if total == 0:
        print(f"[WARN] '{sub_name}' no tiene archivos.")
        return

    os.makedirs(dst_sub, exist_ok=True)

    skip_inter = int(CONFIG["skip_inter"])
    skip_final = int(CONFIG["skip_final"])
    block_size = int(CONFIG["block_size"])
    blocks_cfg: Optional[int] = CONFIG.get("blocks", None)

    # Validación de regla
    if (2 + skip_inter + skip_final) != block_size:
        raise ValueError(
            f"Regla inválida: 2 + {skip_inter} + {skip_final} != {block_size}"
        )

    # Cuántos bloques completos caben en la carpeta
    bloques_posibles = total // block_size

    # Definir límite de bloques a procesar
    if blocks_cfg is None:
        blocks_max = bloques_posibles
        modo = "TODO"
    else:
        blocks_max = min(int(blocks_cfg), bloques_posibles)
        modo = f"HASTA {blocks_cfg}"

    print(f"\nProcesando: {sub_name} | Archivos: {total} | Bloques posibles: {bloques_posibles} | Modo: {modo}")

    i = 0
    bloques_hechos = 0

    while (i + block_size) <= total and bloques_hechos < blocks_max:
        idx1 = i
        idx2 = i + 1 + skip_inter

        for idx in (idx1, idx2):
            src = os.path.join(sub_dir, archivos[idx])
            dst = os.path.join(dst_sub, archivos[idx])
            copiar_si_corresponde(src, dst)

        i += block_size
        bloques_hechos += 1

    print(f"[OK] '{sub_name}' procesada | Bloques: {bloques_hechos}/{blocks_max} | Índice final: {i}")

# ========= MAIN =========

def main():

    if not os.path.isdir(CARPETA_BASE):
        raise FileNotFoundError(f"No existe la carpeta base: {CARPETA_BASE}")

    os.makedirs(DEST_ROOT, exist_ok=True)
    os.makedirs(MASKS_ROOT, exist_ok=True)  # no crea subcarpetas, solo asegura root

    subcarpetas = [
        d for d in os.listdir(CARPETA_BASE)
        if os.path.isdir(os.path.join(CARPETA_BASE, d))
    ]
    subcarpetas = sorted(subcarpetas, key=natural_key if NATURAL_SORT else None)

    print(f"Base: {CARPETA_BASE}")
    print(f"Destino procesadas: {DEST_ROOT}")
    print(f"Destino máscaras:   {MASKS_ROOT}")
    print(f"Subcarpetas encontradas: {len(subcarpetas)}")
    print(f"DELETE_EXISTING: {DELETE_EXISTING}")

    # Solo recorremos subcarpetas en CARPETA_BASE, así que solo borramos cosas asociadas a ellas.
    for sub_name in subcarpetas:
        sub_dir = os.path.join(CARPETA_BASE, sub_name)
        procesar_subcarpeta(sub_dir, sub_name)

    print("\n[FIN] Procesamiento completo.")

if __name__ == "__main__":
    main()