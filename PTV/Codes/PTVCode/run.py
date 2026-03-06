import subprocess
from pathlib import Path

# =========================
# CONFIGURACIÓN
# =========================

RUN_MODE = "both"   # "piv" | "ptv" | "both"

# Scripts que ejecutan cada pipeline
PIV_SCRIPT = Path("run_piv_script.py")
PTV_SCRIPT = Path("run_ptv_script.py")

PIV_ENV = "piv"
PTV_ENV = "yolov11"


def run_in_env(env_name, script):
    print(f"\n==============================")
    print(f"Running {script} in env '{env_name}'")
    print(f"==============================\n")

    subprocess.run(
        ["conda", "run", "-n", env_name, "python", str(script)],
        check=True
    )


def main():

    if RUN_MODE in ("piv", "both"):
        run_in_env(PIV_ENV, PIV_SCRIPT)

    if RUN_MODE in ("ptv", "both"):
        run_in_env(PTV_ENV, PTV_SCRIPT)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()