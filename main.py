from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

PIPELINE = [
	(
		"survey_data",
		PROJECT_ROOT / "xu_li_du_lieu" / "kiemtradl_ghiralog_extract.py",
	),
	(
		"clean_data",
		PROJECT_ROOT / "xu_li_du_lieu" / "clean_db_ghiralog_transform.py",
	),
	(
		"load_prices",
		PROJECT_ROOT / "connect_clickhouse" / "load_prices_to_click_house.py",
	),
	("train_model", PROJECT_ROOT / "run_risk_pipeline.py"),
	("upload_model", PROJECT_ROOT / "upload_model_csv_to_clickhouse.py"),
	(
		"upload_features_all",
		PROJECT_ROOT / "connect_clickhouse" / "features_all.py",
	),
]


def run_step(name: str, script_path: Path) -> None:
	if not script_path.exists():
		raise FileNotFoundError(f"Missing script: {script_path}")

	print(f"[pipeline] Running {name}: {script_path}")
	subprocess.run(
		[sys.executable, str(script_path)],
		check=True,
		cwd=PROJECT_ROOT,
	)
	print(f"[pipeline] Finished {name}")


def main() -> None:
	print("[pipeline] Starting full stock pipeline")
	for name, script_path in PIPELINE:
		run_step(name, script_path)
	print("[pipeline] All steps completed")


if __name__ == "__main__":
	main()
