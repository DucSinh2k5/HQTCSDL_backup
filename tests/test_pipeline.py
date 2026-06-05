import sys
import unittest
from unittest.mock import patch

import main as pipeline_main


class PipelineTests(unittest.TestCase):
    def test_model1_marts_upload_step_runs_immediately_after_train_model1(self):
        step_names = [pipeline_main.unpack_step(step)[0] for step in pipeline_main.PIPELINE]

        self.assertIn("train_model1", step_names)
        self.assertIn("generate_model1_marts", step_names)
        self.assertEqual(
            step_names.index("train_model1") + 1,
            step_names.index("generate_model1_marts"),
        )

        mart_step = next(
            step
            for step in pipeline_main.PIPELINE
            if pipeline_main.unpack_step(step)[0] == "generate_model1_marts"
        )
        name, script_path, args = pipeline_main.unpack_step(mart_step)

        self.assertEqual("generate_model1_marts", name)
        self.assertEqual(
            pipeline_main.PROJECT_ROOT / "models" / "model1" / "generate_marts.py",
            script_path,
        )
        self.assertEqual(["--upload-clickhouse"], args)

    def test_run_step_passes_optional_script_arguments(self):
        script_path = pipeline_main.PROJECT_ROOT / "main.py"

        with patch.object(pipeline_main.subprocess, "run") as run:
            pipeline_main.run_step(
                "sample_step",
                script_path,
                args=["--example-flag"],
            )

        run.assert_called_once_with(
            [sys.executable, str(script_path), "--example-flag"],
            check=True,
            cwd=pipeline_main.PROJECT_ROOT,
        )


if __name__ == "__main__":
    unittest.main()
