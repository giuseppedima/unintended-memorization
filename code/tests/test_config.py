import tempfile
import unittest
from pathlib import Path

from memorization.config import load_config
from memorization.exceptions import ConfigurationError


class TestConfig(unittest.TestCase):
    def test_load_default_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct\n"
                "BACKGROUND_DATASET=Salesforce/wikitext\n"
                "BATCH_SIZE=4\n"
                "NUM_EPOCHS=2\n",
                encoding="utf-8",
            )
            cfg = load_config(env_file)
            self.assertEqual(cfg.model_name, "Qwen/Qwen2.5-0.5B-Instruct")
            self.assertEqual(cfg.background_dataset, "Salesforce/wikitext")
            self.assertEqual(cfg.batch_size, 4)
            self.assertEqual(cfg.num_epochs, 2)

    def test_invalid_batch_size_raises_configuration_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("BATCH_SIZE=16\n", encoding="utf-8")  # 256/16 = 16 < 20 batches
            with self.assertRaises(ConfigurationError):
                load_config(env_file)
