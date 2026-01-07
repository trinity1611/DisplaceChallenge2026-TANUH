import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    def load_asr_config(config_dir: str = "./config") -> Dict[str, Any]:
        return ConfigLoader.load_config(f"{config_dir}/asr.yaml")


# class FileIO:
#     @staticmethod
#     def read_text_file(file_path: str) -> str:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             return f.read()

#     @staticmethod
#     def write_text_file(file_path: str, content: str) -> None:
#         Path(file_path).parent.mkdir(parents=True, exist_ok=True)
#         with open(file_path, 'w', encoding='utf-8') as f:
#             f.write(content)
