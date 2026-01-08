from pathlib import Path
import re


class DataProcessor:
    """
    File I/O utilities for reading and writing text data.
    """

    @staticmethod
    def read_text_file(path: str) -> str:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    @staticmethod
    def write_text_file(path: str, text: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            f.write(text)


class TextCleaner:
    """
    Basic text cleaning utilities for ASR / conversational text.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        if text is None:
            return ""
        
        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"[^\x20-\x7E\u0900-\u097F]+", " ", text)

        return text.strip()
