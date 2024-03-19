from pathlib import Path
from typing import List

from PIL import Image


class BaseColorizer:
    def __init__(self, method_name: str):
        self.method_name = method_name

        self.download_model()

    def get_description(self, benchmark_type: str):
        return ""

    def get_paper_link(self):
        pass

    def generate_chromaticity(self):
        return True

    def colorize(self, input_path: Path, reference_paths: List[Path]) -> dict[str, Image]:
        pass

    def download_model(self):
        pass
