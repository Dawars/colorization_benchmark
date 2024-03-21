"""
https://github.com/piddnad/DDColor

Requirements: modelscope
timm==0.9.2
dlib==19.24.2
lmdb==1.4.1
numpy==1.24.3
opencv_python==4.7.0.72
Pillow==10.1.0
PyYAML==6.0.1
Requests==2.31.0
scipy==1.9.1
torch==2.2.0
torchvision==0.17.0
tqdm==4.65.0
wandb==0.15.5
scikit-image==0.22.0
"""
import sys
from pathlib import Path
from typing import List
from collections import Counter

import cv2
import torch
from PIL import Image
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from colorization_benchmark.model_wrappers.base_colorizer import BaseColorizer


class DDColor(BaseColorizer):
    def __init__(self, model_path: Path, **opts):
        super().__init__("ddcolor")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = pipeline(Tasks.image_colorization, model=str(model_path), device="gpu")

    def get_description(self, benchmark_type: str):
        return ""

    def get_paper_link(self):
        return "https://arxiv.org/abs/2212.11613"

    def colorize(self, input_path: Path, reference_paths: List[Path]):
        result = self.model(str(input_path))
        color = Image.fromarray(result[OutputKeys.OUTPUT_IMG][..., ::-1])
        return {"color": color}
