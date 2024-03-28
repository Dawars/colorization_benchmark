"""
https://github.com/richzhang/colorization/tree/master
Requirements: ipython
"""

import sys
from pathlib import Path
from typing import List

from PIL import Image
import wget
import cv2
import numpy as np
import torch
import einops

from colorization_benchmark.model_wrappers.base_colorizer import BaseColorizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party/colorful_colorization"))

from third_party.colorful_colorization.colorizers import *


class ColorfulColorization(BaseColorizer):
    def __init__(self, model_path: Path, **opts):
        self.model_type = opts["model_type"]
        super().__init__("colorful_colorization" if self.model_type == "eccv16" else "real_time_user_guided_colorization")

        assert self.model_type in ["eccv16", "siggraph17"]

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # load colorizers
        if self.model_type == "eccv16":
            self.model = eccv16(pretrained=True).eval()
        elif self.model_type == "siggraph17":
            self.model = siggraph17(pretrained=True).eval()
        self.model.to(self.device)

    def get_description(self, benchmark_type: str):
        return ""

    def get_paper_link(self):
        return "https://github.com/richzhang/colorization/"

    def colorize(self, input_path: Path, reference_paths: List[Path]):

        # default size to process images is 256x256
        # grab L channel in both original ("orig") and resized ("rs") resolutions
        img = load_img(str(input_path))
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        tens_l_rs = tens_l_rs.to(self.device)

        # colorizer outputs 256x256 ab map
        # resize and concatenate to original L channel
        if self.model_type == "eccv16":
            output = postprocess_tens(tens_l_orig, self.model(tens_l_rs).cpu())
        if self.model_type == "siggraph17":
            output = postprocess_tens(tens_l_orig, self.model(tens_l_rs).cpu())

        return {"color": Image.fromarray((255 * output).astype(np.uint8))}
