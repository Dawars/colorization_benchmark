"""
Unsupervised Deep Exemplar Colorization via Pyramid Dual Non-local Attention (TIP 2023): https://github.com/wd1511/PDNLA-Net

Requirements: termcolor-2.4.0, cuda 11 is needed for cpp compilation

"""
import sys
from pathlib import Path
from typing import List

import torchvision
from PIL import Image
import numpy as np

from colorization_benchmark.model_wrappers.base_colorizer import Colorizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party/pdnla_net"))

from third_party.pdnla_net import Network
from third_party.pdnla_net.colorconvert import *


class PDLNANet(Colorizer):
    description = ""

    def __init__(self, model_path: Path, **opts):
        super(Colorizer).__init__("pdnla_net")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        ckpt_args = ckpt["args"]

        self.myNet = Network.PDNLAnet(ckpt_args.channel).to(self.device)
        self.myNet.load_state_dict(ckpt["PDNLA_ema"])
        self.myNet.eval()
        self.opts = opts

    def colorize(self, input_path: Path, reference_paths: List[Path]):
        gray_image, ref_image = self.image_processing(input_path, reference_paths[0], self.device)
        img3 = self.myNet(gray_image, ref_image)
        color_tensor = gray_replace(gray_image, img3)
        color_tensor = torch.nan_to_num(color_tensor, 0)

        # copied from torchvision.save_image
        grid = torchvision.utils.make_grid(color_tensor, nrow=1, normalize=True, value_range=(-1, 1))
        # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        color = Image.fromarray(ndarr)

        return {"color": color}

    @staticmethod
    def image_processing(imgpath_gray, imgpath_color, device):
        img1 = []
        img2 = []

        img = Image.open(imgpath_gray).convert("RGB").resize((512, 512))
        img_a = (
            torch.from_numpy(np.array(img))
            .to(torch.float32)
            .div(255)
            .add_(-0.5)
            .mul_(2)
            .permute(2, 0, 1)
        )

        img = Image.open(imgpath_color).convert("RGB").resize((512, 512))
        img_b = (
            torch.from_numpy(np.array(img))
            .to(torch.float32)
            .div(255)
            .add_(-0.5)
            .mul_(2)
            .permute(2, 0, 1)
        )
        img1.append(img_a)
        img2.append(img_b)

        img1 = torch.stack(img1, 0).to(device)
        img2 = torch.stack(img2, 0).to(device)
        return img1, img2
