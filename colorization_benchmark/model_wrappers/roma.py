"""
https://github.com/Parskatt/RoMa
This is not colorization but feature matching model

Requirements: git+https://github.com/Parskatt/RoMa.git
optional: xformers (--no-deps)
"""
from pathlib import Path
from typing import List

from PIL import Image
import numpy as np
import kornia
import torch
import torch.nn.functional as F
from torchvision.transforms import PILToTensor
import torchvision.transforms.functional as TF

from roma import roma_outdoor
from roma.utils.utils import tensor_to_pil
from colorization_benchmark.model_wrappers.base_colorizer import BaseColorizer


class RoMA(BaseColorizer):
    def __init__(self, model_type: str = "lab"):
        super().__init__("roma")

        self.model_type = model_type  # lab, rgb
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.roma_model = roma_outdoor(device=self.device)

    def get_description(self, benchmark_type: str):
        return "This method is order dependent."

    def get_paper_link(self):
        return ""

    def colorize(self, input_path: Path, reference_paths: List[Path]):

        H, W = self.roma_model.get_output_resolution()
        im1_gray = Image.open(input_path).convert("L")
        target_size = im1_gray.size
        im1 = im1_gray.convert("RGB").resize((W, H))

        certainties = []
        labs = []
        rgbs = []
        for reference_path in reference_paths:
            ref_image = Image.open(reference_path).convert("RGB")
            im2 = ref_image.resize((W, H))

            # Match
            warp, certainty = self.roma_model.match(input_path, reference_path, device=self.device)

            # warped_tensor = self.roma_model.visualize_warp(warp, (certainty > 0.1).float(), im1, im2, device=self.device)
            warped_tensor, certainty = self.warp(warp[:,:W, 2:], certainty[:,:W], target_size, ref_image)
            certainties.append(certainty[0])
            # get LAB
            white_im = torch.ones_like(certainty, device=self.device)
            vis_im = torch.lerp(white_im, warped_tensor, (certainty > 0.1).float())
            # certainty * warped_tensor + (1 - certainty) * white_im
            lab = kornia.color.rgb_to_lab(vis_im)
            labs.append(lab)
            rgbs.append(warped_tensor)

        labs = torch.stack(labs)
        certainties = torch.stack(certainties)
        rgbs = torch.stack(rgbs)
        index = torch.argmax(certainties, keepdim=True, dim=0)

        rgb = torch.gather(rgbs, 0, torch.tile(index[None], (1, 3, 1, 1)))[0]
        certainties_gather = torch.gather(certainties, 0, index)

        gray_tensor = torch.tile(PILToTensor()(im1_gray).to(self.device) / 255, [3, 1, 1])

        if self.model_type == "rgb":
            combined = torch.lerp(gray_tensor, rgb, (certainties_gather > 0.1).float())
        elif self.model_type == "lab":
            lab = torch.gather(labs, 0, torch.tile(index[None], (1, 3, 1, 1)))[0]
            combined = kornia.color.lab_to_rgb(torch.cat([gray_tensor[0:1] * 100, lab[1:]]))

        color = tensor_to_pil(combined, unnormalize=False)
        return {"color": color}

    def warp(self, warp, certainty, target_size, image_reference):
        warp = TF.resize(warp.permute(2, 0, 1), list(reversed(target_size))).permute(1, 2, 0)
        certainty = TF.resize(certainty[None], list(reversed(target_size)))
        x_A = (torch.tensor(np.array(image_reference)) / 255).to(self.device).permute(2, 0, 1)

        im_A_transfer_rgb = F.grid_sample(
            x_A[None], warp[None], mode="bilinear", align_corners=False
        )[0]

        return im_A_transfer_rgb, certainty
