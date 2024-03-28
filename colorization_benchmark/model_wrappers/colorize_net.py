"""
https://github.com/rensortino/ColorizeNet
https://huggingface.co/rsortino/ColorizeNet
Requirements: pytorch-lightning<2.0
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party/colorize_net"))

from third_party.colorize_net.utils.data import HWC3, apply_color, resize_image
from third_party.colorize_net.utils.ddim import DDIMSampler
from third_party.colorize_net.utils.model import create_model, load_state_dict


class ColorizeNet(BaseColorizer):
    def __init__(self, model_path: Path, **opts):
        super().__init__("colorize_net")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = create_model(model_path / './models/cldm_v21.yaml').cpu()
        self.model.load_state_dict(load_state_dict(
            model_path / 'weights/colorizenet-sd21.ckpt', location='cuda'))
        self.model = self.model.to(self.device)
        self.ddim_sampler = DDIMSampler(self.model)

    def get_description(self, benchmark_type: str):
        return ""

    def get_paper_link(self):
        return "https://github.com/rensortino/ColorizeNet"

    def colorize(self, input_path: Path, reference_paths: List[Path]):
        input_image = cv2.imread(str(input_path))
        input_image = HWC3(input_image)
        img = resize_image(input_image, resolution=512)
        H, W, C = img.shape

        num_samples = 1
        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        # Prepare the input and parameters of the model

        prompt = "Colorize this image"
        n_prompt = ""
        guess_mode = False
        strength = 1.0
        eta = 0.0
        ddim_steps = 20
        scale = 9.0

        cond = {"c_concat": [control], "c_crossattn": [
            self.model.get_learned_conditioning([prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [
            self.model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                [strength] * 13)

        # colorize

        samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                          shape, cond, verbose=False, eta=eta,
                                                          unconditional_guidance_scale=scale,
                                                          unconditional_conditioning=un_cond)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')
                     * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        colored_results = [apply_color(img, result) for result in results]

        return {"color": Image.fromarray(colored_results[0])}

    def download_model(self):
        ckpt_path = (Path(__file__).parent.parent.parent /
                     "third_party/colorize_net/weights/colorizenet-sd21.ckpt")
        sha256_sum = "ded70440cdcc2b61525dba52d73060cae0754bed1e49f83798355a40050f9a0c"
        if not ckpt_path.exists():
            ckpt_path.parent.mkdir(exist_ok=True, parents=True)
            wget.download("https://huggingface.co/rsortino/ColorizeNet/resolve/main/colorizenet-sd21.ckpt",
                          out=str(ckpt_path))
        import hashlib
        assert hashlib.sha256(ckpt_path.read_bytes()).hexdigest() == sha256_sum
