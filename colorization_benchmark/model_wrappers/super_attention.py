"""
https://github.com/hernan0930/Super-attention-for-exemplar-based-image-colorization
MIT Licence
Requirements: torch-scatter
"""

import sys
from pathlib import Path
from typing import List

import gdown
import torchvision
from PIL import Image
import numpy as np
from skimage import io, color
from skimage.color import rgb2gray
from torch.utils.data import default_collate
from torchvision.transforms import transforms as T

from colorization_benchmark.model_wrappers.base_colorizer import BaseColorizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party/super_attention"))

from third_party.super_attention.models.super_skip_unet import *
from third_party.super_attention.models.model_perceptual import *
from third_party.super_attention.utils import imagenet_norm, img_segments_only, resize


class SuperAttention(BaseColorizer):
    def __init__(self, model_path: Path, **opts):
        super().__init__("super_attention")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load model and initializing VGG 19 weights and bias
        self.model_color = gen_color_stride_vgg16(dim=2)
        self.model_color.load_state_dict(
            torch.load(model_path / "save_models/checkpoint.pt", map_location=self.device)['state_dict'])
        self.model_color.to(device=self.device)
        self.model_color.eval()

        self.model_percep = percep_vgg19_bn().to(device=self.device)
        self.model_percep.eval()

        for param_percep in self.model_percep.parameters():
            param_percep.requires_grad = False

        for param_color in self.model_color.parameters():
            param_color.requires_grad = False

        self.target_transform = T.ToTensor()

    def get_description(self, benchmark_type: str):
        return ""

    def get_paper_link(self):
        return "https://openaccess.thecvf.com/content/ACCV2022/papers/Camilo_Super-attention_for_exemplar-based_image_colorization_ACCV_2022_paper.pdf"

    def colorize(self, input_path: Path, reference_paths: List[Path]):
        x_real = rgb2gray(io.imread(input_path, pilmode='RGB'))
        x = rgb2gray(resize(io.imread(input_path, pilmode='RGB'), (224, 224)))  # Reading target images in RGB
        ref_real = io.imread(reference_paths[0], pilmode='RGB')
        ref = resize(io.imread(reference_paths[0], pilmode='RGB'), (224, 224))  # Reading ref images in RGB

        size = 224

        if np.ndim(x) == 3:
            x_luminance_classic_real = (x_real[:, :, 0])
            x_luminance_classic = (x[:, :, 0])
        else:
            x_luminance_classic_real = (x_real)
            x_luminance_classic = (x)
            x = x[:, :, np.newaxis]
            x_real = x_real[:, :, np.newaxis]

        ref_new_color = color.rgb2lab(ref)
        ref_luminance_classic = (ref_new_color[:, :, 0] / 100.0)
        ref_chroma = ref_new_color[:, :, 1:] / 127.0

        # Luminance remapping
        x_luminance_map = (np.std(ref_luminance_classic) / np.std(x_luminance_classic)) * (
                x_luminance_classic - np.mean(x_luminance_classic)) + np.mean(ref_luminance_classic)

        # Calculating superpixel label map for target and reference images (Grayscale)

        # The following operations are assumed to be similar, could be further optimized based on `img_segments_only` implementation
        target_slic = [img_segments_only(x_luminance_classic, 2 ** i, int(size / (2 ** i))) for i in range(3)]
        ref_slic = [img_segments_only(ref_luminance_classic, 2 ** i, int(size / (2 ** i))) for i in range(3)]

        # Applying transformation (To tensor) and replicating tensor for gray scale images
        transforms = [self.target_transform(np.expand_dims(slice, axis=2)) for slice in
                      target_slic + ref_slic]
        target_slic_all = transforms[:3]
        ref_slic_all = transforms[3:6]

        x = self.target_transform(x)
        ref_real = self.target_transform(ref_real)
        ref = self.target_transform(ref)
        x_luminance_map = self.target_transform(x_luminance_map[:, :, np.newaxis])
        x_luminance_classic_real = self.target_transform(x_luminance_classic_real[:, :, np.newaxis])
        x_luminance_classic_real_rep = torch.cat(
            (x_luminance_classic_real.float(), x_luminance_classic_real.float(), x_luminance_classic_real.float()),
            dim=0)
        luminance_replicate_map = torch.cat(
            (x_luminance_map.float(), x_luminance_map.float(), x_luminance_map.float()),
            dim=0)
        ref_chroma = self.target_transform(ref_chroma)
        ref_luminance_replicate = torch.cat([self.target_transform(ref_luminance_classic[:, :, np.newaxis])] * 3,
                                            dim=0)

        # Output: x: target image rgb, ref: reference image rdb, luminance_replicate: target grayscale image replicate, ref_luminance_replicate: reference grayscale image replicate
        # labels_torch: label map target image, labels_ref_torch: label map reference image, x_luminance: target grayscale image, ref_luminance: reference grayscale image.
        #
        (img_target_gray, img_target_gray_real, ref_rgb, ref_gray, target_slic, ref_slic_all, img_ref_ab, img_gray_map,
         gray_real, ref_real) = list([default_collate([tensor]) for tensor in [
            x, x_luminance_classic_real_rep, ref, ref_luminance_replicate, target_slic_all, ref_slic_all, ref_chroma,
            luminance_replicate_map, x_luminance_classic_real_rep, ref_real]])

        # Target data
        img_gray_map = img_gray_map.to(device=self.device, dtype=torch.float)
        img_target_gray = img_target_gray.to(device=self.device, dtype=torch.float)
        gray_real = gray_real.to(device=self.device, dtype=torch.float)
        target_slic = target_slic

        # Loading references
        ref_rgb_torch = ref_rgb.to(device=self.device, dtype=torch.float)
        img_ref_gray = ref_gray.to(device=self.device, dtype=torch.float)
        ref_slic_all = ref_slic_all

        # VGG19 normalization
        img_ref_rgb_norm = imagenet_norm(ref_rgb_torch, self.device)

        # VGG19 normalization

        feat1_pred, feat2_pred, feat3_pred, _, _ = self.model_percep(img_ref_rgb_norm)

        ab_pred, pred_Lab_torch, pred_RGB_torch = self.model_color(img_ref_gray,
                                                                   img_target_gray,
                                                                   target_slic,
                                                                   ref_slic_all,
                                                                   img_gray_map,
                                                                   gray_real,
                                                                   feat1_pred, feat2_pred, feat3_pred,
                                                                   self.device)
        grid = torchvision.utils.make_grid(pred_RGB_torch, normalize=True)
        # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        return {"color": im}

    def download_model(self):
        ckpt_path = (Path(__file__).parent.parent.parent /
                     "third_party/super_attention/save_models/checkpoint.pt")
        if not ckpt_path.exists():
            ckpt_path.parent.mkdir(exist_ok=True, parents=True)
            gdown.download(id="1g9_NWvEEd5VewIlNc-HHKWcRUmGrLSav", output=str(ckpt_path))
