"""
   Copyright (C) <2019> <Satoshi Iizuka and Edgar Simo-Serra>

   This work is licensed under the Creative Commons
   Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy
   of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or
   send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

   Satoshi Iizuka, University of Tsukuba
   iizuka@aoni.waseda.jp, http://iizuka.cs.tsukuba.ac.jp/index_eng.html
   Edgar Simo-Serra, Waseda University
   ess@waseda.jp, https://esslab.jp/~ess/
"""
from pathlib import Path
from typing import List
import argparse

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from colorization_benchmark.model_wrappers.base_colorizer import BaseColorizer
from third_party.deepremaster import utils
from third_party.deepremaster.model.remasternet import NetworkC


class DeepRemaster(BaseColorizer):

    def __init__(self, model_path: Path, **opts):
        super().__init__("deepremaster")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load remaster network
        state_dict = torch.load(model_path)
        self.modelC = NetworkC()
        self.modelC.load_state_dict(state_dict['modelC'])
        self.modelC = self.modelC.to(self.device)
        self.modelC.eval()

        self.opts = opts

    def get_description(self, benchmark_type: str):
        text = ("This model is originally designed for film colorization.\n"
                "To run this benchmark the input image is duplicated 5 times.\n"
                "The reference images are supposed to be colored frames chosen from the movies.\n\n"
                "This means that significant differences in the reference images cannot be used, as illustrated below.\n")
        if benchmark_type == "single_reference":
            text += (
                "('Recolor source' vs other rows)\n")
        elif benchmark_type == "multi_reference":
            text += (
                "\n\nAn interesting finding is that certain objects are colored even when they don't appear on the refernce images, "
                "as long as those colors are present in the reference images.\n"
                "This suggests that instead of semantic to semantic matching between gray and reference image, "
                "semantic to color correspondence is learned (at least partially).\n"
                "For example, the sky is colored blue and the leaves green.\n"
                "The semantic matching takes place in feature space where the spatial information is degraded.\n"
                "See noise test vs gray test.\n")

        return text

    def get_paper_link(self):
        return "https://github.com/satoshiiizuka/siggraphasia2019_remastering"

    def colorize(self, input_path: Path, reference_paths: List[Path]) -> dict[str, Image]:
        # Prepare reference images
        aspect_mean = 0
        minedge_dim = 256
        refs = []
        for v in reference_paths:
            refimg = Image.open(v).convert('RGB')
            w, h = refimg.size
            aspect_mean += w / h
            refs.append(refimg)
        aspect_mean /= len(reference_paths)
        target_w = int(256 * aspect_mean) if aspect_mean > 1 else 256
        target_h = 256 if aspect_mean >= 1 else int(256 / aspect_mean)
        refimgs = torch.FloatTensor(len(reference_paths), 3, target_h, target_w)
        for i, v in enumerate(refs):
            refimg = utils.addMergin(v, target_w=target_w, target_h=target_h)
            refimgs[i] = transforms.ToTensor()(refimg)
        refimgs = refimgs.view(1, refimgs.size(0), refimgs.size(1), refimgs.size(2), refimgs.size(3)).to(
            self.device)

        # Load image
        gray_image = Image.open(input_path).convert("RGB")

        v_w, v_h = gray_image.size
        minwh = min(v_w, v_h)
        scale = 1
        if minwh != self.opts["mindim"]:
            scale = self.opts["mindim"] / minwh
        t_w = round(v_w * scale / 16.) * 16
        t_h = round(v_h * scale / 16.) * 16
        block = 5

        # Process
        with torch.no_grad():
            frame = np.array(gray_image.resize((t_w, t_h)))
            nchannels = frame.shape[2]
            if nchannels == 1 or (frame[..., 0] == frame[..., 1]).all() and (frame[..., 0] == frame[..., 2]).all():
                # frame_l = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_l = frame[..., :1]
                # cv2.imwrite(outputdir_in + '%07d.png' % index, frame_l)
                frame_l = torch.from_numpy(frame_l).view(frame_l.shape[0], frame_l.shape[1], 1)
                frame_l = frame_l.permute(2, 0, 1).float()  # HWC to CHW
                frame_l /= 255.
                frame_l = frame_l.view(1, frame_l.size(0), 1, frame_l.size(1), frame_l.size(2))
            elif nchannels == 3:
                # frame = frame[:, :, ::-1]  ## BGR -> RGB
                # cv2.imwrite(outputdir_in + '%07d.png' % index, frame[:, :, ::-1])
                frame_l, frame_ab = utils.convertRGB2LABTensor(frame)
                frame_l = frame_l.view(1, frame_l.size(0), 1, frame_l.size(1), frame_l.size(2))
                frame_ab = frame_ab.view(1, frame_ab.size(0), 1, frame_ab.size(1), frame_ab.size(2))

            input = frame_l  # luminosity
            input = input.to(self.device)

            # Perform colorization
            block_input = torch.tile(input, (1, 1, block, 1, 1))
            # block_input = torch.cat([input, torch.zeros((1, 1, block - 1, input.shape[-2], input.shape[-1]), device=self.device)], dim=2)
            output_ab = self.modelC(block_input, refimgs)
            output_l = block_input.detach().cpu()  # gray
            output_ab = output_ab.detach().cpu()  # chromaticity

        for i in range(block_input.shape[2]):
            out_l = output_l[0, :, i, :, :]
            out_c = output_ab[0, :, i, :, :]
            output = torch.cat((out_l, out_c), dim=0).numpy().transpose((1, 2, 0))
            output = Image.fromarray(np.uint8(utils.convertLAB2RGB(output) * 255))
            return {"color": output}

    def download_model(self):
        ckpt_path = (Path(__file__).parent.parent.parent /
                     "third_party/deepremaster/model/remasternet.pth.tar")
        md5_sum = "1219f5830e4a7208b1c7ba2f089a16c8"
        if not ckpt_path.exists():
            ckpt_path.parent.mkdir(exist_ok=True, parents=True)
            import wget

            try:
                wget.download("http://iizuka.cs.tsukuba.ac.jp/data/remasternet.pth.tar",
                              out=str(ckpt_path.parent))
            except:
                print("Download manually from http://web.archive.org/web/20230409124520/http://iizuka.cs.tsukuba.ac.jp/data/remasternet.pth.tar"
                      f"and place to {ckpt_path}")
        import hashlib
        assert hashlib.md5(ckpt_path.read_bytes()).hexdigest() == md5_sum


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recoloring')
    parser.add_argument('--input', type=Path, required=True, help='Input image')
    parser.add_argument('--reference_dir', type=Path, required=True, help='Path to the reference image directory')
    parser.add_argument('--mindim', type=int, default='320', help='Length of minimum image edges')
    opt = parser.parse_args()

    colorizer = DeepRemaster("./model/remasternet.pth.tar", mindim=opt.mindim)
    colorizer.colorize(opt.input_path, list(opt.reference_dir.glob("*.jpg")))
