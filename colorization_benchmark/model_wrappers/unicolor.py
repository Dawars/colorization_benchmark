"""
UniColor: https://github.com/luckyhzt/unicolor

Requirements: opencv-python-4.9 git+https://github.com/openai/CLIP.git-1.0 kornia-0.7.2 nltk-3.8.1 numba-0.59

"""
import sys
from pathlib import Path
from typing import List
import argparse


from colorization_benchmark.base_colorizer import Colorizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party/unicolor/sample"))
sys.path.insert(1, str(Path(__file__).parent.parent.parent / "third_party/unicolor/sample/ImageMatch"))
sys.path.insert(2, str(Path(__file__).parent.parent.parent / "third_party/unicolor/framework"))

from third_party.unicolor.sample.colorizer import Colorizer
from third_party.unicolor.sample.utils_func import *


class UniColor(Colorizer):
    method_name = "unicolor"
    description = "This model generate diverse results where the color is not constrained by the reference image."

    def __init__(self, model_path: Path, **opts):
        super(Colorizer).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.colorizer = Colorizer(model_path, self.device, [256, 256], load_clip=True, load_warper=True)

        self.opts = opts

    def colorize(self, input_path: Path, reference_paths: List[Path]):
        gray_image = Image.open(input_path).convert("L")

        attention = None
        if reference_paths:
            ref_image = Image.open(reference_paths[0]).convert('RGB')
            # Get hint points
            with torch.no_grad():
                points, warped = self.colorizer.get_strokes_from_exemplar(gray_image, ref_image)
            # warped = color_resize(gray_image, warped)  # warp chromaticity based on correspondence
            # Show warped image
            # display(warped)
            attention = point_img = draw_strokes(gray_image, [256, 256], points)
            with torch.no_grad():
                output = self.colorizer.sample(gray_image, points, topk=100)

            # output = Image.fromarray(np.concatenate([np.array(point_img), np.array(I_exp)], axis=1)))
        else:
            with torch.no_grad():
                output = self.colorizer.sample(gray_image, [], topk=100)
        return {"color": output, "attention": attention}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recoloring')
    parser.add_argument('--input', type=Path, required=True, help='Input image')
    parser.add_argument('--reference_dir', type=Path, required=True, help='Path to the reference image directory')
    parser.add_argument('--mindim', type=int, default='320', help='Length of minimum image edges')
    opt = parser.parse_args()

    colorizer = UniColor("./model/remasternet.pth.tar", mindim=opt.mindim)
    colorizer.colorize(opt.input_path, list(opt.reference_dir.glob("*.jpg")))
