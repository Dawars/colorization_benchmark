"""
https://github.com/KIMGEONUNG/BigColor

Requirements:timm==0.6.5 torch-ema==0.3 tensorboard==2.16
"""
import sys
from pathlib import Path
from argparse import Namespace
from typing import List
from os.path import join, exists
import pickle
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage, Grayscale, Resize, Compose
from torch_ema import ExponentialMovingAverage
import timm
from math import ceil
import torch
from PIL import Image

from colorization_benchmark.model_wrappers.base_colorizer import BaseColorizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party/bigcolor"))

from third_party.bigcolor.train import Colorizer
from third_party.bigcolor.utils.common_utils import set_seed, rgb2lab, lab2rgb

MODEL2SIZE = {'resnet50d': 224,
              'tf_efficientnet_l2_ns_475': 475}


def fusion(img_gray, img_rgb):
    img_gray *= 100
    ab = rgb2lab(img_rgb)[..., 1:, :, :]
    lab = torch.cat([img_gray, ab], dim=0)
    rgb = lab2rgb(lab)
    return rgb


class BigColor(BaseColorizer):
    def __init__(self, model_path: Path, **opts):
        super().__init__("bigcolor")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        path_config = './pretrained/config.pickle'
        path_ckpt_g = './pretrained/G_ema_256.pth'
        path_ckpt = './ckpts/baseline_1000'
        self.args = Namespace(**opts)

        print('Target Epoch is %03d' % self.args.epoch)

        path_eg = model_path / "ckpts/bigcolor" / ('EG_%03d.ckpt' % self.args.epoch)
        path_eg_ema = model_path / "ckpts/bigcolor" / ('EG_EMA_%03d.ckpt' % self.args.epoch)
        path_args = model_path / "ckpts/bigcolor" / 'args.pkl'

        if not exists(path_eg):
            raise FileNotFoundError(path_eg)
        if not exists(path_args):
            raise FileNotFoundError(path_args)

        # Load Configuratuion
        with open(model_path / path_config, 'rb') as f:
            self.config = pickle.load(f)
        with open(model_path / path_args, 'rb') as f:
            self.args_loaded = pickle.load(f)

        # Load Colorizer
        self.EG = Colorizer(self.config,
                            str(model_path / path_ckpt_g),
                            self.args_loaded.norm_type,
                            id_mid_layer=self.args_loaded.num_layer,
                            activation=self.args_loaded.activation,
                            use_attention=self.args_loaded.use_attention,
                            dim_f=self.args.dim_f)
        self.EG.load_state_dict(torch.load(path_eg, map_location='cpu'), strict=True)
        self.EG_ema = ExponentialMovingAverage(self.EG.parameters(), decay=0.99)
        self.EG_ema.load_state_dict(torch.load(path_eg_ema, map_location='cpu'))

        self.EG.eval()
        self.EG.float()
        self.EG.to(self.device)

        if self.args.use_ema:
            print('Use EMA')
            self.EG_ema.copy_to()

        # Load Classifier
        self.classifier = timm.create_model(
            self.args.cls_model,
            pretrained=True,
            num_classes=1000
        ).to(self.device)
        self.classifier.eval()
        self.size_cls = MODEL2SIZE[self.args.cls_model]

        self.resizer = None
        if self.args.type_resize == 'absolute':
            self.resizer = Resize((self.args.size_target))
        elif self.args.type_resize == 'original':
            self.resizer = Compose([])
        elif self.args.type_resize == 'square':
            self.resizer = Resize((self.args.size_target, self.args.size_target))
        elif self.args.type_resize == 'powerof':
            assert self.args.size_target % (2 ** self.args.num_power) == 0

            def resizer(x):
                length_long = max(x.shape[-2:])
                length_sort = min(x.shape[-2:])
                unit = ceil((length_long * (self.args.size_target / length_sort)) / (2 ** self.args.num_power))
                long = unit * (2 ** self.args.num_power)

                if x.shape[-1] > x.shape[-2]:
                    fn = Resize((self.args.size_target, long))
                else:
                    fn = Resize((long, self.args.size_target))

                return fn(x)

            self.resizer = resizer
        elif self.args.type_resize == 'patch':
            self.resizer = Resize((self.args.size_target))
        else:
            raise Exception('Invalid resize type')

    def get_description(self, benchmark_type: str):
        return "BigColor also has multi-modal versions via the random code z and the class code c which enables class-specific feature extraction"

    def get_paper_link(self):
        return "https://kimgeonung.github.io/assets/bigcolor/bigcolor_main.pdf"

    def colorize(self, input_path: Path, reference_paths: List[Path]):
        set_seed(2)

        im = Image.open(input_path)
        x = ToTensor()(im)
        if x.shape[0] != 1:
            x = Grayscale()(x)

        size = x.shape[1:]

        x = x.unsqueeze(0)
        x = x.to(self.device)
        z = torch.zeros((1, self.args_loaded.dim_z)).to(self.device)
        z.normal_(mean=0, std=0.8)

        # Classification
        x_cls = x.repeat(1, 3, 1, 1)
        x_cls = Resize((self.size_cls, self.size_cls))(x_cls)
        c = self.classifier(x_cls)
        cs = torch.topk(c, self.args.topk)[1].reshape(-1)
        c = torch.LongTensor([cs[0]]).to(self.device)

        for c in cs:
            c = torch.LongTensor([c]).to(self.device)
            x_resize = self.resizer(x)

            if self.args.type_resize == 'patch':
                length = max(x_resize.shape[-2:])
                num_patch = ceil(length / self.args.size_target)
                direction = 'v' if x.shape[-1] < x.shape[-2] else 'h'

                patchs = []
                for i in range(num_patch):
                    patch = torch.zeros((self.args.size_target, self.args.size_target))
                    if i + 1 == num_patch:  # last
                        start = -self.args.size_target
                        end = length
                    else:
                        start = i * self.args.size_target
                        end = (i + 1) * self.args.size_target

                    if direction == 'v':
                        patch = x_resize[..., start:end, :]
                    elif direction == 'h':
                        patch = x_resize[..., :, start:end]
                    else:
                        raise Exception('Invalid direction')
                    patchs.append(patch)

                outputs = [self.EG(patch, c, z).add(1).div(2) for patch in patchs]
                cloth = torch.zeros((1, 3, x_resize.shape[-2],
                                     x_resize.shape[-1]))
                for i in range(num_patch):
                    output = outputs[i]
                    if i + 1 == num_patch:  # last
                        start = -self.args.size_target
                        end = length
                    else:
                        start = i * self.args.size_target
                        end = (i + 1) * self.args.size_target

                    if direction == 'v':
                        cloth[..., start:end, :] = output
                    elif direction == 'h':
                        cloth[..., :, start:end] = output
                    else:
                        raise Exception('Invalid direction')

                output = cloth
                im = ToPILImage()(output.squeeze(0))
                im.show()
                raise NotImplementedError()

            with torch.no_grad():
                output = self.EG(x_resize, c, z)
                output = output.add(1).div(2)

            if self.args.no_upsample:
                size_output = x_resize.shape[-2:]
                x_rs = x_resize.squeeze(0).cpu()
            else:
                size_output = size
                x_rs = x.squeeze(0).cpu()

            output = transforms.Resize(size_output)(output)
            output = output.squeeze(0)
            output = output.detach().cpu()

            if self.args.use_rgb:
                x_img = output
            else:
                x_img = fusion(x_rs, output)
            color = ToPILImage()(x_img)

        return {"color": color}
