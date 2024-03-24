"""
https://github.com/ericsujw/InstColorization

Requirements: https://github.com/facebookresearch/detectron2.git, ipython

"""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import gdown
import numpy as np
import cv2
import torch
from PIL import Image
from skimage.util import img_as_ubyte
from torch.utils.data import default_collate
from torchvision.transforms import transforms

from colorization_benchmark.model_wrappers.base_colorizer import BaseColorizer

# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party/inst_colorization"))

from third_party.inst_colorization.models import create_model
from third_party.inst_colorization import image_util
from third_party.inst_colorization.util import util
from third_party.inst_colorization.options.train_options import TestOptions


def gen_maskrcnn_bbox_fromPred(pred_bbox, pred_scores, box_num_upbound=-1):
    '''
    ## Arguments:
    - pred_data: Detectron2 predict results
    - box_num_upbound: object bounding boxes number. Default: -1 means use all the instances.
    '''
    pred_bbox = pred_bbox.astype(np.int32)
    if box_num_upbound > 0 and pred_bbox.shape[0] > box_num_upbound:
        index_mask = np.argsort(pred_scores, axis=0)[pred_scores.shape[0] - box_num_upbound: pred_scores.shape[0]]
        pred_bbox = pred_bbox[index_mask]
    # pred_scores = pred_data['scores']
    # index_mask = pred_scores > 0.9
    # pred_bbox = pred_bbox[index_mask].astype(np.int32)
    return pred_bbox


class InstColorization(BaseColorizer):
    def __init__(self, model_path: Path, **opts):
        self.model_path = model_path
        super().__init__("inst_colorization")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.predictor = DefaultPredictor(cfg)

        sys.argv = [sys.argv[0]]
        self.opt = TestOptions().parse()
        self.model = create_model(self.opt)
        # fusion_weight_path = "coco_finetuned_mask_256" # fine-tuned on cocostuff
        fusion_weight_path = "coco_finetuned_mask_256_ffs"
        GF_path = model_path / "checkpoints" / fusion_weight_path / "latest_net_GF.pth"
        GF_state_dict = torch.load(GF_path)

        G_path = model_path / "checkpoints" / fusion_weight_path / "latest_net_G.pth"
        G_state_dict = torch.load(G_path)

        GComp_path = model_path / "checkpoints" / fusion_weight_path / "latest_net_GComp.pth"
        GComp_state_dict = torch.load(GComp_path)

        self.model.netGF.load_state_dict(GF_state_dict, strict=False)
        self.model.netG.module.load_state_dict(G_state_dict, strict=False)
        self.model.netGComp.module.load_state_dict(GComp_state_dict, strict=False)
        self.model.netGF.eval()
        self.model.netG.eval()
        self.model.netGComp.eval()

        self.transforms = transforms.Compose(
            [transforms.Resize((self.opt.fineSize, self.opt.fineSize), interpolation=2),
             transforms.ToTensor()])
        self.final_size = self.opt.fineSize
        self.box_num = 8

    def get_description(self, benchmark_type: str):
        return ""

    def get_paper_link(self):
        return "https://arxiv.org/abs/2005.10825"

    def colorize(self, input_path: Path, reference_paths: List[Path]):
        # BB
        img = cv2.imread(str(input_path))
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)
        outputs = self.predictor(l_stack)
        pred_bbox = outputs["instances"].pred_boxes.to(torch.device('cpu')).tensor.numpy()
        pred_scores = outputs["instances"].scores.cpu().data.numpy()

        # data loader
        pred_bbox = gen_maskrcnn_bbox_fromPred(pred_bbox, pred_scores, self.box_num)

        img_list = []
        pil_img = image_util.read_to_pil(input_path)
        img_list.append(self.transforms(pil_img))

        cropped_img_list = []
        index_list = range(len(pred_bbox))
        box_info, box_info_2x, box_info_4x, box_info_8x = np.zeros((4, len(index_list), 6))
        for i in index_list:
            startx, starty, endx, endy = pred_bbox[i]
            box_info[i] = np.array(image_util.get_box_info(pred_bbox[i], pil_img.size, self.final_size))
            box_info_2x[i] = np.array(image_util.get_box_info(pred_bbox[i], pil_img.size, self.final_size // 2))
            box_info_4x[i] = np.array(image_util.get_box_info(pred_bbox[i], pil_img.size, self.final_size // 4))
            box_info_8x[i] = np.array(image_util.get_box_info(pred_bbox[i], pil_img.size, self.final_size // 8))
            cropped_img = self.transforms(pil_img.crop((startx, starty, endx, endy)))
            cropped_img_list.append(cropped_img)
        data_raw = {}
        data_raw['full_img'] = torch.stack(img_list)
        if len(pred_bbox) > 0:
            data_raw['cropped_img'] = torch.stack(cropped_img_list)
            data_raw['box_info'] = torch.from_numpy(box_info).type(torch.long)
            data_raw['box_info_2x'] = torch.from_numpy(box_info_2x).type(torch.long)
            data_raw['box_info_4x'] = torch.from_numpy(box_info_4x).type(torch.long)
            data_raw['box_info_8x'] = torch.from_numpy(box_info_8x).type(torch.long)
            data_raw['empty_box'] = False
        else:
            data_raw['empty_box'] = True

        data_raw = default_collate([data_raw])
        # colorize
        data_raw['full_img'][0] = data_raw['full_img'][0].to(self.device)
        if data_raw['empty_box'][0] == False:
            data_raw['cropped_img'][0] = data_raw['cropped_img'][0].to(self.device)
            box_info = data_raw['box_info'][0]
            box_info_2x = data_raw['box_info_2x'][0]
            box_info_4x = data_raw['box_info_4x'][0]
            box_info_8x = data_raw['box_info_8x'][0]
            cropped_data = util.get_colorization_data(data_raw['cropped_img'], self.opt, ab_thresh=0, p=self.opt.sample_p)
            full_img_data = util.get_colorization_data(data_raw['full_img'], self.opt, ab_thresh=0, p=self.opt.sample_p)
            self.model.set_input(cropped_data)
            self.model.set_fusion_input(full_img_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
            self.model.forward()
        else:
            # count_empty += 1
            full_img_data = util.get_colorization_data(data_raw['full_img'], self.opt, ab_thresh=0, p=self.opt.sample_p)
            self.model.set_forward_without_box(full_img_data)

        out_img = torch.clamp(util.lab2rgb(torch.cat(
            (self.model.full_real_A.type(torch.cuda.FloatTensor), self.model.fake_B_reg.type(torch.cuda.FloatTensor)),
            dim=1), self.opt), 0.0, 1.0)
        out_img = np.transpose(out_img.cpu().data.numpy()[0], (1, 2, 0))
        color = Image.fromarray(img_as_ubyte(out_img))
        return {"color": color}

    def download_model(self):
        ckpt_path = (Path(__file__).parent.parent.parent /
                     "third_party/inst_colorization/checkpoints/coco_finetuned_mask_256_ffs/latest_net_GF.pth")
        save_path = (Path(__file__).parent.parent.parent /
                     "third_party/inst_colorization/checkpoints.zip")
        if not ckpt_path.exists():
            import gdown
            import zipfile
            gdown.download(id="1Xb-DKAA9ibCVLqm8teKd1MWk6imjwTBh", output=str(save_path))
            with zipfile.ZipFile(save_path, "r") as zipref:
                zipref.extractall(self.model_path)
