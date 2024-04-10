"""
https://github.com/yyang181/NTIRE23-VIDEO-COLORIZATION
BiSTNet is released under the MIT license, while some methods adopted in this project are with other licenses.
Please refer to third_party/ntire23_video_colorization/bistnet_ntire2023/LICENSES.md for the careful check, if you are using our code for commercial matters. Thank @milmor so much for bringing up this concern about license.
Requirements: git+https://github.com/open-mmlab/mmediting.git@0.x
pip install -U openmim\<2.0
mim install mmcv-full\<2.0
PyTorch >= 1.8.0 (please do not use 2.0.1)
torchcontrib
yacs
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List

from PIL import Image
import numpy as np
import cv2
import torch
from torchvision.transforms import transforms
import torchvision.transforms as transform_lib

from colorization_benchmark.model_wrappers.base_colorizer import BaseColorizer

sys.path.insert(0,
                str(Path(__file__).parent.parent.parent / "third_party/ntire23_video_colorization/bistnet_ntire2023"))

sys.argv = [sys.argv[0]]
from third_party.ntire23_video_colorization.bistnet_ntire2023.models.ColorVidNet import ColorVidNet_wBasicVSR_v3
from third_party.ntire23_video_colorization.bistnet_ntire2023.models.ColorVidNet import ATB_block as ATB
from third_party.ntire23_video_colorization.bistnet_ntire2023.models.NonlocalNet import WarpNet_debug, VGG19_pytorch
from third_party.unicolor.sample.ImageMatch.models.ColorVidNet import ColorVidNet

# ATB block
from third_party.ntire23_video_colorization.bistnet_ntire2023.models.ColorVidNet import \
    ColorVidNet_wBasicVSR_v2 as ColorVidNet

# RAFT
from third_party.ntire23_video_colorization.bistnet_ntire2023.models.raft_core.raft import RAFT

from third_party.ntire23_video_colorization.bistnet_ntire2023.test_BiSTNet import load_pth, ColorVid_inference

# SuperSloMo
import third_party.ntire23_video_colorization.bistnet_ntire2023.models.superslomo_model as Superslomo
from torchvision import transforms as superslomo_transforms
from models.NonlocalNet import VGG19_pytorch, WarpNet_debug
from third_party.ntire23_video_colorization.bistnet_ntire2023.utils.util import (batch_lab2rgb_transpose_mc, folder2vid,
                                                                                 mkdir_if_not,
                                                                                 save_frames, save_frames_wOriName,
                                                                                 tensor_lab2rgb, uncenter_l)
from third_party.ntire23_video_colorization.bistnet_ntire2023.utils.util_distortion import CenterPad, Normalize, \
    RGB2Lab, ToTensor

from torchvision import utils as vutils
from third_party.ntire23_video_colorization.bistnet_ntire2023.utils.util import gray2rgb_batch
# HED
from third_party.ntire23_video_colorization.bistnet_ntire2023.models.hed import Network as Hed

# Proto Seg
import pickle
from third_party.ntire23_video_colorization.bistnet_ntire2023.models.protoseg_core.segmentor.tester import \
    Tester_inference as Tester
from third_party.ntire23_video_colorization.bistnet_ntire2023.models.protoseg_core.lib.utils.tools.logger import \
    Logger as Log
from third_party.ntire23_video_colorization.bistnet_ntire2023.models.protoseg_core.lib.utils.tools.configer import \
    Configer, args_parser


class Ntire23(BaseColorizer):
    def __init__(self, model_path: Path, **opts):
        super().__init__("BiSTNet")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        epoch = 105000
        dirName_ckp = '20230311_NTIRE2023'
        nonlocal_test_path = model_path / "checkpoints/finetune_test0610/nonlocal_net_iter_6000.pth"
        color_test_path = model_path / "checkpoints/finetune_test0610/colornet_iter_6000.pth"
        fusenet_path = model_path / "checkpoints" / f"{dirName_ckp}/fusenet_iter_{epoch}.pth"
        atb_path = model_path / "checkpoints" / f"{dirName_ckp}/atb_iter_{epoch}.pth"
        flownet_path = model_path / "data/raft-sintel.pth"
        vgg19_path = model_path / "data/vgg19_conv.pth"

        self.nonlocal_net = WarpNet_debug(1)
        self.colornet = ColorVidNet(7)
        self.vggnet = VGG19_pytorch()
        self.fusenet = ColorVidNet_wBasicVSR_v3(33, flag_propagation=False)

        ### Flownet: raft version
        self.flownet = RAFT(argparse.Namespace(**{"model": flownet_path, "small": False, "mixed_precision": False}))

        ### ATB
        self.atb = ATB()

        self.vggnet.load_state_dict(torch.load(vgg19_path))
        for param in self.vggnet.parameters():
            param.requires_grad = False

        load_pth(self.nonlocal_net, nonlocal_test_path)
        load_pth(self.colornet, color_test_path)
        load_pth(self.fusenet, fusenet_path)
        load_pth(self.flownet, flownet_path)
        load_pth(self.atb, atb_path)
        print("succesfully load nonlocal model: ", nonlocal_test_path)
        print("succesfully load color model: ", color_test_path)
        print("succesfully load fusenet model: ", fusenet_path)
        print("succesfully load flownet model: ", flownet_path)
        print("succesfully load atb model: ", atb_path)

        self.fusenet.eval()
        self.fusenet.cuda()
        self.flownet.eval()
        self.flownet.cuda()
        self.atb.eval()
        self.atb.cuda()
        self.nonlocal_net.eval()
        self.colornet.eval()
        self.vggnet.eval()
        self.nonlocal_net.cuda()
        self.colornet.cuda()
        self.vggnet.cuda()

        # HED
        self.hed = Hed().cuda().eval()
        # forward l
        intWidth = 480
        intHeight = 320
        meanlab = [-50, -50, -50]  # (A - mean) / std
        stdlab = [100, 100, 100]  # (A - mean) / std

        # proto seg
        meanlab_protoseg = [0.485, 0.485, 0.485]  # (A - mean) / std
        stdlab_protoseg = [0.229, 0.229, 0.229]  # (A - mean) / std
        self.trans_forward_protoseg_lll = superslomo_transforms.Compose(
            [superslomo_transforms.Normalize(mean=meanlab, std=stdlab),
             superslomo_transforms.Normalize(mean=meanlab_protoseg, std=stdlab_protoseg)])
        opt_image_size = [448, 896]
        self.transform = transforms.Compose(
            # [CenterPad(opt.image_size), transform_lib.CenterCrop(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
            [superslomo_transforms.Resize(opt_image_size), RGB2Lab(), ToTensor(), Normalize()]
        )

        self.transform_full_l = transforms.Compose(
            # [CenterPad(opt.image_size), transform_lib.CenterCrop(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
            [RGB2Lab(), ToTensor(), Normalize()]
        )

    def get_description(self, benchmark_type: str):
        return ("This method is originally designed for video colorization and is limited to 2 reference images.\n"
                "The model is changed so that the input is 2 identical frames. The flow map is set to zeros and "
                "the similarity fusion is done using argmax for an arbitrary number of reference images.")

    def get_paper_link(self):
        return "https://github.com/yyang181/NTIRE23-VIDEO-COLORIZATION"

    def colorize(self, input_path: Path, reference_paths: List[Path]):
        filenames = [str(input_path)] * 2

        I_list = [Image.open(frame_name).convert('RGB') for frame_name in filenames]
        I_list_large = [self.transform(frame1).unsqueeze(0).cuda() for frame1 in I_list]
        opt_image_size_ori = np.shape(I_list[0])[:2]

        I_list_large_full_l = [self.transform_full_l(frame1).unsqueeze(0).cuda() for frame1 in I_list]

        I_list = [torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear") for IA_lab_large in
                  I_list_large]
        ref_name1 = reference_paths[0]
        with torch.no_grad():
            frame_ref1 = Image.open(ref_name1).convert('RGB')
            IB_lab_large1 = self.transform(frame_ref1).unsqueeze(0).cuda()
            IB_lab1 = torch.nn.functional.interpolate(IB_lab_large1, scale_factor=0.5, mode="bilinear")
            I_reference_rgb_from_gray = gray2rgb_batch(IB_lab1[:, 0:1, :, :])
            features_B1 = self.vggnet(I_reference_rgb_from_gray, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

        ref_name2 = reference_paths[-1]
        with torch.no_grad():
            frame_ref2 = Image.open(ref_name2).convert('RGB')
            IB_lab_large2 = self.transform(frame_ref2).unsqueeze(0).cuda()
            IB_lab2 = torch.nn.functional.interpolate(IB_lab_large2, scale_factor=0.5, mode="bilinear")
            I_reference_rgb_from_gray = gray2rgb_batch(IB_lab2[:, 0:1, :, :])
            features_B2 = self.vggnet(I_reference_rgb_from_gray, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

        # ColorVid inference (for each reference image) flag forward reverses frame order
        colorvid1, similarity_map_list1 = ColorVid_inference(I_list, IB_lab1, features_B1, self.vggnet,
                                                             self.nonlocal_net,
                                                             self.colornet,
                                                             joint_training=False, flag_forward=True)
        colorvid2, similarity_map_list2 = ColorVid_inference(I_list, IB_lab2, features_B2, self.vggnet,
                                                             self.nonlocal_net,
                                                             self.colornet,
                                                             joint_training=False, flag_forward=False)
        colorvid2.reverse()
        similarity_map_list2.reverse()

        # fig, axes = plt.subplots(nrows=3, ncols=4)
        # for ax in fig.axes:
        #     ax.set_axis_off()
        # for i in range(len(colorvid1)):
        #     axes[0, 2 * i].imshow(np.array([frame_ref1, frame_ref2][i]) / 255)
        #     axes[i + 1, 0].imshow(color.lab2rgb((colorvid1[i][0].cpu().permute(1, 2, 0)) + np.array([50, 0, 0])))
        #     axes[i + 1, 1].imshow(similarity_map_list1[i][0].cpu().permute(1, 2, 0), vmin=0, vmax=1)
        #     axes[i + 1, 2].imshow(color.lab2rgb((colorvid2[i][0].cpu().permute(1, 2, 0)) + np.array([50, 0, 0])))
        #     axes[i + 1, 3].imshow(similarity_map_list2[i][0].cpu().permute(1, 2, 0), vmin=0, vmax=1)
        # # for ax in fig.axes:
        # #     ax.set_axis_off()
        # plt.savefig("fig.png")
        # plt.close()
        # FUSION SimilarityMap  # todo: Could use argmax for N ref images
        similarityMap = []
        for i in range(len(similarity_map_list1)):
            # Fusion Mask Test
            FusionMask = torch.gt(similarity_map_list1[i], similarity_map_list2[i])
            FusionMask = torch.cat([FusionMask, FusionMask, FusionMask], dim=1)

            Fused_Color = colorvid2[i]
            Fused_Color[FusionMask] = colorvid1[i][FusionMask]
            # plt.imshow(color.lab2rgb((Fused_Color[0].cpu().permute(1, 2, 0)) + np.array([50, 0, 0])))
            # plt.savefig(f"fused_{i}.png")
            similarityMap.append(Fused_Color)

        # HED EdgeMask
        edgemask = self.HED_EdgeMask(I_list)
        # for i in range(len(edgemask)):
        #     plt.imshow(edgemask[i][0].cpu())
        #     plt.savefig(f"edge_{i}.png")

        # Proto Seg
        segmask = self.proto_segmask(I_list, flag_save_protoseg=False)

        # flows_forward, flows_backward = bipropagation(colorvid1, colorvid2, I_list, flownet, atb, flag_save_flow_warp=False)
        # 2 frames are identical, flow should be 0
        flows_forward = flows_backward = [torch.zeros((1, 2, *color_frame.shape[2:]), device=color_frame.device) for
                                          color_frame in colorvid1]
        # plt.imshow(flow_to_image(flows_forward[0])[0].cpu().permute(1, 2, 0))
        # plt.savefig(f"flow_fw.png")
        #
        # plt.imshow(flow_to_image(flows_backward[0])[0].cpu().permute(1, 2, 0))
        # plt.savefig(f"flow_bw.png")

        print('fusenet v1: concat ref1+ref2')
        joint_training = False
        i_idx = 0
        I_current_l = I_list[i_idx][:, :1, :, :]
        I_current_ab = I_list[i_idx][:, 1:, :, :]

        # module: atb_test
        feat_fused, ab_fuse_videointerp, ab_fuse_atb = self.atb(colorvid1, colorvid2, flows_forward, flows_backward)

        fuse_input = torch.cat(
            [I_list[i_idx][:, :1, :, :], colorvid1[i_idx][:, 1:, :, :], colorvid2[i_idx][:, 1:, :, :],
             feat_fused[i_idx], segmask[i_idx, :, :, :].unsqueeze(0), edgemask[i_idx, :, :, :].unsqueeze(0),
             similarityMap[i_idx][:, 1:, :, :]], dim=1)

        with torch.no_grad():
            level1_shape = [fuse_input.shape[2], fuse_input.shape[3]]
            level2_shape = [int(fuse_input.shape[2] / 2), int(fuse_input.shape[3] / 2)]
            level3_shape = [int(fuse_input.shape[2] / 4), int(fuse_input.shape[3] / 4)]

            # v0
            resize_b1tob2 = transform_lib.Resize(level2_shape)
            resize_b2tob3 = transform_lib.Resize(level3_shape)

            input_pyr_b1 = fuse_input
            input_pyr_b2 = resize_b1tob2(fuse_input)
            input_pyr_b3 = resize_b2tob3(input_pyr_b2)

            input_fusenet = [input_pyr_b1, input_pyr_b2, input_pyr_b3]
            output_fusenet = self.fusenet(input_fusenet)

            I_current_ab_predict = output_fusenet[0]

        IA_lab_large = I_list_large_full_l[i_idx]
        curr_bs_l = IA_lab_large[:, 0:1, :, :]
        curr_predict = (
                torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2,
                                                mode="bilinear") * 1.25
        )
        curr_predict = (
            torch.nn.functional.interpolate(curr_predict, size=opt_image_size_ori, mode="bilinear")
        )

        # filtering
        wls_filter_on = True
        lambda_value = 500
        sigma_color = 4
        if wls_filter_on:
            guide_image = uncenter_l(curr_bs_l) * 255 / 100
            wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
                guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), lambda_value, sigma_color
            )
            curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
            curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
            curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0)
            curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)
            curr_predict_filter = torch.cat((curr_predict_a, curr_predict_b), dim=1)
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
        else:
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])

        color = Image.fromarray(IA_predict_rgb)

        return {"color": color}

    def HED_EdgeMask(self, I_list):
        joint_training = False
        I_current_l = torch.cat(I_list, dim=0)[:, :1, :, :]
        I_current_lll = torch.cat([I_current_l, I_current_l, I_current_l], dim=1)

        ###### HED: Edge Detection ######
        tenInput2 = I_current_lll

        with torch.autograd.set_grad_enabled(joint_training):
            hed_edge2 = self.hed(tenInput2).clip(0.0, 1.0)

        hed_edge_ori2 = hed_edge2
        return hed_edge_ori2

    def proto_segmask(self, I_list, flag_save_protoseg=False):
        # trans input resolution
        I_current_l = torch.cat(I_list, dim=0)[:, :1, :, :]
        I_current_lll = torch.cat([I_current_l, I_current_l, I_current_l], dim=1)
        input_protoseg = self.trans_forward_protoseg_lll(I_current_lll)
        args_parser.__setattr__("data:data_dir", str(self.model_path / "models/protoseg_core/Cityscapes"))
        args_parser.__setattr__("network:resume", str(self.model_path / args_parser.__getattribute__("network:resume")))
        args_parser.configs = str(self.model_path / "models/protoseg_core/configs/cityscapes/H_48_D_4_proto.json")
        configer = Configer(args_parser)
        # abs_data_dir = [os.path.expanduser(x) for x in data_dir]
        # project_dir = os.path.dirname(os.path.realpath(__file__))

        model = Tester(configer)

        with torch.no_grad():
            outputs = model.test_deep_exemplar(input_protoseg)
        return outputs

    def download_model(self):
        ckpt_path = (Path(__file__).parent.parent.parent /
                     "third_party/ntire23_video_colorization/bistnet_ntire2023/data/vgg19_gray.pth")
        save_path = (Path(__file__).parent.parent.parent / "third_party/ntire23_video_colorization/bistnet_ntire2023/")

        if not ckpt_path.exists():
            import wget
            import zipfile
            for file in ["checkpoints", "data", "models"]:
                filename = f"{file}.zip"
                download_path = f"https://github.com/yyang181/NTIRE23-VIDEO-COLORIZATION/releases/download/v1.0.3/{filename}"
                (save_path / filename).parent.mkdir(exist_ok=True, parents=True)
                wget.download(str(download_path), out=str(save_path / filename))
                with zipfile.ZipFile(save_path / filename, "r") as zipref:
                    zipref.extractall(save_path)
