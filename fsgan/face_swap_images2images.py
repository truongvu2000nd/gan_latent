""" Images to images face swapping. """

import os
import face_alignment
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import torch
import landmark_transforms
import sys
sys.path.append("..")
import fsgan.utils.utils as utils
from fsgan.utils.bbox_utils import scale_bbox
from fsgan.utils.seg_utils import blend_seg_pred
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils.video_utils import extract_landmarks_bboxes_euler_from_images
from fsgan.utils.img_utils import rgb2tensor, tensor2rgb
from fsgan.utils.bbox_utils import scale_bbox, crop_img, get_main_bbox

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def unnormalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor2bgr(img_tensor):
    output_img = unnormalize(img_tensor.clone(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    output_img = output_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img[:, :, ::-1] * 255).astype('uint8')

    return output_img


def transfer_mask(img1, img2, mask):
    mask = mask.view(mask.shape[0], 1, mask.shape[1], mask.shape[2]).repeat(1, 3, 1, 1).float()
    out = img1 * mask + img2 * (1 - mask)

    return out


def create_pyramid(img, n=1):
    # If input is a list or tuple return it as it is (probably already a pyramid)
    if isinstance(img, (list, tuple)):
        return img

    pyd = [img]
    for i in range(n - 1):
        pyd.append(torch.nn.functional.avg_pool2d(pyd[-1], 3, stride=2, padding=1, count_include_pad=False))

    return pyd


def crop2img(img, crop, bbox):
    scaled_bbox = scale_bbox(bbox)
    scaled_crop = cv2.resize(crop, (scaled_bbox[3], scaled_bbox[2]), interpolation=cv2.INTER_CUBIC)
    left = -scaled_bbox[0] if scaled_bbox[0] < 0 else 0
    top = -scaled_bbox[1] if scaled_bbox[1] < 0 else 0
    right = scaled_bbox[0] + scaled_bbox[2] - img.shape[1] if (scaled_bbox[0] + scaled_bbox[2] - img.shape[1]) > 0 else 0
    bottom = scaled_bbox[1] + scaled_bbox[3] - img.shape[0] if (scaled_bbox[1] + scaled_bbox[3] - img.shape[0]) > 0 else 0
    crop_bbox = np.array([left, top, scaled_bbox[2] - left - right, scaled_bbox[3] - top - bottom])
    scaled_bbox += np.array([left, top, -left - right, -top - bottom])

    out_img = img.copy()
    out_img[scaled_bbox[1]:scaled_bbox[1] + scaled_bbox[3], scaled_bbox[0]:scaled_bbox[0] + scaled_bbox[2]] = \
        scaled_crop[crop_bbox[1]:crop_bbox[1] + crop_bbox[3], crop_bbox[0]:crop_bbox[0] + crop_bbox[2]]

    return out_img


class FSGAN(nn.Module):
    def __init__(
        self, arch='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
        reenactment_model_path='./weights/ijbc_msrunet_256_2_0_reenactment_v1.pth',
        seg_model_path='./weights/lfw_figaro_unet_256_2_0_segmentation_v1.pth',
        inpainting_model_path='./weights/ijbc_msrunet_256_2_0_inpainting_v1.pth',
        blend_model_path='./weights/ijbc_msrunet_256_2_0_blending_v1.pth',
        pose_model_path='./weights/hopenet_robust_alpha1.pth',
        pil_transforms1=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                        'landmark_transforms.Pyramids(2)'),
        pil_transforms2=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                        'landmark_transforms.Pyramids(2)', 'landmark_transforms.LandmarksToHeatmaps'),
        tensor_transforms1=('landmark_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
        tensor_transforms2=('landmark_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
        output_path=None, min_radius=2.0, crop_size=256, reverse_output=False, verbose=0, output_crop=False,
        display=False
    ):
        super().__init__()
        # Initialize models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cpu")
        self.device = device

        # Load face reenactment model
        self.Gr = obj_factory(arch).to(device)
        checkpoint = torch.load(reenactment_model_path)
        self.Gr.load_state_dict(checkpoint['state_dict'])
        self.Gr.eval()

        # Load face segmentation model
        if seg_model_path is not None:
            print('Loading face segmentation model: "' + os.path.basename(seg_model_path) + '"...')
            if seg_model_path.endswith('.pth'):
                checkpoint = torch.load(seg_model_path)
                self.Gs = obj_factory(checkpoint['arch']).to(device)
                self.Gs.load_state_dict(checkpoint['state_dict'])
            else:
                self.Gs = torch.jit.load(seg_model_path, map_location=device)
            if self.Gs is None:
                raise RuntimeError('Failed to load face segmentation model!')
            self.Gs.eval()
        else:
            self.Gs = None

        # Load face inpainting model
        if inpainting_model_path is not None:
            print('Loading face inpainting model: "' + os.path.basename(inpainting_model_path) + '"...')
            if inpainting_model_path.endswith('.pth'):
                checkpoint = torch.load(inpainting_model_path)
                self.Gi = obj_factory(checkpoint['arch']).to(device)
                self.Gi.load_state_dict(checkpoint['state_dict'])
            else:
                self.Gi = torch.jit.load(inpainting_model_path, map_location=device)
            if self.Gi is None:
                raise RuntimeError('Failed to load face segmentation model!')
            self.Gi.eval()
        else:
            self.Gi = None

        # Load face blending model
        checkpoint = torch.load(blend_model_path)
        self.Gb = obj_factory(checkpoint['arch']).to(device)
        self.Gb.load_state_dict(checkpoint['state_dict'])
        self.Gb.eval()

        # Initialize transformations
        pil_transforms1 = obj_factory(pil_transforms1) if pil_transforms1 is not None else []
        pil_transforms2 = obj_factory(pil_transforms2) if pil_transforms2 is not None else []
        tensor_transforms1 = obj_factory(tensor_transforms1) if tensor_transforms1 is not None else []
        tensor_transforms2 = obj_factory(tensor_transforms2) if tensor_transforms2 is not None else []
        self.img_transforms1 = landmark_transforms.ComposePyramids(pil_transforms1 + tensor_transforms1)
        self.img_transforms2 = landmark_transforms.ComposePyramids(pil_transforms2 + tensor_transforms2)

    def _extract_landmarks_bboxes_euler_from_images(self, imgs):
        # frame_indices = []
        # landmarks = []
        # bboxes = []

        # # For each image in the directory
        # for i, img in imgs:
        #     img_rgb = img.permute(2, 0, 1).cpu().numpy()
        #     detected_faces = self.fa.face_detector.detect_from_image(img_rgb.copy())

        #     # Skip current frame there if no faces were detected
        #     if len(detected_faces) == 0:
        #         continue
        #     curr_bbox = get_main_bbox(np.array(detected_faces)[:, :4], img_rgb.shape[:2])
        #     detected_faces = [curr_bbox]

        #     preds = self.fa.get_landmarks(img_rgb, detected_faces)
        #     curr_landmarks = preds[0]
        #     # curr_bbox = detected_faces[0][:4]

        #     # Convert bounding boxes format from [min, max] to [min, size]
        #     curr_bbox[2:] = curr_bbox[2:] - curr_bbox[:2] + 1

        #     # Append to list
        #     frame_indices.append(i)
        #     landmarks.append(curr_landmarks)
        #     bboxes.append(curr_bbox)

        # detected_faces = self.fa.face_detector.de

        # # Convert to numpy array format
        # frame_indices = np.array(frame_indices)
        # landmarks = np.array(landmarks)
        # bboxes = np.array(bboxes)

        # return landmarks, bboxes
        landmarks, _, bboxes = self.fa.get_landmarks_from_batch(imgs, return_bboxes=True)
        landmarks, bboxes = np.array(landmarks), np.array(bboxes)
        return landmarks, bboxes

    def forward(self, img1, img2):
        device = self.device
        source_landmarks, source_bboxes = \
            self._extract_landmarks_bboxes_euler_from_images(img1)

        target_landmarks, target_bboxes = \
            self._extract_landmarks_bboxes_euler_from_images(img2)
        outputs = []
        for k, img1_tensor, img2_tensor in enumerate(zip(img1, img2)):
            # source_img_rgb = img1_tensor.permute(2, 0, 1).cpu().numpy()
            source_img_rgb = TF.to_pil_image(img1_tensor)
            curr_source_tensor, curr_source_landmarks, curr_source_bbox = self.img_transforms1(
                source_img_rgb, source_landmarks[k], source_bboxes[k])

            # target_img_rgb = img2_tensor.permute(2, 0, 1).cpu().numpy()
            target_img_rgb = TF.to_pil_image(img2_tensor)
            curr_target_tensor, curr_target_landmarks, curr_target_bbox = self.img_transforms2(
                target_img_rgb, target_landmarks[k], target_bboxes[k])

            # Face reenactment
            reenactment_input_tensor = []
            for j in range(len(curr_source_tensor)):
                curr_target_landmarks[j] = curr_target_landmarks[j].to(device)
                reenactment_input_tensor.append(torch.cat((curr_source_tensor[j], curr_target_landmarks[j]), dim=0).unsqueeze(0))
            reenactment_img_tensor, reenactment_seg_tensor = self.Gr(reenactment_input_tensor)

            # Segment target image
            target_img_tensor = curr_target_tensor[0].unsqueeze(0).to(device)
            target_seg_pred_tensor = self.Gs(target_img_tensor)
            target_mask_tensor = target_seg_pred_tensor.argmax(1) == 1

            # Remove the background of the aligned face
            aligned_face_mask_tensor = reenactment_seg_tensor.argmax(1) == 1  # face
            aligned_background_mask_tensor = ~aligned_face_mask_tensor
            aligned_img_no_background_tensor = reenactment_img_tensor.clone()
            aligned_img_no_background_tensor.masked_fill_(aligned_background_mask_tensor.unsqueeze(1), -1.0)

            # Complete face
            inpainting_input_tensor = torch.cat((aligned_img_no_background_tensor, target_mask_tensor.unsqueeze(1).float()), dim=1)
            inpainting_input_tensor_pyd = create_pyramid(inpainting_input_tensor, len(curr_target_tensor))
            completion_tensor = self.Gi(inpainting_input_tensor_pyd)

            # Blend faces
            transfer_tensor = transfer_mask(completion_tensor, target_img_tensor, target_mask_tensor)
            blend_input_tensor = torch.cat(
                (transfer_tensor, target_img_tensor, target_mask_tensor.unsqueeze(1).float()), dim=1)
            blend_input_tensor_pyd = create_pyramid(blend_input_tensor, len(curr_target_tensor))
            blend_tensor = self.Gb(blend_input_tensor_pyd)

            # Convert back to numpy images
            blend_img = tensor2rgb(blend_tensor)
            render_img = crop2img(target_img_rgb, blend_img, curr_target_bbox[0].numpy())
            outputs.append(render_img)

        return np.array(outputs)


if __name__ == "__main__":
    fsgan = FSGAN()
