# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Dict, Optional
import argparse
import glob
import torch, torchvision
import random
from collections import defaultdict
from pytorchvideo.transforms.augmentations import AugmentTransform
from pytorchvideo.transforms.transforms import OpSampler


# A dictionary that contains transform names (key) and their corresponding maximum
# transform magnitude (value).
_TRANSFORM_RANDAUG_MAX_PARAMS = {
    "AdjustBrightness": (1, 0.9),
    "AdjustContrast": (1, 0.9),
    "AdjustSaturation": (1, 0.9),
    "AdjustSharpness": (1, 0.9),
    "AutoContrast": None,
    "Equalize": None,
    "Invert": None,
    "Rotate": (0, 30),
    "Posterize": (4, 4),
    "Solarize": (1, 1),
    "ShearX": (0, 0.3),
    "ShearY": (0, 0.3),
    "TranslateX": (0, 0.25),
    "TranslateY": (0, 0.25),
}

# Hyperparameters for sampling magnitude.
# sampling_data_type determines whether uniform sampling samples among ints or floats.
# sampling_min determines the minimum possible value obtained from uniform
# sampling among floats.
# sampling_std determines the standard deviation for gaussian sampling.
SAMPLING_RANDAUG_DEFAULT_HPARAS = {
    "sampling_data_type": "int",
    "sampling_min": 0,
    "sampling_std": 0.5,
}


class RandAugment:
    """
    This implements RandAugment for video. Assume the input video tensor with shape
    (T, C, H, W).

    RandAugment: Practical automated data augmentation with a reduced search space
    (https://arxiv.org/abs/1909.13719)
    """

    def __init__(
        self,
        magnitude: int = 9,
        num_layers: int = 2,
        prob: float = 0.5,
        transform_hparas: Optional[Dict[str, Any]] = None,
        sampling_type: str = "gaussian",
        sampling_hparas: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        This implements RandAugment for video.

        Args:
            magnitude (int): Magnitude used for transform function.
            num_layers (int): How many transform functions to apply for each
                augmentation.
            prob (float): The probablity of applying each transform function.
            transform_hparas (Optional[Dict[Any]]): Transform hyper parameters.
                Needs to have key fill. By default, it uses transform_default_hparas.
            sampling_type (str): Sampling method for magnitude of transform. It should
                be either gaussian or uniform.
            sampling_hparas (Optional[Dict[Any]]): Hyper parameters for sampling. If
                gaussian sampling is used, it needs to have key sampling_std. By
                default, it uses SAMPLING_RANDAUG_DEFAULT_HPARAS.
        """
        assert sampling_type in ["gaussian", "uniform"]
        sampling_hparas = sampling_hparas or SAMPLING_RANDAUG_DEFAULT_HPARAS
        if sampling_type == "gaussian":
            assert "sampling_std" in sampling_hparas

        randaug_fn = [
            AugmentTransform(
                transform_name,
                magnitude,
                prob=prob,
                transform_max_paras=_TRANSFORM_RANDAUG_MAX_PARAMS,
                transform_hparas=transform_hparas,
                sampling_type=sampling_type,
                sampling_hparas=sampling_hparas,
            )
            for transform_name in list(_TRANSFORM_RANDAUG_MAX_PARAMS.keys())
        ]
        self.randaug_fn = OpSampler(randaug_fn, num_sample_op=num_layers)


    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Perform RandAugment to the input video tensor.

        Args:
            video (torch.Tensor): Input video tensor with shape (T, C, H, W).
        """
        return self.randaug_fn(video)


def augment_videos(data_root, category, vid_list, augmenter, vids_needed):
    for i in range(vids_needed):
        sample_video = random.choice(vid_list)
        vid_tensor, _, video_info = torchvision.io.read_video(sample_video)
        vid_tensor_shape = vid_tensor.shape
        vid_tensor = vid_tensor.reshape([vid_tensor_shape[0], vid_tensor_shape[-1], vid_tensor_shape[1], vid_tensor_shape[2]])
        augmented_tensor = augmenter(vid_tensor)

        augmented_file = f"{data_root}/{category}/v_{category}_augmented_{i:02d}.avi" 
        torchvision.io.write_video(augmented_file, augmented_tensor.reshape(vid_tensor_shape), fps=video_info['video_fps'])



def augment_all_classes(data_root, n):
    all_files = list(glob.glob(f'{data_root}/**/*.avi'))
    categories = defaultdict(list)
    for file in all_files:
        category = file.split("/")[-2]
        categories[category].append(file)
    augmenter = RandAugment()
    for category in categories.keys():
        if len(categories[category])<n:
            vids_needed = n-len(categories[category])
            print(f"Augmenting {vids_needed} videos for class {category}")
            augment_videos(data_root, category, categories[category], augmenter, vids_needed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(epilog='\nExample call:\n\n python random_video_aug.py /Users/apdoshi/Downloads/UCF-101 -n 100\n\n')
    parser.add_argument('--data_root', help="<Requred> Path to the UCF101 folder containing all avi videos")
    parser.add_argument('--num', type=int, help="<Requred> Number of minimum videos needed per category")
    args = parser.parse_args()
    if args.data_root[-1]=="/":
        argsdata_root = args.data_root[:-1]
    augment_all_classes(args.data_root, args.num)
