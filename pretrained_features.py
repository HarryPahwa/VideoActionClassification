import cv2
import torch
import torchvision
import numpy as np
import pandas as pd

from torchvision.transforms import Compose, Lambda, CenterCrop, Normalize

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)

def get_transform(model_name='x3d_s'):
    ####################
    # Slow transform
    ####################

    side_size = 182
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 182
    num_frames = 13
    sampling_rate = 6
    frames_per_second = 25

    # Note that this transform is specific to the x3d_s model.
    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size=(crop_size, crop_size))
            ]
        ),
    )
    return transform


def get_model(model_name='x3d_s'):
    model_name = 'x3d_s'
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
    model.eval()
    blocks = list(list(model.children())[0].children())
    model_without_head = torch.nn.Sequential(
        *blocks[:-1],
        list(blocks[-1].children())[0],
        torch.nn.AdaptiveAvgPool3d(output_size=1),
    #     list(blocks[-1].children())[1],
    #     list(blocks[-1].children())[3],
        torch.nn.Flatten()
    )
    return model_without_head

def extract(model, transform, video_paths):
    vid_datas = []
    for video_path in video_paths:
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(0, video.duration)

        # Apply a transform to normalize the video input
        video_data = transform(video_data)

        device = 'cpu'
        # Move the inputs to the desired device
        inputs = video_data["video"]
        inputs = inputs.to(device)
        vid_datas.append(inputs)

    # print(f'stacked {len(vid_datas)} vid datas')
    result = model(torch.stack(vid_datas))
    # print(result.shape)
    if len(video_paths) == 1 and result.shape[0] == 1:
        return result.flatten()
    return result

