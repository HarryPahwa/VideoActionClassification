import os
import glob
import json
import random
from pathlib import Path
import traceback

import cv2
import torch
import torchvision
import numpy as np
import pandas as pd

MEAN_CLIP_LENGTH = 7.21
MEAN_FRAMES = int(MEAN_CLIP_LENGTH * 25)
CLIP_WIDTH = 320
CLIP_HEIGHT = 240

def get_all_avi(dir_root, video_frames_only=True):
    data = {}
    for fpath in glob.glob(f'{dir_root}/**/*.avi'):
        video_frames, audio_frames, metadata = torchvision.io.read_video(fpath)
        if video_frames_only:
            yield video_frames
        else:
            yield (video_frames, audio_frames, metadata)

def to_grayscale(img):
     return np.expand_dims((0.3 * img[:,:,0]) + (0.59 * img[:,:,1]) + (0.11 * img[:,:,2]), -1)


def extract_features(
    avi_tensor,             # The PyTorch video tensor to extract features from
    split_type='sample',    # one of sample or trunc.
    to_length=MEAN_FRAMES,  # The length in frames to truncate the video to/pad (with 0s) to.
    auto_threshold=True,    # If true (default), compute the median of single channel pixel intensities to use as thresholds for Canny edge detection
    median_blur=5,          # If set (default 5), apply median blur with specified range 
    lower=100,              # If auto_threshold is false, the bottom threshold for Canny edge detection
    upper=200,              # If auto_threshold is false, the top threshold for Canny edge detection 
    pool_size=4             # if set (default 4), apply max pooling with specified size at the end to reduce dimensionality.
):
    def auto_canny(image, sigma=0.33):
        # Compute the median of the single channel pixel intensities
        v = np.median(image)

        # Apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(image, lower, upper)
    
    def extract_single(frame):
        if median_blur:
            frame = cv2.medianBlur(frame, 5)

        if auto_threshold:
            edges = auto_canny(to_grayscale(frame).astype('uint8'))
        else:
            edges = cv2.Canny(frame, lower, upper)
        if pool_size is None:
            return torch.tensor(edges)
        max_pool = torch.nn.MaxPool2d(pool_size, return_indices=False, ceil_mode=False)
        pooled = max_pool(torch.tensor(edges).float().unsqueeze(0))[0]
        return pooled

    result = torch.zeros(to_length, CLIP_HEIGHT // pool_size, CLIP_WIDTH // pool_size)
    if split_type == 'sample':
        sampler = torch.linspace(0, avi_tensor.shape[0] - 1, to_length).trunc().int().tolist()
        for i, frame_ind in enumerate(sampler):
            result[i] = extract_single(avi_tensor[frame_ind].numpy())
    else:
        for i in range(min(avi_tensor.shape[0], to_length)):
            result[i] = extract_single(avi_tensor[i].numpy())
    return result

def extract_to_csv(
    data_root,
    output_csv,
    train_list_path=None,
    test_list_path=None,
    to_length=MEAN_FRAMES,  # The length in frames to truncate the video to/pad (with 0s) to.
    num_samples=20,
    auto_threshold=True,    # If true (default), compute the median of single channel pixel intensities to use as thresholds for Canny edge detection
    median_blur=5,          # If set (default 5), apply median blur with specified range 
    lower=100,              # If auto_threshold is false, the bottom threshold for Canny edge detection
    upper=200,              # If auto_threshold is false, the top threshold for Canny edge detection 
    pool_size=4,
    random_seed=15
):
    all_files = list(glob.glob(f'{data_root}/**/*.avi'))
    print(f'Extracting from {len(all_files)} files')
    dirnames = sorted([d for d in os.listdir(data_root) if not d.startswith('.')])

    
    # dirnames.index(object)
    random.seed(random_seed)

    feature_map = {}
    feature_dim = (CLIP_WIDTH * CLIP_HEIGHT) + (to_length * (CLIP_WIDTH // pool_size) * (CLIP_HEIGHT // pool_size))
    result = torch.empty(len(all_files), feature_dim + 1, dtype=torch.uint8)
    invalids = set()
    for i, fpath in enumerate(all_files):
        try:
            currdir = Path(fpath).parent.name
            label = dirnames.index(currdir)
            avi_tensor, _, _ = torchvision.io.read_video(fpath)
            features = extract_features(avi_tensor, to_length=to_length, pool_size=pool_size)
            
            result[i,:(CLIP_WIDTH * CLIP_HEIGHT)] = to_grayscale(avi_tensor[0]).flatten().astype('uint8')
            result[i,(CLIP_WIDTH * CLIP_HEIGHT):-1] = features.flatten()
            result[i,-1] = label
            feature_map[i] = fpath

        except Exception:
            print(f'ERROR AT {i} - {fpath}')
            traceback.print_exc()
            invalids.add(fpath)
        finally:
            if i % 50 == 0:
                print(f'processed {i} out of {len(all_files)}')

    result[result == 255] = 1
    df = pd.DataFrame(result.numpy())
    df['fpath'] = df.index.map(lambda i: feature_map[i])
    df = df[~df['fpath'].isin(invalids)]
    df = df.sample(frac=1, random_state=random_seed)
    
    train_files, test_files = [], []
    if train_list_path is not None:
        with open(train_list_path, 'r') as f:
            train_files = [f'{data_root}/{path.split()[0]}' for path in f.readlines()]
    
    if test_list_path is not None:
        with open(test_list_path, 'r') as f:
            test_files = [f"{data_root}/{path.split()[0]}" for path in f.readlines()]

    df['is_train'] = df['fpath'].isin(train_files)
    df['is_test'] = df['fpath'].isin(test_files)

    print('Saving to csv...')

    if train_list_path is not None and test_list_path is not None:
        df[df['is_train']].drop(['fpath', 'is_train', 'is_test'], axis=1).to_csv('train_' + output_csv, header=False, index=False)
        df[df['is_test']].drop(['fpath', 'is_train', 'is_test'], axis=1).to_csv('test_' + output_csv, header=False, index=False)
    else:
        df.drop('fpath', axis=1).to_csv(output_csv, header=False, index=False)

    with open('feature_map.json', 'w+') as f:
        json.dump({
            'mapping': feature_map,
            'dirnames': dirnames,
            'invalids': list(invalids)
        }, f)

    # result = result[result.sum(dim = -1) != 0]
    

if __name__ == '__main__':
    data_root = '/Users/apdoshi/Downloads/UCF-101'
    train_list_path = '/Users/apdoshi/Downloads/ucfTrainTestlist/trainlist01.txt'
    test_list_path = '/Users/apdoshi/Downloads/ucfTrainTestlist/testlist01.txt'
    
    all_files = list(glob.glob(f'{data_root}/**/*.avi'))
    print(f'Extracting from {len(all_files)} files')
    dirnames = sorted([d for d in os.listdir(data_root) if not d.startswith('.')])

    # random_seed = 15
    # dirnames.index(object)
    # random.seed(random_seed)

    to_length=10
    pool_size=10
    output_csv='edges_restructured.csv'

    feature_map = {}
    feature_dim = (CLIP_WIDTH * CLIP_HEIGHT) + (to_length * (CLIP_WIDTH // pool_size) * (CLIP_HEIGHT // pool_size))
    result = torch.empty(len(all_files), feature_dim + 1, dtype=torch.uint8)
    invalids = set()
    for i, fpath in enumerate(all_files):
        try:
            currdir = Path(fpath).parent.name
            label = dirnames.index(currdir)
            avi_tensor, _, _ = torchvision.io.read_video(fpath)
            features = extract_features(avi_tensor, to_length=to_length, pool_size=pool_size)
            
            result[i,:(CLIP_WIDTH * CLIP_HEIGHT)] = torch.tensor(to_grayscale(avi_tensor[0]).flatten().astype('uint8'))
            result[i,(CLIP_WIDTH * CLIP_HEIGHT):-1] = features.flatten()
            result[i,-1] = label

        except Exception:
            print(f'ERROR AT {i} - {fpath}')
            traceback.print_exc()
            invalids.add(fpath)
        finally:
            if i % 50 == 0:
                print(f'processed {i} out of {len(all_files)}')
            feature_map[i] = fpath

    result[result == 255] = 1
    df = pd.DataFrame(result.numpy())
    df['fpath'] = df.index.map(lambda i: feature_map[i])
    df = df[~df['fpath'].isin(invalids)]
    df = df.sample(frac=1, random_state=15)
    
    train_files, test_files = [], []
    if train_list_path is not None:
        with open(train_list_path, 'r') as f:
            train_files = [f'{data_root}/{path.split()[0]}' for path in f.readlines()]
    
    if test_list_path is not None:
        with open(test_list_path, 'r') as f:
            test_files = [f"{data_root}/{path.split()[0]}" for path in f.readlines()]

    df['is_train'] = df['fpath'].isin(train_files)
    df['is_test'] = df['fpath'].isin(test_files)

    print('Saving to csv...')

    if train_list_path is not None and test_list_path is not None:
        df[df['is_train']].drop(['fpath', 'is_train', 'is_test'], axis=1).to_csv('train_' + output_csv, header=False, index=False)
        df[df['is_test']].drop(['fpath', 'is_train', 'is_test'], axis=1).to_csv('test_' + output_csv, header=False, index=False)
    else:
        df.drop('fpath', axis=1).to_csv(output_csv, header=False, index=False)

    with open('feature_map.json', 'w+') as f:
        json.dump({
            'mapping': feature_map,
            'dirnames': dirnames,
            'invalids': list(invalids)
        }, f)

    