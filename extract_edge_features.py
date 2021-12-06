import os
import glob
import json
import random
import traceback
import argparse
from pathlib import Path

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

def maxpool(img, size):
    max_pool = torch.nn.MaxPool2d(size, return_indices=False, ceil_mode=False)
    pooled = max_pool(torch.tensor(img).float().unsqueeze(0))[0]
    return pooled

def extract_edge_features(
    avi_tensor,             # The PyTorch video tensor to extract features from
    split_type='sample',    # one of sample or trunc.
    num_samples=MEAN_FRAMES,  # The length in frames to truncate the video to/pad (with 0s) to.
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

    result = torch.zeros(num_samples, CLIP_HEIGHT // pool_size, CLIP_WIDTH // pool_size)
    if split_type == 'sample':
        sampler = torch.linspace(0, avi_tensor.shape[0] - 1, num_samples).trunc().int().tolist()
        for i, frame_ind in enumerate(sampler):
            result[i] = extract_single(avi_tensor[frame_ind].numpy())
    else:
        for i in range(min(avi_tensor.shape[0], num_samples)):
            result[i] = extract_single(avi_tensor[i].numpy())
    
    return result

def average_histogram(frame, bins):
    (b, g, r) = cv2.split(frame)
    numPixels = np.prod(frame.shape[:2])
    histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
    histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
    histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
    return np.concatenate([histogramR.flatten(), histogramG.flatten(), histogramB.flatten()])


def extract_all_features(
    data_root,
    first_frame_pool_size=4,
    num_samples=10,
    edges_pool_size=10,
    median_blur=5,
    histogram_num_bins=16,
    random_seed=15
):
    all_files = list(glob.glob(f'{data_root}/**/*.avi'))
    print(f'Extracting from {len(all_files)} files')
    dirnames = sorted([d for d in os.listdir(data_root) if not d.startswith('.')])

    first_frame_feature_dim = (CLIP_WIDTH // first_frame_pool_size) * (CLIP_HEIGHT // first_frame_pool_size)
    edge_feature_dim = (num_samples * (CLIP_WIDTH // edges_pool_size) * (CLIP_HEIGHT // edges_pool_size))
    histogram_feature_dim = num_samples * histogram_num_bins * 3
    feature_dim = first_frame_feature_dim + edge_feature_dim + histogram_feature_dim 
    
    print(f'first_frame_feature_dim: {first_frame_feature_dim}')
    print(f'edge_feature_dim: {edge_feature_dim}')
    print(f'histogram_feature_dim: {histogram_feature_dim}')
    print(f'feature_dim: {feature_dim}')

    result = torch.empty(len(all_files), feature_dim, dtype=torch.float32)
    labels = []
    fpaths = []
    invalids = set()

    for i, fpath in enumerate(all_files):
        try:
            currdir = Path(fpath).parent.name
            
            avi_tensor, _, _ = torchvision.io.read_video(fpath)
            sampler = torch.linspace(0, avi_tensor.shape[0] - 1, num_samples).trunc().int().tolist()

            first_frame_features = maxpool(to_grayscale(avi_tensor[0].numpy())[:,:,0], first_frame_pool_size)
            edge_features = extract_edge_features(avi_tensor, num_samples=num_samples, pool_size=edges_pool_size)
            edge_features[edge_features == 255.] = 1.
            histogram_features = np.concatenate([average_histogram(avi_tensor[sample_ind].numpy(), histogram_num_bins).flatten() for sample_ind in sampler]).flatten()

            result[i,:first_frame_feature_dim] = first_frame_features.flatten()
            result[i,first_frame_feature_dim:first_frame_feature_dim + edge_feature_dim] = edge_features.flatten()
            result[i,first_frame_feature_dim + edge_feature_dim:] = torch.tensor(histogram_features.flatten())
            
            labels.append(currdir)
            fpaths.append(fpath)

        except Exception:
            print(f'ERROR AT {i} - {fpath}')
            traceback.print_exc()
            invalids.add(fpath)
        finally:
            if i % 50 == 0:
                print(f'processed {i} out of {len(all_files)}')


    result = result[result.sum(dim=-1) != 0]
    return result, labels, fpaths

def features_to_csvs(features, labels, fpaths, random_seed, data_root, train_list_path, test_list_path, output_csv):
    assert len(features) == len(labels) and len(fpaths) == len(labels)
    df = pd.DataFrame(features.numpy())

    df['label'] = df.index.map(lambda i: labels[i])
    df['fpath'] = df.index.map(lambda i: fpaths[i])
    
    # df = df[~df['fpath'].isin(invalids)]
    df = df.sample(frac=1, random_state=random_seed)
    
    train_files, test_files = None, None
    if train_list_path is not None:
        with open(train_list_path, 'r') as f:
            train_files = [f'{data_root}/{path.split()[0]}' for path in f.readlines()]
    
    if test_list_path is not None:
        with open(test_list_path, 'r') as f:
            test_files = [f"{data_root}/{path.split()[0]}" for path in f.readlines()]

    df['is_train'] = df['fpath'].isin(train_files)
    df['is_test'] = df['fpath'].isin(test_files)
    print(df['is_train'].sum())
    print(df['is_test'].sum())
    assert (df['is_train'] & df['is_test']).sum() == 0
    print(len(df))

    print('Saving to csvs...')

    # if train_list_path is not None and test_list_path is not None:
    df[df['is_train']].drop(['fpath', 'is_train', 'is_test'], axis=1).to_csv('train_' + output_csv, header=False, index=False)
    df[df['is_test']].drop(['fpath', 'is_train', 'is_test'], axis=1).to_csv('test_' + output_csv, header=False, index=False)

    with open('feature_map.json', 'w+') as f:
        json.dump({
            'train_labels': df[df['is_train']]['label'].tolist(),
            'train_fpaths': df[df['is_train']]['fpath'].tolist(),
            'test_labels': df[df['is_test']]['label'].tolist(),
            'test_fpaths': df[df['is_test']]['fpath'].tolist(),
            'dirnames': sorted([d for d in os.listdir(data_root) if not d.startswith('.')]),
            # 'invalids': list(invalids)
        }, f)

    return df

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('data_root', help="Root of the UCF101 folder containing all avi videos")
    # parser.add_argument('output_csv', help="Output csv to write features + labels to")
    # parser.add_argument('train_file_list', help="Output csv to write features + labels to")
    data_root = '/Users/apdoshi/Downloads/UCF-101'
    train_list_path = '/Users/apdoshi/Downloads/ucfTrainTestlist/trainlist01.txt'
    test_list_path = '/Users/apdoshi/Downloads/ucfTrainTestlist/testlist01.txt'
    output_csv = 'features.csv'

    features, labels, fpaths = extract_all_features(data_root)
    df = features_to_csvs(features, labels, fpaths, 15, data_root, train_list_path, test_list_path, output_csv)
    

if __name__ == '__main__':
    data_root = '/Users/apdoshi/Downloads/UCF-101'
    train_list_path = '/Users/apdoshi/Downloads/ucfTrainTestlist/trainlist01.txt'
    test_list_path = '/Users/apdoshi/Downloads/ucfTrainTestlist/testlist01.txt'
    output_csv = 'features.csv'

    features, labels, fpaths = extract_all_features(data_root)
    df = features_to_csvs(features, labels, fpaths, 15, data_root, train_list_path, test_list_path, output_csv)
    
