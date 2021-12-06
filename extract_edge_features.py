import os
import glob
import json
import random
import traceback
import argparse
from pprint import pprint
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


def edge_features(
    frame,                  # The img to extract features from
    median_blur=5,          # If set (default 5), apply median blur with specified range 
    auto_threshold=True,    # If true (default), compute the median of single channel pixel intensities to use as thresholds for Canny edge detection
    sigma=0.33,             # If auto_threshold is true, the sigma to use
    lower=100,              # If auto_threshold is false, the bottom threshold for Canny edge detection
    upper=200,              # If auto_threshold is false, the top threshold for Canny edge detection 
    pool_size=10             # if set and greater than 1 (default 4), apply max pooling with specified size at the end to reduce dimensionality.
):
    frame = frame.numpy()

    if median_blur:
        frame = cv2.medianBlur(frame, 5)
    
    frame = to_grayscale(frame).astype('uint8')
    
    if auto_threshold:
        # Compute the median of the single channel pixel intensities
        v = np.median(frame)

        # Apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
    
    edges = cv2.Canny(frame, lower, upper)
    
    if pool_size == 1:
        return torch.tensor(edges)
    
    max_pool = torch.nn.MaxPool2d(pool_size, return_indices=False, ceil_mode=False)
    pooled = max_pool(torch.tensor(edges).float().unsqueeze(0))[0]
    return pooled.type(torch.uint8)

def color_histogram_features(frame, bins):
    (b, g, r) = cv2.split(frame.numpy())
    numPixels = np.prod(frame.shape[:2])
    histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
    histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
    histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
    return torch.tensor(np.concatenate([histogramR.flatten(), histogramG.flatten(), histogramB.flatten()]))


def extract_all_features(
    data_root,
    num_files=-1,
    first_frame_pool_size=4,
    num_samples=10,
    edge_detection_pool_size=10,
    edge_detection_auto_threshold=True,
    edge_detection_sigma=0.33,
    edge_detection_threshold_lower=100,
    edge_detection_threshold_upper=200,
    median_blur=5,
    histogram_num_bins=16,
    random_seed=15
):
    all_files = list(glob.glob(f'{data_root}/**/*.avi'))
    n = num_files if num_files > -1 else len(all_files)
    all_files = all_files[:n]

    print(f'Extracting from {len(all_files)} files')
    dirnames = sorted([d for d in os.listdir(data_root) if not d.startswith('.')])

    first_frame_feature_dim = (CLIP_WIDTH // first_frame_pool_size) * (CLIP_HEIGHT // first_frame_pool_size)
    edge_feature_dim = (CLIP_WIDTH // edge_detection_pool_size) * (CLIP_HEIGHT // edge_detection_pool_size)
    histogram_feature_dim = histogram_num_bins * 3
    feature_dim = first_frame_feature_dim + (num_samples * edge_feature_dim) + (num_samples * histogram_feature_dim)
    
    print(f'first_frame_feature_dim: {first_frame_feature_dim}')
    print(f'edge_feature_dim: {num_samples} frames * {edge_feature_dim} features per frame = {num_samples * edge_feature_dim}')
    print(f'histogram_feature_dim: {num_samples} frames * {histogram_feature_dim} features per frame= {num_samples * histogram_feature_dim}')
    print(f'total feature_dim: {feature_dim}')

    first_frame_features_result = torch.zeros(n, first_frame_feature_dim, dtype=torch.uint8)
    edge_features_result = torch.zeros(n, num_samples * edge_feature_dim, dtype=torch.uint8)
    histogram_features_result = torch.zeros(n, num_samples * histogram_feature_dim, dtype=torch.float32)

    labels = []
    fpaths = []
    invalids = set()

    for i, fpath in enumerate(all_files):
        try:
            avi_tensor, _, _ = torchvision.io.read_video(fpath)
            
            assert avi_tensor.shape[1] == 240 and avi_tensor.shape[2] == 320

            # Per-Video features
            first_frame_features_result[i] = maxpool(to_grayscale(avi_tensor[0].numpy())[:,:,0], first_frame_pool_size).type(torch.uint8).flatten()
            
            # Per-Frame features - from num_samples evenly sampled frames across the video
            sampler = torch.linspace(0, avi_tensor.shape[0] - 1, num_samples).trunc().int().tolist()
            for frame_iter_ind, frame_ind in enumerate(sampler): 
                edge_features_result[i,frame_iter_ind * edge_feature_dim:(frame_iter_ind + 1) * edge_feature_dim] = edge_features(
                    avi_tensor[frame_ind],
                    pool_size=edge_detection_pool_size.flatten(),
                    auto_threshold=edge_detection_auto_threshold,
                    sigma=edge_detection_sigma,
                    lower=edge_detection_threshold_lower,
                    upper=edge_detection_threshold_upper
                ).flatten()
                
                histogram_features_result[i,frame_iter_ind * histogram_feature_dim:(frame_iter_ind + 1) * histogram_feature_dim] = color_histogram_features(
                    avi_tensor[frame_ind],
                    bins=histogram_num_bins
                ).flatten()

            currdir = Path(fpath).parent.name
            labels.append(currdir)
            fpaths.append(fpath)

        except Exception:
            print(f'ERROR AT {i} - {fpath}')
            traceback.print_exc()
            invalids.add(fpath)
        finally:
            if i % 50 == 0:
                print(f'processed {i} out of {len(all_files)}')

    edge_features_result[edge_features_result == 255] = 1

    print(f'\nErrored on {len(invalids)} files: {invalids}')
    original_size = len(first_frame_features_result)
    nonempty = first_frame_features_result.sum(dim=-1) != 0
    first_frame_features_result = first_frame_features_result[nonempty]
    edge_features_result = edge_features_result[nonempty]
    histogram_features_result = histogram_features_result[nonempty]

    assert len(invalids) + len(first_frame_features_result) == original_size

    return [first_frame_features_result, edge_features_result, histogram_features_result], labels, fpaths

def features_to_df(all_features, labels, fpaths, random_seed):
    
    df = pd.concat([
        pd.DataFrame(features.numpy())
        for features in all_features
    ], axis=1)
    print(f'{len(df.columns)} features')

    assert len(df) == len(labels) and len(fpaths) == len(labels)

    df['label'] = df.index.map(lambda i: labels[i])
    df['fpath'] = df.index.map(lambda i: fpaths[i])

    df = df.sample(frac=1, random_state=random_seed)
    return df

def save_to_csvs(df, data_root, output_csv_path, splits):
    split_cols = []
    for split_path in splits:
        with open(split_path, 'r') as f:
            split_files = [f'{data_root}/{path.split()[0]}' for path in f.readlines()]
            split_col_name = split_path.split('.txt')[0]
            split_cols.append(split_col_name)
            df[split_col_name] = df['fpath'].isin(split_files)
            print(f'{df[split_col_name].sum()} files in split {split_col_name}')

    if len(split_cols) == 0:
        print('No splits given - saving to 1 csv')
        df.drop(['fpath'] + split_cols, axis=1).to_csv(output_csv_path, header=False, index=False)
    else:
        print(f'Saving to {split_cols} _features.csv')
        for split_col_name in split_cols:
            df[df[split_col_name]].drop(['fpath'] + split_cols, axis=1).to_csv(f'{split_col_name}_{output_csv_path}', header=False, index=False)
    
    return [f'{split_col_name}_{output_csv_path}' for split_col_name in split_cols]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(epilog='\nExample call:\n\n python extract_features.py /Users/apdoshi/Downloads/UCF-101 --output_path features.csv --splits trainlist01.txt testlist01.txt\n\n')
    parser.add_argument('data_root', help="<Requred> Path to the UCF101 folder containing all avi videos")
    parser.add_argument('-o', '--output_path', help="Path to the output csv to write features + labels to, will be prepended with splits (e.g. train_, test_) if splits are given", default='features.csv')
    parser.add_argument('-s','--splits', nargs='+', help='List of train/test split file paths', default=[])
    parser.add_argument('-n','--num_files', type=int, help='Number of files to featurize. If set to -1 (default), will use all files', default=-1)
    parser.add_argument('-ffps', '--first_frame_pool_size', type=int, help='Max pool size to use on the first frame', default=4)
    parser.add_argument('-edps', '--edge_detection_pool_size', type=int, help='Max pool size to use on the edge detected frames', default=10)
    parser.add_argument('-edt', '--edge_detection_threshold_type', type=str, help='Type of thresholding to use. Must be one of \'auto\' or \'manual\'', default='auto')
    parser.add_argument('-eds', '--edge_detection_sigma', type=float, help='Sigma to use when automatically thresholding Canny detector based on median', default=0.33)
    parser.add_argument('-edl', '--edge_detection_threshold_lower', type=int, help='When manually thresholding Canny detector, lower threshold value', default=100)
    parser.add_argument('-edu', '--edge_detection_threshold_upper', type=int, help='When manually thresholding Canny detector, upper threshold value', default=200)
    parser.add_argument('-edps', '--edge_detection_pool_size', type=int, help='Max pool size to use on the edge detected frames', default=10)
    parser.add_argument('-nf', '--num_samples', type=int, help='Number of frames to sample from video for edge detection and histograms', default=10)
    parser.add_argument('-blur', '--median_blur', type=int, help='Median blur size to use before applying edge detection', default=5)
    parser.add_argument('-bins', '--histogram_num_bins', type=int, help='Number of color histogram bins to use', default=16)
    parser.add_argument('-seed', '--random_seed', type=int, help='Random seed to use for shuffling', default=15)
    args = parser.parse_args()

    if not args.edge_detection_threshold_type in ['auto', 'manual']:
        parser.error('--edge_detection_threshold_type must be one of \'auto\' or \'manual\'')

    pprint(vars(args))
    print()

    features, labels, fpaths = extract_all_features(
        args.data_root,
        num_files=args.num_files,
        first_frame_pool_size=args.first_frame_pool_size,
        num_samples=args.num_samples,
        edge_detection_pool_size=args.edge_detection_pool_size,
        edge_detection_auto_threshold=args.edge_detection_threshold_type == 'auto',
        edge_detection_sigma=args.edge_detection_sigma,
        edge_detection_threshold_lower=args.edge_detection_threshold_lower,
        edge_detection_threshold_upper=args.edge_detection_threshold_upper,
        median_blur=args.median_blur,
        histogram_num_bins=args.histogram_num_bins,
        random_seed=args.random_seed
    )
    df = features_to_df(features, labels, fpaths, random_seed=args.random_seed)
    split_csvs = save_to_csvs(df, args.data_root, args.output_path, args.splits)
