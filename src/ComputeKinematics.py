# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:02:35 2025

@author: adeneagle
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import math
from sklearn.preprocessing import StandardScaler
import os

def boxcar_center(a, n):
    a1 = pd.Series(a)
    moving_avg = np.array(a1.rolling(window=n, min_periods=1, center=True).mean())

    return moving_avg

def adp_filt(currdf: pd.DataFrame):
    lIndex = []
    xIndex = []
    yIndex = []
    currdf = np.array(currdf[1:])
    
    for header in range(currdf.shape[1] - 1):
        if currdf[0][header + 1] == "likelihood":
            lIndex.append(header)
        elif currdf[0][header + 1] == "x":
            xIndex.append(header)
        elif currdf[0][header + 1] == "y":
            yIndex.append(header)
    
    curr_df1 = currdf[:, 1:]
    
    datax = curr_df1[1:, np.array(xIndex)]
    datay = curr_df1[1:, np.array(yIndex)]
    data_lh = curr_df1[1:, np.array(lIndex)]
    
    currdf_filt = np.zeros((datax.shape[0], (datax.shape[1]) * 2))
    perc_rect = []
    
    for i in range(data_lh.shape[1]):
        perc_rect.append(0)
        
    for x in tqdm(range(data_lh.shape[1])):
        a, b = np.histogram(data_lh[1:, x].astype(float))
        rise_a = np.where(np.diff(a) >= 0)
        
        if rise_a[0][0] > 1:
            llh = b[rise_a[0][0]]
        else:
            llh = b[rise_a[0][1]]
        
        data_lh_float = data_lh[:, x].astype(float)
        perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        currdf_filt[0, (2 * x):(2 * x + 2)] = np.hstack([datax[0, x], datay[0, x]])
        
        for i in range(1, data_lh.shape[0]):
            if data_lh_float[i] < llh:
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[i, x], datay[i, x]])
                
    currdf_filt = np.array(currdf_filt)
    currdf_filt = currdf_filt.astype(float)
    
    return currdf_filt, perc_rect

def get_features_fast(data, fps=60):
    pass

def get_features(data, fps=60):
    """
    Extracts features based on (x,y) positions
    :param data: list, csv data
    :param fps: scalar, input for camera frame-rate
    :return f_10fps: 2D array, extracted features
    """
    window = int(np.round(0.05 / (1 / fps)) * 2 - 1)
    
    data_n_len = len(data)
    dxy_list = []
    disp_list = []
    
    for r in tqdm(range(data_n_len)):
        if r < data_n_len - 1:
            disp = []
            for c in range(0, data.shape[1], 2):
                disp.append(
                    np.linalg.norm(data[r + 1, c:c + 2] -
                                   data[r, c:c + 2]))
            disp_list.append(disp)
        dxy = []
        for i, j in itertools.combinations(range(0, data.shape[1], 2), 2):
            dxy.append(data[r, i:i + 2] -
                       data[r, j:j + 2])
        dxy_list.append(dxy)
    disp_r = np.array(disp_list)
    dxy_r = np.array(dxy_list)
    
    interp_times = np.arange(data_n_len)
    computed_times = interp_times[:-1] + 0.5
    
    disp_r = np.array([np.interp(interp_times, computed_times, disp_r[:, i])
                       for i in range(disp_r.shape[1])])
    
    print(dxy_r.shape, disp_r.shape)
    
    disp_boxcar = []
    dxy_eu = np.zeros([data_n_len, dxy_r.shape[1]])
    ang = np.zeros([data_n_len, dxy_r.shape[1]])
    dxy_boxcar = []
    ang_boxcar = []
    
    for l in tqdm(range(disp_r.shape[1])):
        disp_boxcar.append(boxcar_center(disp_r[:, l], window))
        
    for k in tqdm(range(dxy_r.shape[1])):
        for kk in range(data_n_len):
            dxy_eu[kk, k] = np.linalg.norm(dxy_r[kk, k, :])
            if kk < data_n_len - 1:
                b_3d = np.hstack([dxy_r[kk + 1, k, :], 0])
                a_3d = np.hstack([dxy_r[kk, k, :], 0])
                c = np.cross(b_3d, a_3d)
                ang[kk, k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                    math.atan2(np.linalg.norm(c),
                                               np.dot(dxy_r[kk, k, :], dxy_r[kk + 1, k, :])))
        dxy_boxcar.append(boxcar_center(dxy_eu[:, k], window))
        ang_boxcar.append(boxcar_center(ang[:, k], window))
    disp_feat = np.array(disp_boxcar)
    dxy_feat = np.array(dxy_boxcar)
    ang_feat = np.array(ang_boxcar)
    
    print(disp_feat.shape, dxy_feat.shape, ang_feat.shape)
    features = np.vstack((dxy_feat, ang_feat, disp_feat.T))
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.T).T
    
    return features, scaled_features

def save_video_features(path, save_dir):
    print("=== PYTHON SAVE_VIDEO_FEATURES DEBUG ===")
    print(f"Input path: {path}")
    print(f"Input save_dir: {save_dir}")
    
    # Check if input is a file path or data matrix
    if isinstance(path, str):
        # Input is a file path - load the data
        print("Step 1: Reading CSV data...")
        data = pd.read_csv(path)
        print(f"CSV data shape: {data.shape}")
        
        print("Step 2: Running adp_filt...")
        currdf, perc_rect = adp_filt(data)
        print(f"adp_filt output shape: {currdf.shape}")
        print(f"perc_rect: {perc_rect}")
    else:
        # Input is data matrix - use it directly
        print("Step 1: Using provided data matrix directly")
        print(f"Data shape: {path.shape}")
        currdf = path
        perc_rect = []  # Not available when using pre-processed data
    
    print("Step 3: Running get_features...")
    features, scaled_features = get_features(currdf, 60)
    print(f"get_features output shapes - features: {features.shape}, scaled_features: {scaled_features.shape}")
    
    if isinstance(path, str):
        file_name = os.path.basename(path).split(".")[0] + "_kinematics"
        output_path = os.path.join(save_dir, file_name)
        np.save(output_path, features)
        print(f"Features saved to: {output_path}")
    
    return features, scaled_features
    
def main():
    in_paths = [
        r"D:\neurobehavior\processed_data\videos\animal_ey4152\042822\2022-04-28_14-04-20DLC_resnet50_bottomup_clearSep21shuffle1_700000.csv",
        r"D:\neurobehavior\processed_data\videos\animal_ey4152\042922\2022-04-29_15-30-37DLC_resnet50_bottomup_clearSep21shuffle1_700000.csv"
        ]
    
    save_dirs = [
        r"C:\AdenCode\Data\Hsu\animal_ey4152\042822\2022-04-28_14-04-20",
        r"C:\AdenCode\Data\Hsu\animal_ey4152\042922\2022-04-29_15-30-37"
        ]
    
    for in_path, out_path in zip(in_paths, save_dirs):
        features, scaled_features = save_video_features(in_path, out_path)
    
        print(features.shape, scaled_features.shape)

if __name__ == "__main__":
    main()