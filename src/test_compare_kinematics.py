#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python companion for test_compare_kinematics.m
Reads a CSV (already sliced to first 20 data rows) and writes numpy outputs.
Usage: python test_compare_kinematics.py <sliced_csv> <out_dir>
"""

import sys
import os
import numpy as np
import pandas as pd

# Local import
from ComputeKinematics import adp_filt, get_features

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_compare_kinematics.py <sliced_csv> <out_dir>")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_dir = sys.argv[2]

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(2)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Python] Reading CSV: {csv_path}")
    data = pd.read_csv(csv_path)

    print("[Python] Running adp_filt...")
    currdf, _ = adp_filt(data)

    print("[Python] Running get_features...")
    features, scaled_features = get_features(currdf, 60)

    np.save(os.path.join(out_dir, 'python_features.npy'), features)
    np.save(os.path.join(out_dir, 'python_scaled_features.npy'), scaled_features)
    print(f"[Python] Saved outputs to {out_dir}")

if __name__ == "__main__":
    main()
