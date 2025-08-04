# Criticality Decoding Accuracy Analysis

This script combines criticality measurements (d2) with decoding accuracy analysis using sliding windows, allowing both PCA and UMAP dimensionality reduction options.

## Overview

The `criticality_decoding_accuracy.m` script performs the following analysis:

1. **Data Loading**: Loads neural spike data and behavior labels from naturalistic recordings
2. **Sliding Window Analysis**: Uses the same sliding window approach as `criticality_compare.m`
3. **Criticality Measurement**: Calculates d2 (distance to criticality) for each window
4. **Decoding Analysis**: Performs SVM-based behavior decoding using PCA or UMAP dimensionality reduction
5. **Correlation Analysis**: Tests correlation between d2 and decoding accuracy
6. **Visualization**: Creates plots showing temporal dynamics and correlations

## Key Features

- **Dimensionality Reduction Options**: Supports both PCA and UMAP (UMAP requires MATLAB toolbox)
- **Flexible Window Analysis**: Uses optimal bin and window sizes determined automatically
- **Multiple Behavior Analysis**: Supports transition-based, within-behavior, and all-behavior decoding
- **Pre-trained SVM Models**: Trains SVM models on all data, then applies to sliding windows
- **Separate Bin Sizes**: Allows different bin sizes for criticality (d2) and SVM analysis
- **Loads Existing Results**: Can load d2 data from criticality_compare.m results
- **Comprehensive Output**: Saves results, creates plots, and provides correlation statistics

## Parameters

### Analysis Parameters
- `dimReductionMethod`: 'pca' or 'umap' (UMAP requires MATLAB toolbox)
- `nDim`: Number of dimensions to use (default: 6)
- `transOrWithin`: 'trans', 'within', or 'all' behavior analysis
- `stepSize`: Sliding window step size in seconds (default: 2)
- `kernelFunction`: SVM kernel function (default: 'polynomial')
- `svmBinSize`: Bin size for SVM analysis (default: 0.1 seconds)

### Criticality Parameters
- `pOrder`: AR model order for d2 calculation (default: 10)
- `critType`: Criticality type for d2 calculation (default: 2)

### Window Parameters
- `candidateFrameSizes`: Candidate bin sizes for optimization
- `candidateWindowSizes`: Candidate window sizes for optimization
- `minSpikesPerBin`: Minimum spikes per bin requirement
- `maxSpikesPerBin`: Maximum spikes per bin requirement
- `minBinsPerWindow`: Minimum bins per window requirement

## Output

### Results Structure
The script saves a results structure containing:
- `naturalistic.d2`: d2 values for each area and time window (loaded from criticality_compare.m or calculated)
- `naturalistic.decodingAccuracy`: Decoding accuracy for each area and time window
- `naturalistic.startS`: Time points for each measurement
- `naturalistic.corrResults`: Correlation analysis results
- `svmModels`: Pre-trained SVM models for each area
- `svmBinSize`: Bin size used for SVM analysis
- `params`: All analysis parameters

### Plots
1. **Temporal Dynamics**: d2 and decoding accuracy over time for each brain area
2. **Correlation Matrices**: Correlation between d2 and decoding accuracy
3. **Correlation Summary**: Bar plot of correlation coefficients across areas

### Files
- `criticality_decoding_results_[method].mat`: Results structure
- `criticality_decoding_[area]_[method].png`: Temporal dynamics plots
- `correlation_summary_[method].png`: Correlation summary plot

## Usage

1. **Basic Usage** (PCA):
   ```matlab
   % Run with default PCA settings
   criticality_decoding_accuracy
   ```

2. **UMAP Usage** (requires MATLAB toolbox):
   ```matlab
   % Change the method in the script
   dimReductionMethod = 'umap';
   criticality_decoding_accuracy
   ```

3. **Custom Parameters**:
   ```matlab
   % Modify parameters in the script
   nDim = 8;
   transOrWithin = 'all'; % Use all behavior labels
   stepSize = 5; % 5 second steps
   criticality_decoding_accuracy
   ```

## Dependencies

### Required Functions
- `myYuleWalker3.m`: AR model fitting
- `getFixedPointDistance2.m`: Distance to criticality calculation
- `find_optimal_bin_and_window.m`: Optimal parameter finding
- `neural_matrix_ms_to_frames.m`: Data binning
- `curate_behavior_labels.m`: Behavior data processing

### Optional Dependencies
- UMAP MATLAB toolbox (for UMAP dimensionality reduction)

## Testing

Run `test_criticality_decoding.m` to test basic functionality:
```matlab
test_criticality_decoding
```

## Comparison with Other Scripts

This script combines elements from:
- `criticality_compare.m`: Sliding window criticality analysis (loads d2 data from this script)
- `pca_svm_decode_behaviors.m`: PCA-based behavior decoding
- `umap_svm_decode_behaviors.m`: UMAP-based behavior decoding

The key innovations are:
1. **Pre-trained Models**: SVM models trained on all data, then applied to sliding windows
2. **Separate Bin Sizes**: Different bin sizes for criticality (d2) and SVM analysis
3. **Loads Existing Results**: Can load d2 data from criticality_compare.m results
4. **Window-based Decoding**: Uses the same temporal windows as criticality analysis

## Notes

- UMAP functionality requires the MATLAB UMAP toolbox
- The script automatically falls back to PCA if UMAP is not available
- Results are saved with the dimensionality reduction method in the filename
- Correlation analysis includes p-values and significance testing 