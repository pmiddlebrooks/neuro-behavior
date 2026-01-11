# Criticality Analysis Tools

This directory contains MATLAB tools for analyzing neural criticality, a theoretical framework that suggests neural systems operate near a phase transition between ordered and disordered states. These tools implement various methods to detect and quantify critical behavior in neural data.

## Overview

Criticality analysis examines whether neural systems operate at a critical point where they exhibit scale-free behavior, power-law distributions, and optimal information processing capabilities. The tools in this directory provide multiple approaches to assess criticality in neural spike data.

## Main Scripts

### `criticality_script.m`
**Primary analysis script for comprehensive criticality assessment**

- **Purpose**: Main script for analyzing neural criticality across different brain areas and conditions
- **Features**:
  - Compares spontaneous vs. task-based neural activity
  - Analyzes avalanche distributions across different brain areas (M23, M56, DS, VS)
  - Tests multiple bin sizes and thresholds for optimal avalanche detection
  - Implements PCA preprocessing options
  - Generates heatmaps showing avalanche counts and unique sizes across parameter spaces
- **Usage**: Run this script to perform comprehensive criticality analysis on your neural data

### `criticality_mr_estimator.m`
**Multistep Regression (MR) estimator for branching ratio analysis**

- **Purpose**: Implements the MR estimator method from Wilting and Priesemann (2018)
- **Features**:
  - Estimates branching ratio using multistep regression approach
  - Analyzes sliding windows across time series
  - Compares task blocks (correct vs. error trials)
  - Includes shuffled data controls
  - Supports both spontaneous and task-based data
- **Key Parameters**:
  - `kMax`: Maximum lag for regression (default: 10 seconds)
  - `windowSize`: Sliding window size (default: 45 seconds)
  - `stepSize`: Window step size (default: 2 seconds)

### `criticality_shew.m`
**Shew et al. (2009) criticality analysis using AR models**

- **Purpose**: Implements distance-to-criticality analysis using autoregressive models
- **Features**:
  - Fits AR models to neural activity time series
  - Calculates distance to criticality using fixed point analysis
  - Compares different experimental blocks and conditions
  - Includes shuffled data controls
  - Analyzes both spontaneous and task-based data
- **Key Parameters**:
  - `pOrder`: AR model order (default: 10)
  - `isiMult`: ISI multiplier for bin size (default: 10)
  - `critType`: Criticality type parameter (default: 2)

### 'Sequence for sliding window correlating criticality with neural/behavior'
- criticality_compare.m:
  - Run and save criticality metrics in a sliding window
  - Set a window size and step size for all future comparison analyses.
  - These parameters will be used for other anlyses
- criticality_decoding_accuracy.m:
  - Correlates decoding accuracy with d2 metric
  - Save results to be used for other analyses
- ../metastability/hmm_mazz.m
  - Fits an hmm to spiking data a la Mazzuato metastability
  - Uses its own bin sizes, but uses same window and step size.
- criticality_hmm_behavior_sliding_window.m:
  - Does all the correlations between criticality X hmm X behavior metrics


## Utility Functions

### Core Criticality Metrics

#### `compute_kappa.m`
**Calculates κ (kappa) criticality metric from Shew et al. (2009)**

```matlab
kappa = compute_kappa(avalancheSizes)
```
- **Input**: Vector of avalanche sizes
- **Output**: κ value (κ = 1 at criticality, < 1 subcritical, > 1 supercritical)
- **Method**: Compares empirical vs. theoretical cumulative distribution functions

#### `compute_DFA.m`
**Detrended Fluctuation Analysis for long-range temporal correlations**

```matlab
alpha = compute_DFA(signal, plotFlag)
```
- **Input**: 1D time series vector (e.g., LFP, summed spikes, kinematics).
- **Output**: DFA exponent α indicating long-range correlations
- **Interpretation**:
  - α ≈ 0.5: No long-range correlations (random)
  - 0.5 < α < 1.0: Long-range temporal correlations
  - α ≈ 1.5: Brownian noise
  - α > 1.5: Highly persistent signals

#### `distance_to_criticality.m`
**Calculates Distance to Criticality Coefficient from Ma et al. (2019)**

```matlab
dcc = distance_to_criticality(tauFitted, alphaFitted, gammaFitted)
```
- **Input**: Fitted exponents (τ, α, γ) from avalanche analysis
- **Output**: Distance to criticality coefficient
- **Method**: Compares predicted vs. observed γ values

### Branching Ratio Methods

#### `stepwise_branching_ratio.m`
**Computes stepwise branching ratio from activity vector**

```matlab
br = stepwise_branching_ratio(activityVec)
```
- **Input**: 1D activity vector with zeros separating avalanches
- **Output**: Average of all within-avalanche A(t+1)/A(t) ratios
- **Method**: Calculates stepwise ratios within each avalanche

#### `weighted_branching_ratio.m`
**Computes weighted branching ratio from activity vector**

```matlab
br = weighted_branching_ratio(activityVec)
```
- **Input**: 1D activity vector with zeros separating avalanches
- **Output**: Weighted global branching ratio
- **Method**: Global weighted average across all avalanches

#### `branching_ratio_mr_estimation.m`
**MR estimator for branching ratio using multistep regression**

```matlab
result = branching_ratio_mr_estimation(data, maxSlopes)
```
- **Input**: Time series vector, maximum lag (default: 40)
- **Output**: Struct with branching ratio, autocorrelation time, and fit parameters
- **Method**: Fits exponential model to regression slopes across different lags

### Data Quality Assessment

#### `avalanches_bin_sufficiency.m`
**Assesses whether dataset has sufficient bins for reliable criticality analysis**

```matlab
sufficient = avalanches_bin_sufficiency(dataMat)
```
- **Input**: Binary spike matrix
- **Output**: Boolean indicating sufficient data quality
- **Criteria**:
  - Minimum 10,000 avalanches
  - Minimum 50 unique avalanche sizes
  - Power-law exponent in range 1.2-3.0
- **Features**: Provides detailed feedback on data quality and recommendations

## Usage Examples

### Basic Criticality Analysis
```matlab
% Load and prepare your neural data
opts = neuro_behavior_options;
get_standard_data;

% Run comprehensive analysis
criticality_script;

% Or run specific methods
result = branching_ratio_mr_estimation(popActivity, 40);
kappa = compute_kappa(avalancheSizes);
alpha = compute_DFA(dataMat, true);
```

### Data Quality Check
```matlab
% Check if your data is sufficient for criticality analysis
sufficient = avalanches_bin_sufficiency(dataMat);
if sufficient
    fprintf('Data quality is sufficient for criticality analysis\n');
else
    fprintf('Consider collecting more data or adjusting parameters\n');
end
```

## Key Concepts

### Criticality Indicators
1. **Power-law distributions** in avalanche sizes and durations
2. **Branching ratio ≈ 1** indicating critical dynamics
3. **Long-range temporal correlations** (DFA α > 0.5)
4. **Optimal information processing** at critical point

### Data Requirements
- **Minimum data**: 10,000+ avalanches, 50+ unique sizes
- **Temporal resolution**: At least 10× mean ISI
- **Spatial coverage**: Multiple neurons/areas for robust analysis
- **Duration**: Sufficient time for stable statistics

### Parameter Selection
- **Bin size**: Balance temporal resolution with statistical power
- **Threshold**: Often median-based or percentile-based
- **Model order**: AR model order (typically 5-15)
- **Window size**: Balance temporal stability with resolution

## References

1. Shew, W. L., et al. (2009). Neuronal avalanches imply maximum dynamic range in cortical networks at criticality. *Journal of Neuroscience*.
2. Wilting, J., & Priesemann, V. (2018). 25 years of criticality in neuroscience. *Nature Reviews Neuroscience*.
3. Ma, Z., et al. (2019). Criticality or supersolidity in neural networks. *Physical Review E*.

## Dependencies

These tools depend on the main neuro-behavior analysis framework:
- `neuro_behavior_options.m`
- `get_paths.m`
- `get_standard_data.m`
- `neural_matrix_mark_data.m`
- `neural_matrix_ms_to_frames.m`

Additional dependencies may include:
- `plfit.m` for power-law fitting (Clauset's method)
- `myYuleWalker3.m` for AR model fitting
- `getFixedPointDistance2.m` for distance calculations 