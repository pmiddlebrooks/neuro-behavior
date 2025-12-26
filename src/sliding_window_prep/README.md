# Sliding Window Analysis Framework

This directory contains shared utilities and data preparation functions for sliding window analyses across different analysis types (criticality, complexity, metastability, etc.).

## Directory Structure

```
sliding_window_prep/
├── data_prep/
│   ├── load_sliding_window_data.m          # Main data loading function
│   ├── load_naturalistic_data.m            # Naturalistic data loader
│   ├── load_reach_data.m                   # Reach task data loader
│   ├── load_schall_data.m                  # Schall data loader
│   ├── load_hong_data.m                    # Hong data loader
│   └── compute_lfp_binned_envelopes.m       # LFP envelope computation
│
└── utils/
    ├── validate_workspace_vars.m            # Variable validation
    ├── create_results_path.m               # Results path/filename creation
    ├── setup_plotting.m                    # Plotting configuration
    └── calculate_window_indices.m          # Window index calculations
```

## Usage

### Loading Data

Instead of using `criticality_sliding_data_prep.m` which sets workspace variables, use:

```matlab
% Load data into a structure
dataStruct = load_sliding_window_data('reach', 'spikes', 'sessionName', 'Y15_27-Aug-2025 14_02_21_NeuroBeh.mat');

% Or for LFP
dataStruct = load_sliding_window_data('naturalistic', 'lfp', 'sessionName', 'ag/ag112321/recording1');
```

### Running Analyses

Each analysis type has been converted to a function. For example, complexity analysis:

```matlab
% Set up configuration
config = struct();
config.slidingWindowSize = 20;
config.stepSize = 1;
config.binSize = 0.02;  % For spikes
config.nShuffles = 3;
config.makePlots = true;

% Run analysis
results = complexity_analysis(dataStruct, config);
```

### Backward Compatibility

The original scripts still work! They now use the new functions internally but maintain the same workspace variable interface.

## Benefits

1. **Reproducibility**: Functions with clear inputs/outputs instead of workspace variables
2. **Reduced Duplication**: Common code consolidated into utilities
3. **Easier Testing**: Functions can be tested independently
4. **Better Organization**: Shared code in one place, analysis-specific code in respective folders

## Migration Status

- ✅ Utility functions created
- ✅ Data loading function created
- ✅ Complexity analysis converted to function
- ⏳ Criticality analyses (in progress)
- ⏳ Wrapper scripts for backward compatibility

