# Sliding Window Analysis Framework

This directory contains shared utilities for sliding window analyses across different analysis types (criticality, complexity, metastability, etc.).

**Note**: Data loading functions have been moved to `src/data_prep/` since they are common to all analyses, not just sliding window analyses.

## Directory Structure

```
src/
├── data_prep/                              # General data loading functions (shared across all analyses)
│   ├── load_naturalistic_data.m            # Spontaneous data loader
│   ├── load_reach_data.m                   # Reach task data loader
│   ├── load_schall_data.m                  # Schall data loader
│   ├── load_hong_data.m                    # Hong data loader
│   └── compute_lfp_binned_envelopes.m     # LFP envelope computation
│
└── sliding_window_prep/
    ├── data_prep/
    │   └── load_sliding_window_data.m      # Sliding window-specific data loading wrapper
    └── utils/
        ├── validate_workspace_vars.m       # Variable validation
        ├── create_results_path.m           # Results path/filename creation
        ├── setup_plotting.m                # Plotting configuration
        ├── calculate_window_indices.m       # Window index calculations
        └── plot_sliding_window_base.m      # Common plotting utilities
```

## Usage

### Loading Data

Instead of using `criticality_sliding_data_prep.m` which sets workspace variables, use:

```matlab
% Load data into a structure
dataStruct = load_sliding_window_data('reach', 'spikes', 'sessionName', 'Y15_27-Aug-2025 14_02_21_NeuroBeh.mat');

% Or for LFP
dataStruct = load_sliding_window_data('spontaneous', 'lfp', 'sessionName', 'ag/ag112321/recording1');
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

### Plotting Utilities

The `plot_sliding_window_base.m` module provides common plotting infrastructure:

```matlab
% Get plotting utilities
base = plot_sliding_window_base();

% Setup figure
figHandle = base.setup_figure(909, plotConfig.targetPos);

% Setup subplots (with tight_subplot fallback)
ha = base.setup_subplots(numRows, 1);
if isempty(ha)
    % Fallback to standard subplot
    for i = 1:numRows
        ha(i) = subplot(numRows, 1, i);
    end
end

% Get area colors
areaColors = base.get_area_colors();

% Configure axes
base.configure_axes(ha(1), 'xlabel', 'Time (s)', 'ylabel', 'd2', 'grid', true);

% Add event markers
base.add_reach_markers(ha(1), startS{a}, dataStruct.reachStart);
base.add_block2_marker(ha(1), dataStruct.startBlock2);

% Save figure
base.save_figure(gcf, config.saveDir, 'criticality_ar_win20', ...
    'filePrefix', plotConfig.filePrefix);
```

**Available Utilities:**
- `setup_figure()` - Create and configure figure with monitor detection
- `setup_subplots()` - Setup subplots with tight_subplot fallback
- `get_area_colors()` - Get standard area color definitions
- `configure_axes()` - Configure axes (tick labels, grid, limits, labels)
- `add_reach_markers()` - Add vertical lines at reach onsets
- `add_block2_marker()` - Add vertical line at block 2 start
- `add_response_markers()` - Add vertical lines at response onsets (schall data)
- `add_trial_type_line()` - Add trial type line (hong data)
- `save_figure()` - Save figure with standard naming and export settings

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

