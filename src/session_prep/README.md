# Session preparation utilities

Shared loading and utilities for **session-based** analyses (any temporal segmentation:
non-overlapping blocks, sliding windows, trials, etc.).

`sliding_window_prep/` remains unchanged for existing sliding-window pipelines.
New analyses should prefer this folder when segmentation is not inherently "sliding."

## Layout

```
session_prep/
├── data_prep/
│   └── load_session_data.m
└── utils/
    ├── bin_spikes.m
    ├── create_results_path.m
    ├── setup_plotting.m
    └── validate_workspace_vars.m
```

Session loaders delegate to `src/data_prep/` (`load_reach_data`, `load_spontaneous_data`, `load_interval_data`, etc.).

## Usage

```matlab
addpath('src/session_prep/data_prep');
addpath('src/session_prep/utils');
addpath('src/data_prep');

% Reach / schall: sessionName only
dataStruct = load_session_data('reach', 'spikes', ...
    'sessionName', sessionName, 'opts', opts);

% Spontaneous / interval: subjectName + sessionName
% dataStruct = load_session_data('interval', 'spikes', ...
%     'subjectName', 'ey9166', 'sessionName', 'ey9166_2026_04_03', 'opts', opts);
results = criticality_prg_analysis(dataStruct, config);
```
