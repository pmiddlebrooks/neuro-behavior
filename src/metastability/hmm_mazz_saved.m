%% Load Saved HMM Analysis Results
% This script loads previously saved HMM analysis results based on data type
% and brain area. It automatically finds the most recent file matching the criteria.

% Clear workspace and close figures
clear; clc; close all;

%% Parameters - Set these to load the desired data
% Choose data type: 'Nat' for naturalistic data, 'Reach' for reach data
natOrReach = 'Reach'; % 'Nat' or 'Reach'

% Choose brain area: 'M23', 'M56', 'DS', 'VS'
brainArea = 'M56'; % 'M23', 'M56', 'DS', 'VS'

%% Load paths and setup
paths = get_paths;
hmmdir = fullfile(paths.dropPath, 'hmm');

% Check if hmm directory exists
if ~exist(hmmdir, 'dir')
    error('HMM directory not found: %s', hmmdir);
end

%% Find and load the most recent HMM results file
fprintf('Looking for HMM results files...\n');
fprintf('Data type: %s\n', natOrReach);
fprintf('Brain area: %s\n', brainArea);

% Get all files in the hmm directory
files = dir(fullfile(hmmdir, '*.mat'));
fileNames = {files.name};

% Filter files that match our criteria
% Look for files with pattern: HMM_results_[DataType]_[BrainArea]_[Timestamp].mat
matchingFiles = {};
for i = 1:length(fileNames)
    fileName = fileNames{i};
    
    % Check if file matches our pattern
    if contains(fileName, 'HMM_results_') && ...
       contains(fileName, ['_' natOrReach '_']) && ...
       contains(fileName, ['_' brainArea '_']) && ...
       ~contains(fileName, 'FAILED') % Exclude failed analyses
        
        matchingFiles{end+1} = fileName;
    end
end

if isempty(matchingFiles)
    error('No matching HMM results files found for %s data in %s area.\nCheck the hmm directory: %s', ...
          natOrReach, brainArea, hmmdir);
end

% Sort files by date (most recent first)
filePaths = fullfile(hmmdir, matchingFiles);
fileInfo = dir(filePaths{1});
for i = 2:length(filePaths)
    fileInfo(i) = dir(filePaths{i});
end
[~, sortIdx] = sort([fileInfo.datenum], 'descend');
matchingFiles = matchingFiles(sortIdx);

% Load the most recent file
mostRecentFile = matchingFiles{1};
filePath = fullfile(hmmdir, mostRecentFile);

fprintf('Loading most recent file: %s\n', mostRecentFile);

% Load the data
try
    loadedData = load(filePath);
    
    % Check if the expected structure exists
    if isfield(loadedData, 'hmm_results_save')
        hmm_results = loadedData.hmm_results_save;
        fprintf('Successfully loaded HMM results!\n');
    else
        error('File does not contain expected hmm_results_save structure');
    end
    
catch ME
    error('Failed to load file %s: %s', mostRecentFile, ME.message);
end

%% Display loaded data information
fprintf('\n=== Loaded HMM Analysis Information ===\n');
fprintf('File: %s\n', mostRecentFile);
fprintf('Analysis Date: %s\n', hmm_results.metadata.analysis_date);
fprintf('Data Type: %s\n', hmm_results.metadata.data_type);
fprintf('Brain Area: %s\n', hmm_results.metadata.brain_area);
fprintf('Model Selection Method: %s\n', hmm_results.metadata.model_selection_method);

% Check if analysis was successful
if isfield(hmm_results.metadata, 'analysis_status')
    fprintf('Analysis Status: %s\n', hmm_results.metadata.analysis_status);
    if strcmp(hmm_results.metadata.analysis_status, 'FAILED')
        fprintf('Error Message: %s\n', hmm_results.metadata.error_message);
        fprintf('WARNING: This analysis failed. Results may be incomplete.\n');
    end
end

fprintf('\nData Parameters:\n');
fprintf('  Number of neurons: %d\n', hmm_results.data_params.num_neurons);
fprintf('  Number of trials: %d\n', hmm_results.data_params.num_trials);
fprintf('  Bin size: %.6f seconds\n', hmm_results.data_params.bin_size);
fprintf('  Frame size: %.6f seconds\n', hmm_results.data_params.frame_size);
fprintf('  Collection start: %.1f seconds\n', hmm_results.data_params.collect_start);
fprintf('  Collection duration: %.1f seconds\n', hmm_results.data_params.collect_duration);

fprintf('\nTrial Parameters:\n');
fprintf('  Trial duration: %.1f seconds\n', hmm_results.trial_params.trial_duration);

% Display reach-specific parameters if applicable
if strcmp(natOrReach, 'Reach') && isfield(hmm_results.trial_params, 'pre_time')
    fprintf('  Pre-reach time: %.1f seconds\n', hmm_results.trial_params.pre_time);
    fprintf('  Post-reach time: %.1f seconds\n', hmm_results.trial_params.post_time);
end

% Display HMM results if analysis was successful
if isfield(hmm_results, 'best_model') && ~strcmp(hmm_results.metadata.analysis_status, 'FAILED')
    fprintf('\nHMM Results:\n');
    fprintf('  Number of states: %d\n', hmm_results.best_model.num_states);
    fprintf('  Log-likelihood: %.2f\n', hmm_results.best_model.log_likelihood);
    fprintf('  Best state index: %d\n', hmm_results.best_model.best_state_index);
    
    % Display transition matrix dimensions
    [rows, cols] = size(hmm_results.best_model.transition_matrix);
    fprintf('  Transition matrix: %dx%d\n', rows, cols);
    
    % Display emission matrix dimensions
    [rows, cols] = size(hmm_results.best_model.emission_matrix);
    fprintf('  Emission matrix: %dx%d\n', rows, cols);
end

fprintf('\n=== Data Structure Available ===\n');
fprintf('The loaded data is available in the variable: hmm_results\n');
fprintf('Main fields:\n');
fprintf('  hmm_results.metadata - Analysis metadata\n');
fprintf('  hmm_results.data_params - Data parameters\n');
fprintf('  hmm_results.trial_params - Trial parameters\n');

if isfield(hmm_results, 'best_model')
    fprintf('  hmm_results.best_model - Best HMM model parameters\n');
end

if isfield(hmm_results, 'hmm_results')
    fprintf('  hmm_results.hmm_results - Complete HMM results structure\n');
end

if isfield(hmm_results, 'state_sequences')
    fprintf('  hmm_results.state_sequences - Detected state sequences\n');
end

if isfield(hmm_results, 'posterior_probabilities')
    fprintf('  hmm_results.posterior_probabilities - Posterior state probabilities\n');
end

fprintf('\nReady for further analysis!\n');

%% Optional: List all available files for this data type and brain area
fprintf('\n=== All Available Files for %s %s ===\n', natOrReach, brainArea);
for i = 1:length(matchingFiles)
    fileInfo = dir(fullfile(hmmdir, matchingFiles{i}));
    fprintf('%d. %s (Modified: %s)\n', i, matchingFiles{i}, ...
            datestr(fileInfo.datenum, 'yyyy-mm-dd HH:MM:SS'));
end

%% Example: Access loaded data
% You can now access the loaded data using:
% 
% % Get metadata
% dataType = hmm_results.metadata.data_type;
% brainArea = hmm_results.metadata.brain_area;
% 
% % Get data parameters
% numNeurons = hmm_results.data_params.num_neurons;
% numTrials = hmm_results.data_params.num_trials;
% 
% % Get HMM model (if analysis was successful)
% if isfield(hmm_results, 'best_model')
%     transitionMatrix = hmm_results.best_model.transition_matrix;
%     emissionMatrix = hmm_results.best_model.emission_matrix;
%     numStates = hmm_results.best_model.num_states;
% end
% 
% % Get state sequences (if analysis was successful)
% if isfield(hmm_results, 'state_sequences')
%     stateSequences = hmm_results.state_sequences;
% end
% 
% % Get posterior probabilities (if analysis was successful)
% if isfield(hmm_results, 'posterior_probabilities')
%     posteriorProbs = hmm_results.posterior_probabilities;
% end 