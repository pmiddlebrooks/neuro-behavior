function [hmm_results] = hmm_load_saved_model(natOrReach, brainArea, varargin)
%HMM_LOAD_SAVED_MODEL Load saved HMM analysis results for use with new data
%
%   [hmm_model, hmm_params, metadata, continuous_results] = hmm_load_saved_model(natOrReach, brainArea)
%   [hmm_model, hmm_params, metadata, continuous_results] = hmm_load_saved_model(natOrReach, brainArea, 'fileIndex', idx)
%
%   INPUTS:
%       natOrReach - String: 'Nat' for naturalistic data, 'Reach' for reach data
%       brainArea  - String: 'M23', 'M56', 'DS', 'VS'
%       fileIndex  - (Optional) Integer: Index of file to load (1 = most recent, 2 = second most recent, etc.)
%
%   OUTPUTS (in hmm_results):
%       hmm_model  - Structure containing the trained HMM model:
%           .transition_matrix - KxK transition probability matrix
%           .emission_matrix   - Kx(N+1) emission probability matrix (N neurons + silence)
%           .num_states       - Number of hidden states K
%           .log_likelihood   - Log-likelihood of the trained model
%           .best_state_index - Index of the best state configuration
%
%       hmm_params - Structure containing HMM parameters and settings:
%           .bin_size         - Time bin size in seconds
%           .frame_size       - Frame size in seconds
%           .trial_duration   - Trial duration used for training
%           .num_neurons      - Number of neurons in the model
%           .model_selection_method - Method used for model selection
%           .hmm_parameters   - Original HMM parameters from training
%
%       metadata   - Structure containing analysis metadata:
%           .data_type        - 'Nat' or 'Reach'
%           .brain_area       - Brain area analyzed
%           .analysis_date    - Date of analysis
%           .analysis_status  - 'SUCCESS' or 'FAILED'
%           .file_path        - Path to the loaded file
%           .file_name        - Name of the loaded file
%
%       continuous_results - Structure containing continuous time series:
%           .sequence         - Continuous state sequence over time
%           .pStates         - Posterior state probabilities over time
%           .totalTime       - Total time in seconds
%
%   EXAMPLE:
%       % Load the most recent HMM model for M56 naturalistic data
%       [model, params, meta, cont_results] = hmm_load_saved_model('Nat', 'M56');
%
%       % Access continuous results
%       state_sequence = cont_results.sequence;
%       state_probabilities = cont_results.pStates;
%
%   SEE ALSO:
%       hmm_mazz.m, hmm_mazz_plot.m, hmm_vs_behavior.m

% Parse optional inputs
p = inputParser;
addRequired(p, 'natOrReach', @ischar);
addRequired(p, 'brainArea', @ischar);
addParameter(p, 'fileIndex', 1, @isnumeric);
parse(p, natOrReach, brainArea, varargin{:});

natOrReach = p.Results.natOrReach;
brainArea = p.Results.brainArea;
fileIndex = p.Results.fileIndex;

% Validate inputs
validDataTypes = {'Nat', 'Reach'};
validBrainAreas = {'M23', 'M56', 'DS', 'VS'};

if ~ismember(natOrReach, validDataTypes)
    error('Invalid data type. Must be one of: %s', strjoin(validDataTypes, ', '));
end

if ~ismember(brainArea, validBrainAreas)
    error('Invalid brain area. Must be one of: %s', strjoin(validBrainAreas, ', '));
end

if fileIndex < 1
    error('File index must be >= 1');
end

%% Load paths and setup
try
    paths = get_paths;
    hmmdir = fullfile(paths.dropPath, 'metastability');
catch
    error('Could not load paths. Make sure get_paths() function is available.');
end

% Check if hmm directory exists
if ~exist(hmmdir, 'dir')
    error('HMM directory not found: %s', hmmdir);
end

%% Find and load the HMM results file
fprintf('Looking for HMM results files...\n');
fprintf('Data type: %s\n', natOrReach);
fprintf('Brain area: %s\n', brainArea);

% Get all files in the hmm directory
files = dir(fullfile(hmmdir, '*.mat'));
fileNames = {files.name};

% Filter files that match our criteria
matchingFiles = {};
for i = 1:length(fileNames)
    fileName = fileNames{i};
    
    % Check if file matches our pattern
    if contains(fileName, 'HMM_results_') && ...
       contains(fileName, ['_' natOrReach '_']) && ...
       contains(fileName, ['_' brainArea]) && ...
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

% Check if requested file index exists
if fileIndex > length(matchingFiles)
    error('File index %d exceeds number of available files (%d)', fileIndex, length(matchingFiles));
end

% Load the requested file
selectedFile = matchingFiles{fileIndex};
filePath = fullfile(hmmdir, selectedFile);

fprintf('Loading file %d of %d: %s\n', fileIndex, length(matchingFiles), selectedFile);

% Load the data
try
    loadedData = load(filePath);
    
    % Check if the expected structure exists
    if isfield(loadedData, 'hmm_res')
        hmm_results = loadedData.hmm_res;
        fprintf('Successfully loaded HMM results!\n');
    else
        error('File does not contain expected hmm_res structure');
    end
    
catch ME
    error('Failed to load file %s: %s', selectedFile, ME.message);
end

% %% Extract and organize the outputs
% 
% % Check if analysis was successful
% if isfield(hmm_results.metadata, 'analysis_status') && ...
%    strcmp(hmm_results.metadata.analysis_status, 'FAILED')
%     warning('This analysis failed. Results may be incomplete.');
% end
% 
% % Extract HMM model
% hmm_model = struct();
% if isfield(hmm_results, 'best_model')
%     hmm_model.transition_matrix = hmm_results.best_model.transition_matrix;
%     hmm_model.emission_matrix = hmm_results.best_model.emission_matrix;
%     hmm_model.num_states = hmm_results.best_model.num_states;
%     hmm_model.log_likelihood = hmm_results.best_model.log_likelihood;
%     hmm_model.best_state_index = hmm_results.best_model.best_state_index;
% else
%     error('No HMM model found in the loaded data');
% end
% 
% % Extract HMM parameters
% hmm_params = struct();
% hmm_params.bin_size = hmm_results.data_params.bin_size;
% hmm_params.frame_size = hmm_results.data_params.frame_size;
% hmm_params.trial_duration = hmm_results.trial_params.trial_duration;
% hmm_params.num_neurons = hmm_results.data_params.num_neurons;
% hmm_params.model_selection_method = hmm_results.metadata.model_selection_method;
% 
% % Include original HMM parameters if available
% if isfield(hmm_results, 'hmm_parameters')
%     hmm_params.hmm_parameters = hmm_results.hmm_parameters;
% end
% 
% % Extract metadata
% metadata = struct();
% metadata.data_type = hmm_results.metadata.data_type;
% metadata.brain_area = hmm_results.metadata.brain_area;
% metadata.analysis_date = hmm_results.metadata.analysis_date;
% metadata.analysis_status = hmm_results.metadata.analysis_status;
% metadata.file_path = filePath;
% metadata.file_name = selectedFile;
% 
% % Include additional metadata if available
% if isfield(hmm_results.metadata, 'error_message')
%     metadata.error_message = hmm_results.metadata.error_message;
% end
% 
% %% Display summary information
% fprintf('\n=== Loaded HMM Model Summary ===\n');
% fprintf('File: %s\n', selectedFile);
% fprintf('Analysis Date: %s\n', metadata.analysis_date);
% fprintf('Data Type: %s\n', metadata.data_type);
% fprintf('Brain Area: %s\n', metadata.brain_area);
% fprintf('Number of States: %d\n', hmm_model.num_states);
% fprintf('Number of Neurons: %d\n', hmm_params.num_neurons);
% fprintf('Trial Duration: %.1f seconds\n', hmm_params.trial_duration);
% fprintf('Bin Size: %.6f seconds\n', hmm_params.bin_size);

if strcmp(hmm_results.metadata.analysis_status, 'SUCCESS')
    fprintf('Log-likelihood: %.2f\n', hmm_results.best_model.log_likelihood);
    fprintf('Model Status: SUCCESS\n');
else
    fprintf('Model Status: FAILED\n');
end

fprintf('\nModel ready for use with new data!\n');

end 