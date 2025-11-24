function [hmm_results] = hmm_load_saved_model(natOrReach, varargin)
%HMM_LOAD_SAVED_MODEL Load saved HMM analysis results for use with new data
%
%   hmm_results = hmm_load_saved_model(natOrReach)
%   hmm_results = hmm_load_saved_model(natOrReach, 'brainArea', 'M56')
%   hmm_results = hmm_load_saved_model(natOrReach, 'brainArea', 'M56', 'binSize', 0.01, 'minDur', 0.04)
%
%   INPUTS:
%       natOrReach - String: 'Nat' for naturalistic data, 'Reach' for reach data
%       brainArea  - (Optional) String: 'M23', 'M56', 'DS', 'VS'. If not specified, returns all areas
%       fileIndex  - (Optional) Integer: Index of file to load (1 = most recent, 2 = second most recent, etc.)
%       binSize    - (Optional) Numeric: Bin size in seconds to filter files (e.g., 0.01)
%       minDur     - (Optional) Numeric: Minimum duration in seconds to filter files (e.g., 0.04)
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
%       % Load all areas from the most recent file
%       all_results = hmm_load_saved_model('Nat');
%
%       % Load specific area from most recent file
%       hmm_res = hmm_load_saved_model('Nat', 'brainArea', 'M56');
%
%       % Load specific area from file with specific binSize and minDur
%       hmm_res = hmm_load_saved_model('Nat', 'brainArea', 'M56', 'binSize', 0.01, 'minDur', 0.04);
%
%       % Access continuous results
%       state_sequence = hmm_res.continuous_results.sequence;
%       state_probabilities = hmm_res.continuous_results.pStates;
%
%   SEE ALSO:
%       hmm_mazz.m, hmm_mazz_plot.m, hmm_vs_behavior.m

% Parse optional inputs
p = inputParser;
addRequired(p, 'natOrReach', @ischar);
addParameter(p, 'brainArea', [], @(x) ischar(x) || isempty(x));
addParameter(p, 'fileIndex', 1, @isnumeric);
addParameter(p, 'binSize', [], @isnumeric);
addParameter(p, 'minDur', [], @isnumeric);
parse(p, natOrReach, varargin{:});

natOrReach = p.Results.natOrReach;
brainArea = p.Results.brainArea;
fileIndex = p.Results.fileIndex;
binSize = p.Results.binSize;
minDur = p.Results.minDur;

% Validate inputs
validDataTypes = {'Nat', 'Reach'};
validBrainAreas = {'M23', 'M56', 'DS', 'VS'};

if ~ismember(natOrReach, validDataTypes)
    error('Invalid data type. Must be one of: %s', strjoin(validDataTypes, ', '));
end

if ~isempty(brainArea) && ~ismember(brainArea, validBrainAreas)
    error('Invalid brain area. Must be one of: %s', strjoin(validBrainAreas, ', '));
end

if fileIndex < 1
    error('File index must be >= 1');
end

%% Load paths and setup
try
    paths = get_paths;
catch
    error('Could not load paths. Make sure get_paths() function is available.');
end

% Map brain area name to index (if brainArea is specified)
areaMap = containers.Map({'M23', 'M56', 'DS', 'VS'}, {1, 2, 3, 4});
if ~isempty(brainArea)
    if ~isKey(areaMap, brainArea)
        error('Invalid brain area: %s', brainArea);
    end
    areaIdx = areaMap(brainArea);
else
    areaIdx = [];
end

%% Find and load the HMM results file
fprintf('Looking for HMM results files...\n');
fprintf('Data type: %s\n', natOrReach);
if ~isempty(brainArea)
    fprintf('Brain area: %s\n', brainArea);
else
    fprintf('Loading all brain areas\n');
end

% If both binSize and minDur are specified, construct filename directly (like hmm_mazz_peri_nat.m)
if ~isempty(binSize) && ~isempty(minDur)
    if strcmpi(natOrReach, 'Reach')
        % For Reach data: need to search session folders
        reachResultsDir = paths.reachResultsPath;
        if ~exist(reachResultsDir, 'dir')
            error('Reach results directory not found: %s', reachResultsDir);
        end
        
        % Get all session folders
        sessionDirs = dir(fullfile(reachResultsDir, '*'));
        sessionDirs = sessionDirs([sessionDirs.isdir] & ~strncmp({sessionDirs.name}, '.', 1));
        
        % Construct filename
        filename = sprintf('hmm_mazz_reach_bin%.3f_minDur%.3f.mat', binSize, minDur);
        
        % Search for this file in all session folders
        filePath = [];
        for s = 1:length(sessionDirs)
            sessionPath = fullfile(reachResultsDir, sessionDirs(s).name);
            candidatePath = fullfile(sessionPath, filename);
            if exist(candidatePath, 'file')
                filePath = candidatePath;
                break;
            end
        end
        
        if isempty(filePath)
            error('File not found: %s\nSearched in all session folders under: %s', filename, reachResultsDir);
        end
        
        selectedFile = filename;
        
    else
        % For Naturalistic data: construct filename directly
        hmmdir = fullfile(paths.dropPath, 'metastability');
        if ~exist(hmmdir, 'dir')
            error('HMM directory not found: %s', hmmdir);
        end
        
        filename = sprintf('hmm_mazz_nat_bin%.3f_minDur%.3f.mat', binSize, minDur);
        filePath = fullfile(hmmdir, filename);
        
        if ~exist(filePath, 'file')
            error('File not found: %s\nMake sure hmm_mazz.m has been run for this dataset.', filePath);
        end
        
        selectedFile = filename;
    end
    
    fprintf('Loading file: %s\n', selectedFile);
    
else
    % Search for files when binSize/minDur not fully specified
    matchingFiles = {};
    filePaths = {};
    
    if strcmpi(natOrReach, 'Reach')
        % For Reach data: look in session-specific folders
        reachResultsDir = paths.reachResultsPath;
        if ~exist(reachResultsDir, 'dir')
            error('Reach results directory not found: %s', reachResultsDir);
        end
        
        % Get all session folders
        sessionDirs = dir(fullfile(reachResultsDir, '*'));
        sessionDirs = sessionDirs([sessionDirs.isdir] & ~strncmp({sessionDirs.name}, '.', 1));
        
        % Search for hmm_mazz_reach files in each session folder
        for s = 1:length(sessionDirs)
            sessionPath = fullfile(reachResultsDir, sessionDirs(s).name);
            files = dir(fullfile(sessionPath, 'hmm_mazz_reach_bin*.mat'));
            for f = 1:length(files)
                matchingFiles{end+1} = files(f).name;
                filePaths{end+1} = fullfile(sessionPath, files(f).name);
            end
        end
        
    else
        % For Naturalistic data: look in metastability folder
        hmmdir = fullfile(paths.dropPath, 'metastability');
        if ~exist(hmmdir, 'dir')
            error('HMM directory not found: %s', hmmdir);
        end
        
        % Search for hmm_mazz_nat_bin*.mat files
        files = dir(fullfile(hmmdir, 'hmm_mazz_nat_bin*.mat'));
        for f = 1:length(files)
            matchingFiles{end+1} = files(f).name;
            filePaths{end+1} = fullfile(hmmdir, files(f).name);
        end
    end
    
    if isempty(matchingFiles)
        error('No matching HMM results files found for %s data.\nCheck the results directories.', natOrReach);
    end
    
    % Filter files by binSize and minDur if partially specified
    if ~isempty(binSize) || ~isempty(minDur)
        filteredFiles = {};
        filteredPaths = {};
        
        % Tolerance for floating point comparison
        tol = 1e-5;
        
        for f = 1:length(matchingFiles)
            filename = matchingFiles{f};
            
            % Parse filename to extract binSize and minDur
            % Format: hmm_mazz_reach_bin0.010_minDur0.040.mat or hmm_mazz_nat_bin0.010_minDur0.040.mat
            binTokens = regexp(filename, 'bin([\d.]+)', 'tokens');
            minDurTokens = regexp(filename, 'minDur([\d.]+)', 'tokens');
            
            if ~isempty(binTokens) && ~isempty(minDurTokens)
                fileBinSize = str2double(binTokens{1}{1});
                fileMinDur = str2double(minDurTokens{1}{1});
                
                % Check if this file matches the requested parameters
                binMatches = isempty(binSize) || abs(fileBinSize - binSize) < tol;
                minDurMatches = isempty(minDur) || abs(fileMinDur - minDur) < tol;
                
                if binMatches && minDurMatches
                    filteredFiles{end+1} = filename;
                    filteredPaths{end+1} = filePaths{f};
                end
            end
        end
        
        if isempty(filteredFiles)
            if ~isempty(binSize) && ~isempty(minDur)
                error('No matching HMM results files found for %s data with binSize=%.3f and minDur=%.3f', ...
                    natOrReach, binSize, minDur);
            elseif ~isempty(binSize)
                error('No matching HMM results files found for %s data with binSize=%.3f', natOrReach, binSize);
            else
                error('No matching HMM results files found for %s data with minDur=%.3f', natOrReach, minDur);
            end
        end
        
        matchingFiles = filteredFiles;
        filePaths = filteredPaths;
    end
    
    % Sort files by date (most recent first)
    fileInfo = dir(filePaths{1});
    for i = 2:length(filePaths)
        fileInfo(i) = dir(filePaths{i});
    end
    [~, sortIdx] = sort([fileInfo.datenum], 'descend');
    matchingFiles = matchingFiles(sortIdx);
    filePaths = filePaths(sortIdx);
    
    % Check if requested file index exists
    if fileIndex > length(matchingFiles)
        error('File index %d exceeds number of available files (%d)', fileIndex, length(matchingFiles));
    end
    
    % Load the requested file
    selectedFile = matchingFiles{fileIndex};
    filePath = filePaths{fileIndex};
    
    fprintf('Loading file %d of %d: %s\n', fileIndex, length(matchingFiles), selectedFile);
end

% Load the data
try
    loadedData = load(filePath);
    
    % Both Reach and Naturalistic data now use 'results' struct
    if ~isfield(loadedData, 'results')
        error('File does not contain expected ''results'' structure');
    end
    
    % Extract results for requested brain area, or return all areas
    if ~isempty(brainArea)
        % Extract specific brain area
        if isempty(loadedData.results.hmm_results{areaIdx})
            error('No HMM results found for brain area %s in this file', brainArea);
        end
        hmm_results = loadedData.results.hmm_results{areaIdx};
        fprintf('Successfully loaded HMM results for %s area!\n', brainArea);
    else
        % Return entire results structure with all areas
        hmm_results = loadedData.results;
        fprintf('Successfully loaded HMM results for all areas!\n');
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

% Display status (only if single area was loaded)
if ~isempty(brainArea)
    if isfield(hmm_results, 'metadata') && isfield(hmm_results.metadata, 'analysis_status')
        if strcmp(hmm_results.metadata.analysis_status, 'SUCCESS')
            if isfield(hmm_results, 'best_model') && isfield(hmm_results.best_model, 'log_likelihood')
                fprintf('Log-likelihood: %.2f\n', hmm_results.best_model.log_likelihood);
            end
            fprintf('Model Status: SUCCESS\n');
        else
            fprintf('Model Status: FAILED\n');
        end
    end
    fprintf('\nModel ready for use with new data!\n');
else
    % Display summary for all areas
    fprintf('\nLoaded results for %d areas:\n', length(hmm_results.areas));
    for a = 1:length(hmm_results.areas)
        if ~isempty(hmm_results.hmm_results{a})
            fprintf('  %s: %d states\n', hmm_results.areas{a}, hmm_results.numStates(a));
        end
    end
    fprintf('\nAll areas ready for use!\n');
end

end 