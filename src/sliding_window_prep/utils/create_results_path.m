function resultsPath = create_results_path(analysisType, sessionType, sessionName, saveDir, varargin)
% CREATE_RESULTS_PATH Create standardized results file path
%
% Variables:
%   analysisType - Type of analysis: 'criticality_ar', 'criticality_av', 
%                  'criticality_lfp', 'complexity', etc.
%   sessionType - Data type: 'reach', 'naturalistic', 'schall', 'hong'
%   sessionName - Session name (optional, can be empty)
%   saveDir - Base save directory
%   varargin - Optional name-value pairs:
%       'filenameSuffix' - Additional suffix (e.g., '_pca')
%       'dataSource' - 'spikes' or 'lfp' (required for complexity and rqa analyses)
%       'createDir' - Create directory if it doesn't exist (default: true)
%
% Goal:
%   Create standardized file paths for saving analysis results. Handles
%   different naming conventions for different analysis types and data types.
%
% Returns:
%   resultsPath - Full path to results file

    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'filenameSuffix', '', @ischar);
    addParameter(p, 'dataSource', '', @ischar);
    addParameter(p, 'createDir', true, @islogical);
    parse(p, varargin{:});
    
    filenameSuffix = p.Results.filenameSuffix;
    dataSource = p.Results.dataSource;
    createDir = p.Results.createDir;
    
    % Create directory if needed (including any subdirectories)
    if createDir && ~exist(saveDir, 'dir')
        [status, msg] = mkdir(saveDir);
        if ~status
            warning('Failed to create directory %s: %s', saveDir, msg);
        end
    end
    
    % Build filename based on analysis type
    switch analysisType
        case 'criticality_ar'
            if strcmp(sessionType, 'reach')
                if ~isempty(sessionName)
                    filename = sprintf('criticality_sliding_window_ar%s_%s.mat', ...
                        filenameSuffix, sessionName);
                else
                    error('sessionName must be provided for reach data criticality_ar analysis');
                end
            else
                filename = sprintf('criticality_sliding_window_ar%s.mat', filenameSuffix);
            end
            
        case 'criticality_av'
            if strcmp(sessionType, 'reach')
                if ~isempty(sessionName)
                    filename = sprintf('criticality_sliding_window_av%s_%s.mat', ...
                        filenameSuffix, sessionName);
                else
                    error('sessionName must be provided for reach data criticality_av analysis');
                end
            else
                filename = sprintf('criticality_sliding_window_av%s.mat', filenameSuffix);
            end
            
        case 'criticality_lfp'
            if ~isempty(sessionName)
                filename = sprintf('criticality_sliding_lfp_%s.mat', sessionName);
            else
                filename = sprintf('criticality_sliding_lfp.mat');
            end
            
        case {'complexity', 'lzc'}
            if isempty(dataSource)
                error('dataSource must be provided for LZC analysis');
            end
            if strcmp(sessionType, 'reach') && ~isempty(sessionName)
                filename = sprintf('lzc_sliding_window_%s_%s.mat', ...
                    dataSource, sessionName);
            elseif ~isempty(sessionType)
                filename = sprintf('lzc_sliding_window_%s_%s.mat', ...
                    dataSource, sessionType);
            else
                filename = sprintf('lzc_sliding_window_%s.mat', dataSource);
            end
            
        case 'rqa'
            if isempty(dataSource)
                error('dataSource must be provided for rqa analysis');
            end
            if strcmp(sessionType, 'reach') && ~isempty(sessionName)
                filename = sprintf('rqa_sliding_window_%s_%s.mat', ...
                    dataSource, sessionName);
            elseif ~isempty(sessionType)
                filename = sprintf('rqa_sliding_window_%s_%s.mat', ...
                    dataSource, sessionType);
            else
                filename = sprintf('rqa_sliding_window_%s.mat', dataSource);
            end
            
        otherwise
            % Generic fallback
            if ~isempty(sessionName)
                filename = sprintf('%s_%s.mat', analysisType, sessionName);
            else
                filename = sprintf('%s.mat', analysisType);
            end
    end
    
    resultsPath = fullfile(saveDir, filename);
    
    % Ensure the full directory path exists (in case filename includes subdirectories)
    resultsDir = fileparts(resultsPath);
    if createDir && ~isempty(resultsDir) && ~exist(resultsDir, 'dir')
        [status, msg] = mkdir(resultsDir);
        if ~status
            warning('Failed to create results directory %s: %s', resultsDir, msg);
        end
    end
end

