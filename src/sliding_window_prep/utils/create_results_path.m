function resultsPath = create_results_path(analysisType, sessionType, sessionName, saveDir, varargin)
% CREATE_RESULTS_PATH Create standardized results file path
%
% Variables:
%   analysisType - Type of analysis: 'criticality_ar', 'criticality_av', 
%                  'criticality_lfp', 'complexity', etc.
%   sessionType - Data type: 'reach', 'spontaneous', 'schall', 'hong'
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
                    filename = sprintf('criticality_sliding_window_ar%s_%s.mat', ...
                        filenameSuffix, sessionName);
            
        case 'criticality_av'
                    filename = sprintf('criticality_sliding_window_av%s_%s.mat', ...
                        filenameSuffix, sessionName);
            
        case 'criticality_lfp'
                filename = sprintf('criticality_sliding_lfp%s_%s.mat', filenameSuffix, sessionName);
            
        case {'lzc'}
            if isempty(dataSource)
                error('dataSource must be provided for LZC analysis');
            end
            % Only add dataSource to filename if it's 'lfp' (not 'spikes')
            if strcmp(dataSource, 'lfp')
                filename = sprintf('lzc_sliding_window_%s%s_%s.mat', ...
                    dataSource, filenameSuffix, sessionName);
            else
                filename = sprintf('lzc_sliding_window%s_%s.mat', ...
                    filenameSuffix, sessionName);
            end
            
        case 'rqa'
            if isempty(dataSource)
                error('dataSource must be provided for rqa analysis');
            end
            % Only add dataSource to filename if it's 'lfp' (not 'spikes')
            % Note: PCA dimensions should be included in filenameSuffix by caller
            if strcmp(dataSource, 'lfp')
                filename = sprintf('rqa_sliding_window_%s%s_%s.mat', ...
                    dataSource, filenameSuffix, sessionName);
            else
                filename = sprintf('rqa_sliding_window%s_%s.mat', ...
                    filenameSuffix, sessionName);
            end
            
        otherwise
            % Generic fallback
            if ~isempty(sessionName)
                filename = sprintf('%s%s_%s.mat', analysisType, filenameSuffix, sessionName);
            else
                filename = sprintf('%s%s.mat', analysisType, filenameSuffix);
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

