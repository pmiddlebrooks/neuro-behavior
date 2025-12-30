function resultsPath = create_results_path(analysisType, dataType, slidingWindowSize, sessionName, saveDir, varargin)
% CREATE_RESULTS_PATH Create standardized results file path
%
% Variables:
%   analysisType - Type of analysis: 'criticality_ar', 'criticality_av', 
%                  'criticality_lfp', 'complexity', etc.
%   dataType - Data type: 'reach', 'naturalistic', 'schall', 'hong'
%   slidingWindowSize - Window size in seconds
%   sessionName - Session name (optional, can be empty)
%   saveDir - Base save directory
%   varargin - Optional name-value pairs:
%       'filenameSuffix' - Additional suffix (e.g., '_pca')
%       'dataSource' - 'spikes' or 'lfp' (for complexity analysis)
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
            if strcmp(dataType, 'reach')
                if ~isempty(sessionName)
                    filename = sprintf('criticality_sliding_window_ar%s_win%d_%s.mat', ...
                        filenameSuffix, slidingWindowSize, sessionName);
                else
                    error('sessionName must be provided for reach data criticality_ar analysis');
                end
            else
                filename = sprintf('criticality_sliding_window_ar%s_win%d.mat', ...
                    filenameSuffix, slidingWindowSize);
            end
            
        case 'criticality_av'
            if strcmp(dataType, 'reach')
                if ~isempty(sessionName)
                    filename = sprintf('criticality_sliding_window_av%s_win%d_%s.mat', ...
                        filenameSuffix, slidingWindowSize, sessionName);
                else
                    error('sessionName must be provided for reach data criticality_av analysis');
                end
            else
                filename = sprintf('criticality_sliding_window_av%s_win%d.mat', ...
                    filenameSuffix, slidingWindowSize);
            end
            
        case 'criticality_lfp'
            if ~isempty(sessionName)
                filename = sprintf('criticality_sliding_lfp_win%.1f_%s.mat', ...
                    slidingWindowSize, sessionName);
            else
                filename = sprintf('criticality_sliding_lfp_win%.1f.mat', slidingWindowSize);
            end
            
        case 'complexity'
            if ~isempty(dataSource)
                if strcmp(dataType, 'reach') && ~isempty(sessionName)
                    filename = sprintf('complexity_sliding_window_%s_win%.1f_%s.mat', ...
                        dataSource, slidingWindowSize, sessionName);
                elseif ~isempty(dataType)
                    filename = sprintf('complexity_sliding_window_%s_win%.1f_%s.mat', ...
                        dataSource, slidingWindowSize, dataType);
                else
                    filename = sprintf('complexity_sliding_window_%s_win%.1f.mat', ...
                        dataSource, slidingWindowSize);
                end
            else
                if strcmp(dataType, 'reach') && ~isempty(sessionName)
                    filename = sprintf('complexity_sliding_window_win%.1f_%s.mat', ...
                        slidingWindowSize, sessionName);
                else
                    filename = sprintf('complexity_sliding_window_win%.1f.mat', slidingWindowSize);
                end
            end
            
        case 'rqa'
            if ~isempty(dataSource)
                if strcmp(dataType, 'reach') && ~isempty(sessionName)
                    filename = sprintf('rqa_sliding_window_%s_win%.1f_%s.mat', ...
                        dataSource, slidingWindowSize, sessionName);
                elseif ~isempty(dataType)
                    filename = sprintf('rqa_sliding_window_%s_win%.1f_%s.mat', ...
                        dataSource, slidingWindowSize, dataType);
                else
                    filename = sprintf('rqa_sliding_window_%s_win%.1f.mat', ...
                        dataSource, slidingWindowSize);
                end
            else
                if strcmp(dataType, 'reach') && ~isempty(sessionName)
                    filename = sprintf('rqa_sliding_window_win%.1f_%s.mat', ...
                        slidingWindowSize, sessionName);
                else
                    filename = sprintf('rqa_sliding_window_win%.1f.mat', slidingWindowSize);
                end
            end
            
        otherwise
            % Generic fallback
            if ~isempty(sessionName)
                filename = sprintf('%s_win%.1f_%s.mat', analysisType, slidingWindowSize, sessionName);
            else
                filename = sprintf('%s_win%.1f.mat', analysisType, slidingWindowSize);
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

