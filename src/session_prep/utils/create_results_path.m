function resultsPath = create_results_path(analysisType, sessionType, sessionName, saveDir, varargin)
% CREATE_RESULTS_PATH Standardized results file path for session-based analyses
%
% Variables:
%   analysisType - e.g. 'criticality_prg'
%   sessionType - 'reach', 'spontaneous', 'schall', 'hong'
%   sessionName - Session label used in the filename
%   saveDir - Output directory
%   varargin - 'filenameSuffix', 'createDir' (default true)
%
% Returns:
%   resultsPath - Full path to the .mat results file

    p = inputParser;
    addParameter(p, 'filenameSuffix', '', @ischar);
    addParameter(p, 'createDir', true, @islogical);
    parse(p, varargin{:});

    filenameSuffix = p.Results.filenameSuffix;
    createDir = p.Results.createDir;

    if createDir && ~exist(saveDir, 'dir')
        [status, msg] = mkdir(saveDir);
        if ~status
            warning('Failed to create directory %s: %s', saveDir, msg);
        end
    end

    switch analysisType
        case 'criticality_prg'
            filename = sprintf('criticality_prg_blocks_%s.mat', sessionName);

        otherwise
            if ~isempty(sessionName)
                filename = sprintf('%s%s_%s.mat', analysisType, filenameSuffix, sessionName);
            else
                filename = sprintf('%s%s.mat', analysisType, filenameSuffix);
            end
    end

    resultsPath = fullfile(saveDir, filename);

    resultsDir = fileparts(resultsPath);
    if createDir && ~isempty(resultsDir) && ~exist(resultsDir, 'dir')
        [status, msg] = mkdir(resultsDir);
        if ~status
            warning('Failed to create results directory %s: %s', resultsDir, msg);
        end
    end
end
