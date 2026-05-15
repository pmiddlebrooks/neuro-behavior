function plotConfig = setup_plotting(saveDir, varargin)
% SETUP_PLOTTING Common plotting configuration for session-based analyses
%
% Variables:
%   saveDir - Base directory for saved figures
%   varargin - 'sessionName', 'dataBaseName', 'figureNum', 'savePlots'
%
% Returns:
%   plotConfig - targetPos, filePrefix, figureNum, saveDir, savePlots

    p = inputParser;
    addParameter(p, 'sessionName', '', @ischar);
    addParameter(p, 'dataBaseName', '', @ischar);
    addParameter(p, 'figureNum', 900, @isnumeric);
    addParameter(p, 'savePlots', true, @islogical);
    parse(p, varargin{:});

    sessionName = p.Results.sessionName;
    dataBaseName = p.Results.dataBaseName;
    figureNum = p.Results.figureNum;
    savePlots = p.Results.savePlots;

    monitorPositions = get(0, 'MonitorPositions');
    if size(monitorPositions, 1) >= 2
        targetPos = monitorPositions(size(monitorPositions, 1), :);
    else
        targetPos = monitorPositions(1, :);
    end

    if ~isempty(sessionName)
        filePrefix = strrep(sessionName, filesep, '_');
        filePrefix = filePrefix(1:min(15, numel(filePrefix)));
    elseif ~isempty(dataBaseName)
        filePrefix = dataBaseName(1:min(15, numel(dataBaseName)));
    else
        [~, dirName] = fileparts(saveDir);
        filePrefix = dirName(1:min(15, numel(dirName)));
    end

    plotConfig = struct();
    plotConfig.targetPos = targetPos;
    plotConfig.filePrefix = filePrefix;
    plotConfig.figureNum = figureNum;
    plotConfig.saveDir = saveDir;
    plotConfig.savePlots = savePlots;
end
