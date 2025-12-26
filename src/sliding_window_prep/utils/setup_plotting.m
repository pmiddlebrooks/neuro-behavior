function plotConfig = setup_plotting(saveDir, varargin)
% SETUP_PLOTTING Setup common plotting configuration
%
% Variables:
%   saveDir - Save directory (used to extract file prefix)
%   varargin - Optional name-value pairs:
%       'sessionName' - Session name (takes priority for file prefix)
%       'dataBaseName' - Data base name (fallback for file prefix)
%       'figureNum' - Default figure number (default: 900)
%
% Goal:
%   Extract common plotting configuration: monitor positions, file prefix,
%   target figure position. Used across all sliding window analysis plots.
%
% Returns:
%   plotConfig - Structure with fields:
%       .targetPos - Target monitor position [x, y, width, height]
%       .filePrefix - File prefix for saving plots
%       .figureNum - Default figure number

    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'sessionName', '', @ischar);
    addParameter(p, 'dataBaseName', '', @ischar);
    addParameter(p, 'figureNum', 900, @isnumeric);
    parse(p, varargin{:});
    
    sessionName = p.Results.sessionName;
    dataBaseName = p.Results.dataBaseName;
    figureNum = p.Results.figureNum;
    
    % Detect monitors and set target position
    monitorPositions = get(0, 'MonitorPositions');
    if size(monitorPositions, 1) >= 2
        targetPos = monitorPositions(size(monitorPositions, 1), :);
    else
        targetPos = monitorPositions(1, :);
    end
    
    % Extract filename prefix (priority: sessionName > dataBaseName > saveDir)
    if ~isempty(sessionName)
        filePrefix = sessionName(1:min(8, length(sessionName)));
    elseif ~isempty(dataBaseName)
        filePrefix = dataBaseName(1:min(8, length(dataBaseName)));
    elseif ~isempty(saveDir)
        [~, dirName, ~] = fileparts(saveDir);
        filePrefix = dirName(1:min(8, length(dirName)));
    else
        filePrefix = '';
    end
    
    % Create output structure
    plotConfig = struct();
    plotConfig.targetPos = targetPos;
    plotConfig.filePrefix = filePrefix;
    plotConfig.figureNum = figureNum;
end

