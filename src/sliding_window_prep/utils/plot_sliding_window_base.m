function base = plot_sliding_window_base()
% PLOT_SLIDING_WINDOW_BASE Base utilities for sliding window plotting
%
% Variables:
%   None (returns structure with utility functions)
%
% Goal:
%   Provides common plotting infrastructure for all sliding window analyses:
%   - Figure setup (monitor detection, positioning, tight_subplot)
%   - Area colors
%   - Event markers (reach onsets, block 2, response onsets, trial types)
%   - Axis configuration
%   - Figure saving
%
% Returns:
%   base - Structure containing utility functions:
%       .setup_figure() - Setup figure with monitor detection
%       .get_area_colors() - Get standard area color definitions
%       .setup_subplots() - Setup subplots with tight_subplot fallback
%       .configure_axes() - Configure axes (tick labels, grid, etc.)
%       .add_reach_markers() - Add reach onset markers
%       .add_block2_marker() - Add block 2 marker
%       .add_response_markers() - Add response onset markers (schall)
%       .add_trial_type_line() - Add trial type line (hong)
%       .save_figure() - Save figure with standard naming

    base = struct();
    
    % ========== Figure Setup ==========
    base.setup_figure = @setup_figure;
    base.setup_subplots = @setup_subplots;
    base.get_area_colors = @get_area_colors;
    base.configure_axes = @configure_axes;
    
    % ========== Event Markers ==========
    base.add_reach_markers = @add_reach_markers;
    base.add_block2_marker = @add_block2_marker;
    base.add_response_markers = @add_response_markers;
    base.add_trial_type_line = @add_trial_type_line;
    
    % ========== Figure Saving ==========
    base.save_figure = @save_figure;
end

% ========== Figure Setup Functions ==========

function figHandle = setup_figure(figureNum, targetPos)
% SETUP_FIGURE Create and configure figure
%
% Variables:
%   figureNum - Figure number
%   targetPos - Target position [x, y, width, height]
%
% Returns:
%   figHandle - Figure handle

    figHandle = figure(figureNum);
    clf(figHandle);
    set(figHandle, 'Units', 'pixels');
    set(figHandle, 'Position', targetPos);
end

function ha = setup_subplots(numRows, numCols, varargin)
% SETUP_SUBPLOTS Setup subplots with tight_subplot fallback
%
% Variables:
%   numRows - Number of rows
%   numCols - Number of columns
%   varargin - Optional spacing parameters for tight_subplot:
%       [gapH, gapW] - Horizontal and vertical gaps (default: [0.035 0.04])
%       [lowMargin, highMargin] - Bottom and top margins (default: [0.03 0.08])
%       [leftMargin, rightMargin] - Left and right margins (default: [0.08 0.04])
%
% Returns:
%   ha - Array of axes handles (or empty if using standard subplot)

    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'gapH', 0.035, @isnumeric);
    addParameter(p, 'gapW', 0.04, @isnumeric);
    addParameter(p, 'lowMargin', 0.03, @isnumeric);
    addParameter(p, 'highMargin', 0.08, @isnumeric);
    addParameter(p, 'leftMargin', 0.08, @isnumeric);
    addParameter(p, 'rightMargin', 0.04, @isnumeric);
    parse(p, varargin{:});
    
    % Check if tight_subplot is available
    useTightSubplot = exist('tight_subplot', 'file');
    
    if useTightSubplot
        ha = tight_subplot(numRows, numCols, ...
            [p.Results.gapH, p.Results.gapW], ...
            [p.Results.lowMargin, p.Results.highMargin], ...
            [p.Results.leftMargin, p.Results.rightMargin]);
    else
        % Return empty - caller should use standard subplot
        ha = [];
    end
end

function areaColors = get_area_colors()
% GET_AREA_COLORS Get standard area color definitions
%
% Returns:
%   areaColors - Cell array of RGB colors for M23, M56, DS, VS

    areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1]};  % Red, Green, Blue, Magenta
end

function configure_axes(ax, varargin)
% CONFIGURE_AXES Configure axes with standard settings
%
% Variables:
%   ax - Axes handle (optional, defaults to gca)
%   varargin - Optional name-value pairs:
%       'xlabel' - X-axis label
%       'ylabel' - Y-axis label
%       'title' - Plot title
%       'xlim' - X-axis limits [min, max]
%       'ylim' - Y-axis limits [min, max]
%       'grid' - Show grid (default: true)
%       'hold' - Hold on (default: false)

    if nargin < 1 || isempty(ax)
        ax = gca;
    end
    
    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'xlabel', '', @ischar);
    addParameter(p, 'ylabel', '', @ischar);
    addParameter(p, 'title', '', @ischar);
    addParameter(p, 'xlim', [], @(x) isnumeric(x) && length(x) == 2);
    addParameter(p, 'ylim', [], @(x) isnumeric(x) && length(x) == 2);
    addParameter(p, 'grid', true, @islogical);
    addParameter(p, 'hold', false, @islogical);
    parse(p, varargin{:});
    
    if p.Results.hold
        hold(ax, 'on');
    end
    
    % Set tick label modes
    set(ax, 'XTickLabelMode', 'auto');
    set(ax, 'YTickLabelMode', 'auto');
    set(ax, 'XTickMode', 'auto');
    set(ax, 'YTickMode', 'auto');
    
    % Set labels
    if ~isempty(p.Results.xlabel)
        xlabel(ax, p.Results.xlabel);
    end
    if ~isempty(p.Results.ylabel)
        ylabel(ax, p.Results.ylabel);
    end
    if ~isempty(p.Results.title)
        title(ax, p.Results.title);
    end
    
    % Set limits
    if ~isempty(p.Results.xlim)
        xlim(ax, p.Results.xlim);
    end
    if ~isempty(p.Results.ylim)
        ylim(ax, p.Results.ylim);
    end
    
    % Set grid
    if p.Results.grid
        grid(ax, 'on');
    end
end

% ========== Event Marker Functions ==========

function add_reach_markers(ax, startS, reachStart, varargin)
% ADD_REACH_MARKERS Add vertical lines at reach onsets
%
% Variables:
%   ax - Axes handle (optional, defaults to gca)
%   startS - Time vector for current plot
%   reachStart - Vector of reach onset times
%   varargin - Optional name-value pairs:
%       'color' - Line color (default: [0.5 0.5 0.5])
%       'lineWidth' - Line width (default: 0.8)
%       'lineStyle' - Line style (default: '--')
%       'alpha' - Line alpha (default: 0.7)

    if nargin < 1 || isempty(ax)
        ax = gca;
    end
    
    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'color', [0.5 0.5 0.5], @isnumeric);
    addParameter(p, 'lineWidth', 0.8, @isnumeric);
    addParameter(p, 'lineStyle', '--', @ischar);
    addParameter(p, 'alpha', 0.7, @isnumeric);
    parse(p, varargin{:});
    
    if isempty(startS) || isempty(reachStart)
        return;
    end
    
    % Filter reach onsets to only show those within the current plot's time range
    plotTimeRange = [startS(1), startS(end)];
    reachOnsetsInRange = reachStart(...
        reachStart >= plotTimeRange(1) & reachStart <= plotTimeRange(2));
    
    if ~isempty(reachOnsetsInRange)
        for i = 1:length(reachOnsetsInRange)
            xline(ax, reachOnsetsInRange(i), ...
                'Color', p.Results.color, ...
                'LineWidth', p.Results.lineWidth, ...
                'LineStyle', p.Results.lineStyle, ...
                'Alpha', p.Results.alpha);
        end
    end
end

function add_block2_marker(ax, startBlock2, varargin)
% ADD_BLOCK2_MARKER Add vertical line at block 2 start
%
% Variables:
%   ax - Axes handle (optional, defaults to gca)
%   startBlock2 - Block 2 start time
%   varargin - Optional name-value pairs:
%       'color' - Line color (default: [1 0 0] or [.8 0 0])
%       'lineWidth' - Line width (default: 3)

    if nargin < 1 || isempty(ax)
        ax = gca;
    end
    
    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'color', [.8 0 0], @isnumeric);
    addParameter(p, 'lineWidth', 3, @isnumeric);
    parse(p, varargin{:});
    
    if ~isempty(startBlock2)
        xline(ax, startBlock2, ...
            'Color', p.Results.color, ...
            'LineWidth', p.Results.lineWidth);
    end
end

function add_response_markers(ax, startS, responseOnset, varargin)
% ADD_RESPONSE_MARKERS Add vertical lines at response onsets (schall data)
%
% Variables:
%   ax - Axes handle (optional, defaults to gca)
%   startS - Time vector for current plot
%   responseOnset - Vector of response onset times
%   varargin - Optional name-value pairs (same as add_reach_markers)

    if nargin < 1 || isempty(ax)
        ax = gca;
    end
    
    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'color', [0.5 0.5 0.5], @isnumeric);
    addParameter(p, 'lineWidth', 0.8, @isnumeric);
    addParameter(p, 'lineStyle', '--', @ischar);
    addParameter(p, 'alpha', 0.7, @isnumeric);
    parse(p, varargin{:});
    
    if isempty(startS) || isempty(responseOnset)
        return;
    end
    
    % Filter response onsets to only show those within the current plot's time range
    plotTimeRange = [startS(1), startS(end)];
    responseOnsetsInRange = responseOnset(...
        responseOnset >= plotTimeRange(1) & responseOnset <= plotTimeRange(2));
    
    if ~isempty(responseOnsetsInRange)
        for i = 1:length(responseOnsetsInRange)
            xline(ax, responseOnsetsInRange(i), ...
                'Color', p.Results.color, ...
                'LineWidth', p.Results.lineWidth, ...
                'LineStyle', p.Results.lineStyle, ...
                'Alpha', p.Results.alpha);
        end
    end
end

function add_trial_type_line(ax, T, varargin)
% ADD_TRIAL_TYPE_LINE Add trial type line (hong data)
%
% Variables:
%   ax - Axes handle (optional, defaults to gca)
%   T - Table with startTime_oe and trialType fields
%   varargin - Optional name-value pairs:
%       'color' - Line color (default: [1 .3 .3])
%       'lineWidth' - Line width (default: 2)
%       'scale' - Scale factor for trialType (default: 1/8)

    if nargin < 1 || isempty(ax)
        ax = gca;
    end
    
    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'color', [1 .3 .3], @isnumeric);
    addParameter(p, 'lineWidth', 2, @isnumeric);
    addParameter(p, 'scale', 1/8, @isnumeric);
    parse(p, varargin{:});
    
    if ~isempty(T) && isfield(T, 'startTime_oe') && isfield(T, 'trialType')
        plot(ax, T.startTime_oe, T.trialType * p.Results.scale, ...
            'Color', p.Results.color, ...
            'LineWidth', p.Results.lineWidth, ...
            'LineStyle', '-');
    end
end

% ========== Figure Saving Function ==========

function save_figure(figHandle, saveDir, filename, varargin)
% SAVE_FIGURE Save figure with standard naming and export settings
%
% Variables:
%   figHandle - Figure handle (optional, defaults to gcf)
%   saveDir - Directory to save figure
%   filename - Filename (without extension)
%   varargin - Optional name-value pairs:
%       'filePrefix' - Prefix to add to filename (default: '')
%       'resolution' - Export resolution (default: 300)
%       'format' - File format (default: 'png')

    if nargin < 1 || isempty(figHandle)
        figHandle = gcf;
    end
    
    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'filePrefix', '', @ischar);
    addParameter(p, 'resolution', 300, @isnumeric);
    addParameter(p, 'format', 'png', @ischar);
    parse(p, varargin{:});
    
    % Construct full filename
    if ~isempty(p.Results.filePrefix)
        fullFilename = sprintf('%s_%s.%s', p.Results.filePrefix, filename, p.Results.format);
    else
        fullFilename = sprintf('%s.%s', filename, p.Results.format);
    end
    
    filePath = fullfile(saveDir, fullFilename);
    
    % Save figure
    exportgraphics(figHandle, filePath, 'Resolution', p.Results.resolution);
    
    fprintf('Saved figure to: %s\n', filePath);
end
