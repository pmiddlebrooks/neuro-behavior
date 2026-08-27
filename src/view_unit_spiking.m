%%
% View Unit Spiking
%
% Load one brain area from any session type and step through units in an
% interactive plot of binned spiking (rate in spikes/s).
%
% Variables (configure below, or set in the workspace before running):
%   sessionType, subjectName, sessionName
%   brainArea              - Single or merged area (e.g. 'M56', 'M23M56');
%                            '' lists available areas and stops
%   brainAreaCombinations  - Merged-area defs (M23M56 / M2356 by default)
%   collectStart, collectEnd - Time window (s); collectEnd = [] is session end
%   binSize                - Spike-count bin width (s)
%
% Controls:
%   Prev / Next buttons, slider, or keyboard Left/Right (also A/D, comma/period)
%   Home / End jump to first / last unit
%   Type an index or cluster ID in the box and press Enter
%
% Goal:
%   Inspect each unit's activity in the requested collect window.


%% Configuration

brainArea = 'M23M56';
brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
collectStart = 1145;
collectEnd = 1165;          % [] = full loaded session
binSize = 0.025;          % s

% Load session
opts = neuro_behavior_options();
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.05;
opts.maxFiringRate = 200;

subjectNameForLoad = '';
if ~isempty(subjectName)
    subjectNameForLoad = subjectName;
end
loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
dataStruct = load_session_data(sessionType, 'spikes', loadArgs{:});

if isempty(brainArea)
    fprintf('Available areas: %s\n', strjoin(dataStruct.areas, ', '));
    error('Set brainArea to one of the areas listed above (or a combined name such as M23M56).');
end

[dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
    dataStruct, brainArea, brainAreaCombinations);
if ~areaOk
    fprintf('Available areas: %s\n', strjoin(dataStruct.areas, ', '));
    error('Brain area "%s" not available in this session.', brainArea);
end

areaIdx = resolve_view_area_index(dataStruct, brainArea);
neuronIds = dataStruct.idLabel{areaIdx};
if isempty(neuronIds)
    error('No units in area %s.', brainArea);
end
neuronIds = neuronIds(:);

% Collect window
if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart') ...
        && ~isempty(dataStruct.spikeData.collectStart)
    collectStart = max(collectStart, dataStruct.spikeData.collectStart);
end
sessionEnd = resolve_view_session_end(dataStruct, collectStart);
collectEnd = clamp_collect_end_to_session(collectEnd, sessionEnd, collectStart);
if collectEnd <= collectStart
    error('collectEnd (%.1f) must be greater than collectStart (%.1f).', collectEnd, collectStart);
end

fprintf('\n=== View unit spiking ===\n');
fprintf('Session [%s]: %s\n', sessionType, sessionName);
fprintf('Area: %s (%d units)\n', dataStruct.areas{areaIdx}, numel(neuronIds));
fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', ...
    collectStart, collectEnd, (collectEnd - collectStart) / 60);
fprintf('binSize: %.3f s\n', binSize);
fprintf('Keys: Left/Right step, Home/End first/last\n');

inWin = dataStruct.spikeTimes >= collectStart & dataStruct.spikeTimes < collectEnd;
viewData = struct();
viewData.spikeTimes = dataStruct.spikeTimes(inWin);
viewData.spikeClusters = dataStruct.spikeClusters(inWin);
viewData.neuronIds = neuronIds;
viewData.areaName = dataStruct.areas{areaIdx};
viewData.sessionType = sessionType;
viewData.sessionName = char(sessionName);
viewData.collectStart = collectStart;
viewData.collectEnd = collectEnd;
viewData.binSize = binSize;
viewData.binEdges = collectStart:binSize:collectEnd;
if viewData.binEdges(end) < collectEnd
    viewData.binEdges(end + 1) = collectEnd;
end
viewData.binCenters = viewData.binEdges(1:end-1) + binSize / 2;
viewData.unitIdx = 1;
viewData.nUnits = numel(neuronIds);

launch_unit_spike_viewer(viewData);

%% Local functions

function areaIdx = resolve_view_area_index(dataStruct, brainArea)
% RESOLVE_VIEW_AREA_INDEX - Index of requested (possibly combined) area
if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
    areaIdx = dataStruct.areasToTest(1);
    return;
end
areaIdx = find(strcmp(dataStruct.areas, brainArea), 1);
if isempty(areaIdx)
    error('Could not find area "%s" after selection.', brainArea);
end
end

function sessionEnd = resolve_view_session_end(dataStruct, collectStart)
% RESOLVE_VIEW_SESSION_END - Session end time (s) from loaded spikes / opts
sessionEnd = [];
if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectEnd') ...
        && ~isempty(dataStruct.spikeData.collectEnd)
    sessionEnd = dataStruct.spikeData.collectEnd;
elseif isfield(dataStruct, 'spikeTimes') && ~isempty(dataStruct.spikeTimes)
    sessionEnd = max(dataStruct.spikeTimes);
elseif isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectEnd') ...
        && ~isempty(dataStruct.opts.collectEnd)
    sessionEnd = dataStruct.opts.collectEnd;
end
if isempty(sessionEnd) || ~isfinite(sessionEnd) || sessionEnd <= collectStart
    error('Could not resolve session end. Set collectEnd explicitly.');
end
end

function launch_unit_spike_viewer(viewData)
% LAUNCH_UNIT_SPIKE_VIEWER - Interactive figure to step through units
fig = figure('Color', 'w', 'Name', 'Unit spiking', ...
    'NumberTitle', 'off', 'Units', 'normalized', ...
    'Position', [0.12 0.18 0.76 0.62], ...
    'KeyPressFcn', @on_unit_view_key);
viewData.ax = axes('Parent', fig, 'Position', [0.08 0.22 0.88 0.70]);
viewData.lblStatus = uicontrol(fig, 'Style', 'text', 'Units', 'normalized', ...
    'Position', [0.22 0.08 0.56 0.06], 'BackgroundColor', 'w', ...
    'FontSize', 11, 'HorizontalAlignment', 'center');
viewData.editJump = uicontrol(fig, 'Style', 'edit', 'Units', 'normalized', ...
    'Position', [0.46 0.02 0.12 0.05], 'FontSize', 11, ...
    'String', '1', 'Callback', @on_unit_view_jump);
uicontrol(fig, 'Style', 'pushbutton', 'Units', 'normalized', ...
    'Position', [0.08 0.02 0.12 0.05], 'String', '< Prev', ...
    'FontSize', 11, 'Callback', @(~, ~) step_unit_view(fig, -1));
uicontrol(fig, 'Style', 'pushbutton', 'Units', 'normalized', ...
    'Position', [0.80 0.02 0.12 0.05], 'String', 'Next >', ...
    'FontSize', 11, 'Callback', @(~, ~) step_unit_view(fig, 1));
viewData.sldUnit = uicontrol(fig, 'Style', 'slider', 'Units', 'normalized', ...
    'Position', [0.22 0.025 0.22 0.04], ...
    'Min', 1, 'Max', max(2, viewData.nUnits), ...
    'Value', 1, 'SliderStep', slider_step_for_n(viewData.nUnits), ...
    'Callback', @on_unit_view_slider);
set(fig, 'UserData', viewData);
draw_current_unit(fig);
end

function step = slider_step_for_n(nUnits)
% SLIDER_STEP_FOR_N - Slider minor/major step for integer unit indices
if nUnits <= 1
    step = [1 1];
    return;
end
minorStep = 1 / (nUnits - 1);
step = [minorStep, min(1, 10 * minorStep)];
end

function step_unit_view(fig, deltaIdx)
% STEP_UNIT_VIEW - Move current unit by deltaIdx and redraw
if ~isgraphics(fig)
    return;
end
viewData = get(fig, 'UserData');
viewData.unitIdx = min(viewData.nUnits, max(1, viewData.unitIdx + deltaIdx));
set(fig, 'UserData', viewData);
draw_current_unit(fig);
end

function on_unit_view_slider(src, ~)
% ON_UNIT_VIEW_SLIDER - Jump to the slider unit index
fig = ancestor(src, 'figure');
viewData = get(fig, 'UserData');
viewData.unitIdx = min(viewData.nUnits, max(1, round(get(src, 'Value'))));
set(fig, 'UserData', viewData);
draw_current_unit(fig);
end

function on_unit_view_jump(src, ~)
% ON_UNIT_VIEW_JUMP - Jump to typed index or cluster ID
fig = ancestor(src, 'figure');
viewData = get(fig, 'UserData');
rawStr = strtrim(get(src, 'String'));
jumpVal = str2double(rawStr);
if ~isfinite(jumpVal)
    return;
end
if jumpVal >= 1 && jumpVal <= viewData.nUnits && abs(jumpVal - round(jumpVal)) < 1e-9
    viewData.unitIdx = round(jumpVal);
else
    matchIdx = find(viewData.neuronIds == jumpVal, 1);
    if isempty(matchIdx)
        fprintf('No unit index or cluster ID matching %s\n', rawStr);
        return;
    end
    viewData.unitIdx = matchIdx;
end
set(fig, 'UserData', viewData);
draw_current_unit(fig);
end

function on_unit_view_key(src, event)
% ON_UNIT_VIEW_KEY - Keyboard stepping (ignored while typing in the jump box)
curObj = get(src, 'CurrentObject');
if ~isempty(curObj) && isgraphics(curObj) && strcmpi(get(curObj, 'Type'), 'uicontrol') ...
        && strcmpi(get(curObj, 'Style'), 'edit')
    return;
end
switch event.Key
    case {'leftarrow', 'a', 'comma'}
        step_unit_view(src, -1);
    case {'rightarrow', 'd', 'period'}
        step_unit_view(src, 1);
    case 'home'
        viewData = get(src, 'UserData');
        viewData.unitIdx = 1;
        set(src, 'UserData', viewData);
        draw_current_unit(src);
    case 'end'
        viewData = get(src, 'UserData');
        viewData.unitIdx = viewData.nUnits;
        set(src, 'UserData', viewData);
        draw_current_unit(src);
end
end

function draw_current_unit(fig)
% DRAW_CURRENT_UNIT - Bin and plot the current unit in the collect window
viewData = get(fig, 'UserData');
unitIdx = viewData.unitIdx;
neuronId = viewData.neuronIds(unitIdx);
unitSpikes = viewData.spikeTimes(viewData.spikeClusters == neuronId);
spikeCounts = histcounts(unitSpikes, viewData.binEdges);
rateHz = spikeCounts(:)' / viewData.binSize;
meanRate = numel(unitSpikes) / (viewData.collectEnd - viewData.collectStart);

ax = viewData.ax;
cla(ax);
hold(ax, 'on');
plot(ax, viewData.binCenters, rateHz, 'Color', [0.15 0.35 0.7], 'LineWidth', 0.8);
yline(ax, meanRate, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1);
hold(ax, 'off');
xlim(ax, [viewData.collectStart, viewData.collectEnd]);
ylabel(ax, 'Spikes / s');
xlabel(ax, 'Time (s)');
title(ax, sprintf('%s | %s | %s | unit %d / %d | ID %g | %d spikes | mean %.2f Hz | bin %.0f ms', ...
    viewData.sessionType, viewData.sessionName, viewData.areaName, ...
    unitIdx, viewData.nUnits, neuronId, numel(unitSpikes), meanRate, ...
    viewData.binSize * 1000), 'Interpreter', 'none');
box(ax, 'off');
set(ax, 'TickDir', 'out');
grid(ax, 'on');

set(viewData.lblStatus, 'String', sprintf('Unit %d / %d   cluster ID %g', ...
    unitIdx, viewData.nUnits, neuronId));
set(viewData.editJump, 'String', sprintf('%d', unitIdx));
if viewData.nUnits > 1
    set(viewData.sldUnit, 'Value', unitIdx);
end
end
