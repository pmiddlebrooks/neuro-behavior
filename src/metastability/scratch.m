%% scratch.m — State x behavior occupancy heatmaps
%
% 1) P(behavior | state): time in each behavior while in state s; row-normalize per state.
% 2) P(state | behavior): time in each state while in behavior b; row-normalize per behavior.
%    (behaviors on y-axis, states on x-axis)
%
% Prerequisites: get_paths, hmm_load_saved_model on path.


paths = get_paths;

%% Config: session and saved HMM file (match run_hmm_mazz / hmm_mazz_analysis)
sessionType = 'spontaneous';   % 'spontaneous' | 'reach'
sessionName = 'ag112321_1';    % session folder / basename
brainArea = 'M56';             % 'M23' | 'M56' | 'DS' | 'VS'
binSizeLoad = 0.002;
minDurLoad = 0.04;
collectStartLoad = [];         % [] unless windowed save used _start_XX_end_XX
collectEndLoad = 120 * 60;      % e.g. 12*60; [] for full-session filename

% Optional time window for occupancy (seconds); [] = full HMM sequence
timeRangeSec = [];             % e.g. [0, 600];
timeOffsetSec = 0;             % add to HMM bin times if clocks differ

%% Load HMM results
loadArgs = {};
if ~isempty(brainArea)
    loadArgs = [loadArgs, {'brainArea'}, {brainArea}]; %#ok<AGROW>
end
if ~isempty(binSizeLoad)
    loadArgs = [loadArgs, {'binSize'}, {binSizeLoad}]; %#ok<AGROW>
end
if ~isempty(minDurLoad)
    loadArgs = [loadArgs, {'minDur'}, {minDurLoad}]; %#ok<AGROW>
end
if ~isempty(collectStartLoad)
    loadArgs = [loadArgs, {'collectStart'}, {collectStartLoad}]; %#ok<AGROW>
end
if ~isempty(collectEndLoad)
    loadArgs = [loadArgs, {'collectEnd'}, {collectEndLoad}]; %#ok<AGROW>
end

fprintf('Loading HMM via hmm_load_saved_model(%s, ...)\n', sessionType);
hmmLoaded = hmm_load_saved_model(sessionType, loadArgs{:});

% Single-area struct from loader when brainArea set; else pick from multi-area cell
if isfield(hmmLoaded, 'hmm_results') && iscell(hmmLoaded.hmm_results)
    areaMap = containers.Map({'M23', 'M56', 'DS', 'VS'}, {1, 2, 3, 4});
    if isempty(brainArea)
        error('brainArea must be set when loader returns multi-area results.');
    end
    hmmSingle = hmmLoaded.hmm_results{areaMap(brainArea)};
    areaLabel = brainArea;
else
    hmmSingle = hmmLoaded;
    if isfield(hmmSingle, 'metadata') && isfield(hmmSingle.metadata, 'brain_area')
        areaLabel = hmmSingle.metadata.brain_area;
    else
        areaLabel = brainArea;
    end
end

if isempty(hmmSingle) || ~isfield(hmmSingle, 'continuous_results')
    error('No continuous_results in loaded HMM struct.');
end

sequence = hmmSingle.continuous_results.sequence(:);
if isempty(sequence)
    error('continuous_results.sequence is empty.');
end

binSize = double(hmmSingle.HmmParam.BinSize);
numStatesModel = hmmSingle.best_model.num_states;
timeAxisFull = ((1:numel(sequence))' * binSize) + timeOffsetSec;

if isempty(timeRangeSec)
    tStart = timeAxisFull(1);
    tEnd = timeAxisFull(end);
else
    tStart = min(timeRangeSec);
    tEnd = max(timeRangeSec);
    keepMask = timeAxisFull >= tStart & timeAxisFull <= tEnd;
    sequence = sequence(keepMask);
    timeAxisFull = timeAxisFull(keepMask);
end

if isempty(sequence)
    error('No HMM bins in selected time window.');
end

%% Behavior labels (same discovery as hmm_mazz_plot_states_ethogram.m)
configBhv = struct();
configBhv.sessionName = sessionName;
configBhv.paths = paths;
[bhvTimeVec, bhvID] = scratch_load_behavior_vectors(configBhv);

bhvAtBin = interp1(bhvTimeVec, double(bhvID(:)), timeAxisFull(:), 'nearest', NaN);

%% State x behavior time (seconds), then row-normalize per state
stateList = (1:numStatesModel)';
behaviorCodes = unique(bhvAtBin(isfinite(bhvAtBin)));
behaviorCodes = sort(behaviorCodes(:)');
if isempty(behaviorCodes)
    error('No finite behavior labels aligned to HMM bins.');
end

numStatesPlot = numel(stateList);
numBehaviors = numel(behaviorCodes);
stateBhvTimeSec = zeros(numStatesPlot, numBehaviors);

codeToCol = containers.Map('KeyType', 'double', 'ValueType', 'double');
for colIdx = 1:numBehaviors
    codeToCol(behaviorCodes(colIdx)) = colIdx;
end

for binIdx = 1:numel(sequence)
    stateLabel = double(sequence(binIdx));
    behaviorLabel = bhvAtBin(binIdx);
    if stateLabel < 1 || stateLabel > numStatesPlot || ~isfinite(behaviorLabel)
        continue;
    end
    colIdx = codeToCol(behaviorLabel);
    stateBhvTimeSec(stateLabel, colIdx) = stateBhvTimeSec(stateLabel, colIdx) + binSize;
end

stateTickLabels = arrayfun(@(k) sprintf('S%d', k), stateList, 'UniformOutput', false);
bhvTickLabels = arrayfun(@(c) sprintf('B%d', c), behaviorCodes, 'UniformOutput', false);
whiteToRedMap = scratch_white_to_red_colormap(256);

% P(behavior | state): normalize each state row across behaviors
stateBhvNorm = scratch_normalize_rows(stateBhvTimeSec);

%% Heatmap 1: states (y) x behaviors (x), P(behavior | state)
figHeat = figure('Color', 'w', 'Name', sprintf('P(bhv|state) | %s', areaLabel));
imagesc(stateBhvNorm);
axis xy;
colorbar;
caxis([0, 1]);
colormap(whiteToRedMap);
set(gca, 'XTick', 1:numBehaviors, 'XTickLabel', bhvTickLabels, 'XTickLabelRotation', 45);
set(gca, 'YTick', 1:numStatesPlot, 'YTickLabel', stateTickLabels);
xlabel('Behavior code');
ylabel('HMM state');
title(sprintf('%s | P(behavior | state) | %.3g–%.3g s', areaLabel, tStart, tEnd), ...
    'Interpreter', 'none');
sgtitle(sprintf('%s | bin=%.4g s | assigned bins only', areaLabel, binSize), ...
    'Interpreter', 'none');

fprintf('P(behavior|state): %s, states 1..%d, %d behavior codes\n', ...
    areaLabel, numStatesModel, numBehaviors);
fprintf('State x behavior time (s) before normalize:\n');
disp(array2table(stateBhvTimeSec, 'VariableNames', bhvTickLabels, ...
    'RowNames', stateTickLabels));

% P(state | behavior): same counts, behaviors as rows, normalize per behavior
bhvStateTimeSec = stateBhvTimeSec';
bhvStateNorm = scratch_normalize_rows(bhvStateTimeSec);

%% Heatmap 2: behaviors (y) x states (x), P(state | behavior)
figHeatBhv = figure('Color', 'w', 'Name', sprintf('P(state|bhv) | %s', areaLabel));
imagesc(bhvStateNorm);
axis xy;
colorbar;
caxis([0, 1]);
colormap(whiteToRedMap);
set(gca, 'XTick', 1:numStatesPlot, 'XTickLabel', stateTickLabels, 'XTickLabelRotation', 45);
set(gca, 'YTick', 1:numBehaviors, 'YTickLabel', bhvTickLabels);
xlabel('HMM state');
ylabel('Behavior code');
title(sprintf('%s | P(state | behavior) | %.3g–%.3g s', areaLabel, tStart, tEnd), ...
    'Interpreter', 'none');
sgtitle(sprintf('%s | bin=%.4g s | assigned bins only', areaLabel, binSize), ...
    'Interpreter', 'none');

fprintf('P(state|behavior): behavior time (s) before normalize:\n');
disp(array2table(bhvStateTimeSec, 'VariableNames', stateTickLabels, ...
    'RowNames', bhvTickLabels));

%% --- local functions ---

function normMat = scratch_normalize_rows(countMat)
% SCRATCH_NORMALIZE_ROWS Divide each row by its sum (rows with zero sum stay zero).

normMat = zeros(size(countMat));
for rowIdx = 1:size(countMat, 1)
    rowTotal = sum(countMat(rowIdx, :));
    if rowTotal > 0
        normMat(rowIdx, :) = countMat(rowIdx, :) ./ rowTotal;
    end
end

end

function cmap = scratch_white_to_red_colormap(nSteps)
% SCRATCH_WHITE_TO_RED_COLORMAP Colormap from white (0) to red (1).

cmap = [ones(nSteps, 1), linspace(1, 0, nSteps)', linspace(1, 0, nSteps)'];

end

function [bhvTimeVec, bhvID] = scratch_load_behavior_vectors(config)
% SCRATCH_LOAD_BEHAVIOR_VECTORS Load behavior_labels*.csv (ethogram-style).

if isfield(config, 'bhvTimeVec') && isfield(config, 'bhvID') ...
        && ~isempty(config.bhvTimeVec) && ~isempty(config.bhvID)
    bhvTimeVec = double(config.bhvTimeVec(:));
    bhvID = config.bhvID(:);
    return;
end

if ~isfield(config, 'sessionName') || isempty(config.sessionName)
    error('config.sessionName required for behavior CSV lookup.');
end

paths = config.paths;
[~, sessionBaseName, ~] = fileparts(config.sessionName);
if isempty(sessionBaseName)
    sessionBaseName = config.sessionName;
end

pathParts = strsplit(config.sessionName, filesep);
subDir = pathParts{1}(1:min(2, numel(pathParts{1})));
sessionFolderBhv = fullfile(paths.spontaneousDataPath, subDir, sessionBaseName);
if ~isfolder(sessionFolderBhv)
    sessionFolderFlat = fullfile(paths.spontaneousDataPath, sessionBaseName);
    if isfolder(sessionFolderFlat)
        sessionFolderBhv = sessionFolderFlat;
    end
end

csvFiles = dir(fullfile(sessionFolderBhv, 'behavior_labels*.csv'));
if isempty(csvFiles)
    error('No behavior_labels*.csv under %s', sessionFolderBhv);
end
if numel(csvFiles) > 1
    warning('Multiple behavior_labels CSV; using %s', csvFiles(1).name);
end

dataFull = readtable(fullfile(sessionFolderBhv, csvFiles(1).name));
codeCol = scratch_pick_table_column(dataFull, {'Code', 'code', 'ID', 'id'});
timeCol = scratch_pick_table_column(dataFull, {'Time', 'time'});
bhvID = double(dataFull.(codeCol)(:));
bhvTimeVec = double(dataFull.(timeCol)(:));

if numel(bhvTimeVec) > 1 && ~issorted(bhvTimeVec)
    [bhvTimeVec, sortOrd] = sort(bhvTimeVec);
    bhvID = bhvID(sortOrd);
end

fprintf('Loaded %d behavior samples from %s\n', numel(bhvID), csvFiles(1).name);

end

function colName = scratch_pick_table_column(tbl, candidates)
vn = tbl.Properties.VariableNames;
for k = 1:numel(candidates)
    hit = strcmp(vn, candidates{k});
    if any(hit)
        colName = vn{find(hit, 1)};
        return;
    end
end
error('Table missing columns %s', strjoin(candidates, ', '));
end
