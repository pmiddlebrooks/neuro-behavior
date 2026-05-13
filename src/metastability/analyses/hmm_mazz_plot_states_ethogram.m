function hmm_mazz_plot_states_ethogram(hmmRes, config)
% HMM_MAZZ_PLOT_STATES_ETHOGRAM HMM state posteriors + behavior ethogram in one figure.
%
% Variables:
%   hmmRes - Top-level results from hmm_mazz_analysis / hmm_load_saved_model, or a
%            single-area hmm struct with .continuous_results, .HmmParam, .best_model.
%
%   config - Struct (required fields marked *):
%       .timeRangeSec*     - [tStart tEnd] seconds, same time base as HMM bins and
%                           behavior CSV Time (typically 0 = collection window start).
%       .sessionName      - Spontaneous session id (e.g. 'ag112321_1') for loading
%                           behavior_labels*.csv under spontaneousDataPath. Omit if
%                           .bhvTimeVec and .bhvID are supplied.
%       .brainArea        - 'M23'|'M56'|'DS'|'VS' when hmmRes lists multiple areas.
%       .paths            - Output of get_paths(); default: get_paths().
%       .bhvTimeVec       - Optional: behavior sample times (s), same length as .bhvID.
%       .bhvID            - Optional: behavior label per sample (numeric / int).
%       .behaviorCsvPath  - Optional: full path to behavior_labels*.csv (overrides
%                           sessionName-based discovery).
%       .timeOffsetSec    - Add to HMM bin times before compare/plot (default 0).
%
% Goal:
%   Top row: same style as hmm_mazz_plot — translucent state segments from thresholded
%   sequence plus posterior traces. Bottom row: behavior label ethogram (filled runs)
%   aligned in time, following spontaneous/criticality_spontaneous_clusters and
%   sfn2025/d2_task_vs_naturalistic patterns.

if nargin < 2 || isempty(config)
    error('config is required with field .timeRangeSec = [tStart tEnd].');
end

if ~isfield(config, 'timeRangeSec') || isempty(config.timeRangeSec) ...
        || numel(config.timeRangeSec) ~= 2
    error('config.timeRangeSec must be a two-element vector [tStart tEnd] in seconds.');
end

timeRangeSec = double(config.timeRangeSec(:)');
tStart = min(timeRangeSec);
tEnd = max(timeRangeSec);
if ~(isfinite(tStart) && isfinite(tEnd)) || tEnd <= tStart
    error('config.timeRangeSec must give a finite interval with tEnd > tStart.');
end

if ~isfield(config, 'timeOffsetSec') || isempty(config.timeOffsetSec)
    timeOffsetSec = 0;
else
    timeOffsetSec = double(config.timeOffsetSec);
end

% Resolve single-area HMM (same logic as hmm_mazz_plot.m)
if isfield(hmmRes, 'hmm_results') && iscell(hmmRes.hmm_results)
    if isfield(config, 'brainArea') && ~isempty(config.brainArea)
        areaMap = containers.Map({'M23', 'M56', 'DS', 'VS'}, {1, 2, 3, 4});
        if ~isKey(areaMap, config.brainArea)
            error('Unknown brain area "%s" in config.brainArea.', config.brainArea);
        end
        areaIdx = areaMap(config.brainArea);
        hmmResSingle = hmmRes.hmm_results{areaIdx};
        areaLabel = hmmRes.areas{areaIdx};
    else
        areaIdx = 1;
        while areaIdx <= numel(hmmRes.hmm_results) && isempty(hmmRes.hmm_results{areaIdx})
            areaIdx = areaIdx + 1;
        end
        if areaIdx > numel(hmmRes.hmm_results)
            error('No HMM results found in provided structure.');
        end
        hmmResSingle = hmmRes.hmm_results{areaIdx};
        areaLabel = hmmRes.areas{areaIdx};
    end
else
    hmmResSingle = hmmRes;
    if isfield(hmmResSingle, 'metadata') && isfield(hmmResSingle.metadata, 'brain_area')
        areaLabel = hmmResSingle.metadata.brain_area;
    else
        areaLabel = 'UnknownArea';
    end
end

if isempty(hmmResSingle) || ~isfield(hmmResSingle, 'continuous_results')
    error('hmmResSingle is empty or missing continuous_results.');
end

sequence = hmmResSingle.continuous_results.sequence(:);
probabilities = hmmResSingle.continuous_results.pStates;
if isempty(sequence) || isempty(probabilities)
    error('continuous_results.sequence or .pStates is empty.');
end

HmmParam = hmmResSingle.HmmParam;
if ~isfield(HmmParam, 'BinSize') || isempty(HmmParam.BinSize)
    error('HmmParam.BinSize is required.');
end
binSize = double(HmmParam.BinSize);

% Match hmm_mazz_plot time axis: bin k -> k * binSize (seconds from window start)
numBins = numel(sequence);
timeAxisFull = (1:numBins)' * binSize + timeOffsetSec;

idxSlice = find(timeAxisFull >= tStart & timeAxisFull <= tEnd);
if isempty(idxSlice)
    error(['No HMM bins fall in [%.6g, %.6g] s (have %.6g .. %.6g s). ' ...
        'Check timeRangeSec and timeOffsetSec.'], tStart, tEnd, timeAxisFull(1), timeAxisFull(end));
end

sequence = sequence(idxSlice);
probabilities = probabilities(idxSlice, :);
timeAxis = timeAxisFull(idxSlice);
numStatesProb = size(probabilities, 2);

stateColors = distinguishable_colors(max(numStatesProb, 4));

% --- Behavior vectors ---
[bhvTimeVec, bhvID] = local_load_behavior_vectors(config);

inWin = bhvTimeVec >= tStart & bhvTimeVec <= tEnd;
bhvTimeWin = bhvTimeVec(inWin);
bhvIDWin = bhvID(inWin);
if isempty(bhvTimeWin)
    warning('No behavior samples in time window; ethogram row will be empty.');
end

codesNat = unique(bhvIDWin(:)');
codesNat = codesNat(isfinite(codesNat));
if isempty(codesNat)
    codesNat = -1:15;
end
try
    behaviorColors = colors_for_behaviors(codesNat);
catch
    behaviorColors = lines(max(numel(codesNat), 4));
end

% --- Figure: HMM on top, ethogram bottom ---
figHandle = figure('Color', 'w', 'Name', sprintf('HMM + behavior ethogram | %s', areaLabel));
tiledLayoutHandle = tiledlayout(figHandle, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

axHmm = nexttile(tiledLayoutHandle);
hold(axHmm, 'on');

for stateIdx = 1:numStatesProb
    stateMask = sequence == stateIdx;
    if any(stateMask)
        diffMask = diff([0; stateMask; 0]);
        segStartIdx = find(diffMask == 1);
        segEndIdx = find(diffMask == -1) - 1;
        for segmentIdx = 1:numel(segStartIdx)
            xStart = timeAxis(segStartIdx(segmentIdx));
            xEnd = timeAxis(segEndIdx(segmentIdx));
            patch(axHmm, [xStart xEnd xEnd xStart], [0 0 1 1], stateColors(stateIdx, :), ...
                'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
    end
end

for stateIdx = 1:numStatesProb
    plot(axHmm, timeAxis, probabilities(:, stateIdx), ...
        'Color', stateColors(stateIdx, :), 'LineWidth', 1.5);
end

ylabel(axHmm, 'State probability');
ylim(axHmm, [0, 1]);
title(axHmm, sprintf('HMM states (%s)', areaLabel), 'Interpreter', 'none');
grid(axHmm, 'on');
hold(axHmm, 'off');

axEth = nexttile(tiledLayoutHandle);
hold(axEth, 'on');

for bhvCode = codesNat(:)'
    bhvMask = bhvIDWin == bhvCode;
    if ~any(bhvMask)
        continue;
    end
    diffMask = diff([0; bhvMask; 0]);
    starts = find(diffMask == 1);
    ends = find(diffMask == -1) - 1;
    colorIdx = find(codesNat == bhvCode, 1);
    if isempty(colorIdx) || colorIdx > size(behaviorColors, 1)
        plotColor = [0.5, 0.5, 0.5];
    else
        plotColor = behaviorColors(colorIdx, :);
    end
    for b = 1:numel(starts)
        xStart = bhvTimeWin(starts(b));
        xEnd = bhvTimeWin(ends(b));
        fill(axEth, [xStart, xEnd, xEnd, xStart], [0, 0, 1, 1], plotColor, 'EdgeColor', 'none');
    end
end

ylabel(axEth, 'Behavior');
ylim(axEth, [0, 1]);
title(axEth, 'Behavior ethogram', 'Interpreter', 'none');
grid(axEth, 'on');
hold(axEth, 'off');

xlim(axHmm, [tStart, tEnd]);
xlim(axEth, [tStart, tEnd]);
xlabel(axEth, 'Time (s)');

linkaxes([axHmm, axEth], 'x');

sgtitle(figHandle, sprintf('%s | %.3g–%.3g s', areaLabel, tStart, tEnd), 'Interpreter', 'none');

fprintf('hmm_mazz_plot_states_ethogram: %s, bins %d–%d, window [%.4g, %.4g] s\n', ...
    areaLabel, idxSlice(1), idxSlice(end), tStart, tEnd);

end

function [bhvTimeVec, bhvID] = local_load_behavior_vectors(config)
% LOCAL_LOAD_BEHAVIOR_VECTORS Load or accept behavior time series for ethogram.

if isfield(config, 'bhvTimeVec') && isfield(config, 'bhvID') ...
        && ~isempty(config.bhvTimeVec) && ~isempty(config.bhvID)
    bhvTimeVec = double(config.bhvTimeVec(:));
    bhvID = config.bhvID(:);
    if numel(bhvTimeVec) ~= numel(bhvID)
        error('config.bhvTimeVec and config.bhvID must have the same length.');
    end
    return;
end

if isfield(config, 'behaviorCsvPath') && ~isempty(config.behaviorCsvPath)
    csvPath = config.behaviorCsvPath;
    if ~exist(csvPath, 'file')
        error('behavior CSV not found: %s', csvPath);
    end
    dataFull = readtable(csvPath);
else
    if ~isfield(config, 'sessionName') || isempty(config.sessionName)
        error(['Provide config.sessionName for spontaneous behavior CSV lookup, or pass ', ...
            'config.bhvTimeVec and config.bhvID, or config.behaviorCsvPath.']);
    end

    if isfield(config, 'paths') && ~isempty(config.paths)
        paths = config.paths;
    else
        paths = get_paths;
    end

    [~, sessionBaseName, ~] = fileparts(config.sessionName);
    if isempty(sessionBaseName)
        sessionBaseName = config.sessionName;
    end

    pathParts = strsplit(config.sessionName, filesep);
    subDir = pathParts{1}(1:min(2, numel(pathParts{1})));
    dataPathBhv = fullfile(paths.spontaneousDataPath, subDir);
    sessionFolderBhv = fullfile(dataPathBhv, sessionBaseName);

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
        warning('Multiple behavior_labels CSV files; using %s', csvFiles(1).name);
    end
    csvPath = fullfile(sessionFolderBhv, csvFiles(1).name);
    dataFull = readtable(csvPath);
end

if isempty(dataFull) || height(dataFull) == 0
    error('Behavior table is empty.');
end

codeCol = local_pick_table_column(dataFull, {'Code', 'code', 'ID', 'id'});
timeCol = local_pick_table_column(dataFull, {'Time', 'time'});

bhvID = double(dataFull.(codeCol)(:));
bhvTimeVec = double(dataFull.(timeCol)(:));

if numel(bhvTimeVec) > 1 && ~issorted(bhvTimeVec)
    [bhvTimeVec, sortOrd] = sort(bhvTimeVec);
    bhvID = bhvID(sortOrd);
end

end

function colName = local_pick_table_column(tbl, candidates)
% LOCAL_PICK_TABLE_COLUMN First matching variable name on table.

vn = tbl.Properties.VariableNames;
for k = 1:numel(candidates)
    hit = strcmp(vn, candidates{k});
    if any(hit)
        colName = vn{find(hit, 1)};
        return;
    end
end
error('Table missing one of columns: %s. Found: %s', ...
    strjoin(candidates, ', '), strjoin(vn, ', '));
end
