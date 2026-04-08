%%
% spontaneous_behavior_clusters
% Find behavior clusters in spontaneous sessions based on label composition rules.
%
% Variables:
%   sessionName - Spontaneous session name (e.g., 'ag112321_1')
%   includeBhv - Cell array; each cell is a vector of labels to include
%   thresholdProp - Minimum proportion of included labels inside a valid cluster
%   minDur - Minimum cluster duration in seconds
%   maxConsecutiveNonDur - Max allowed consecutive non-included duration (seconds)
%   smoothLabels - Whether to smooth behavior labels before clustering
%   splitLongClustersForYield - If true, repeatedly split clusters while valid
%   splitTargetDur - Guides preferred split location (seconds) inside each segment
%   splitMinGapDur - Min non-included run duration (seconds) preferred as split boundary
%   maxSplitIterations - Safety cap on split attempts (avoids infinite loops)
%
% Goal:
%   For each includeBhv{k}, find all valid clusters such that:
%     1) cluster starts/ends with included labels,
%     2) included-label proportion >= thresholdProp,
%     3) duration >= minDur,
%     4) no consecutive non-included segment longer than maxConsecutiveNonDur.
%   Then store cluster labels and time windows, and plot duration histograms.

%% ============================= Configuration =============================
sessionName = 'ag112321_1';
collectEnd = []; % 60 * 60;  % seconds; set [] to use full behavior recording

% Each cell defines one behavior family to cluster.
includeBhv = {
    [0 1 2 13 14 15]
    [5:12]
    };

thresholdProp = 0.95;
minDur = 8;  % seconds
maxConsecutiveNonDur = 0.30;  % seconds

% Optional smoothing (same style as spontaneous_behavior_sequences.m)
smoothLabels = false;
smoothingWindow = 0.25;  % seconds

% Optional yield-improvement splitting (repeatedly splits until no further valid split)
splitLongClustersForYield = true;
splitTargetDur = 8;   % seconds (guides preferred split location inside try_split_once)
splitMinGapDur = 0.20; % seconds (prefer splits on non-include runs >= this)
maxSplitIterations = 10;  % safety cap; warns if reached

figureId = 32051;
nHistCols = 3;

%% ============================= Load Behavior Data =============================
pathParts = strsplit(sessionName, filesep);
subDir = pathParts{1}(1:2);

paths = get_paths;
sessionFolder = fullfile(paths.spontaneousDataPath, subDir, sessionName);

csvFiles = dir(fullfile(sessionFolder, 'behavior_labels*.csv'));
if isempty(csvFiles)
    error('No CSV file starting with "behavior_labels" found in %s', sessionFolder);
elseif numel(csvFiles) > 1
    warning('Multiple behavior_labels*.csv files found. Using first: %s', csvFiles(1).name);
end

bhvTable = readtable(fullfile(sessionFolder, csvFiles(1).name));
if ~ismember('Code', bhvTable.Properties.VariableNames)
    error('Behavior table must contain a "Code" column.');
end

if ~ismember('Time', bhvTable.Properties.VariableNames)
    error('Behavior table must contain a "Time" column.');
end

if ~isempty(collectEnd)
    bhvTable(bhvTable.Time > collectEnd, :) = [];
end

bopts = struct();
bopts.fsBhv = estimate_behavior_fs(bhvTable.Time);
bopts.smoothingWindow = smoothingWindow;
bopts.summarize = false;

labelsRaw = bhvTable.Code(:);
if smoothLabels
    labelsUsed = behavior_label_smoothing(labelsRaw, bopts);
else
    labelsUsed = labelsRaw;
end

timeAxis = bhvTable.Time(:);
if numel(timeAxis) ~= numel(labelsUsed)
    error('Time and label vectors are mismatched in length.');
end

frameDt = median(diff(timeAxis));
if ~(isfinite(frameDt) && frameDt > 0)
    error('Invalid behavior time axis; could not estimate frame duration.');
end

%% ============================= Cluster Extraction =============================
nGroups = numel(includeBhv);
clusterResults = cell(1, nGroups);
summaryTable = table('Size', [nGroups, 7], ...
    'VariableTypes', {'double', 'string', 'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'groupIdx', 'includeLabels', 'nClusters', 'meanDurSec', ...
    'medianDurSec', 'minDurSec', 'maxDurSec'});

fprintf('\n=== spontaneous_behavior_clusters ===\n');
fprintf('Session: %s | thresholdProp=%.3f | minDur=%.3fs | maxConsecutiveNonDur=%.3fs\n', ...
    sessionName, thresholdProp, minDur, maxConsecutiveNonDur);

for groupIdx = 1:nGroups
    includeLabels = includeBhv{groupIdx}(:)';
    if isempty(includeLabels)
        warning('includeBhv{%d} is empty. Skipping.', groupIdx);
        clusterResults{groupIdx} = empty_group_result(groupIdx, includeLabels);
        continue;
    end

    resultStruct = find_clusters_for_include_group( ...
        labelsUsed, timeAxis, includeLabels, thresholdProp, minDur, ...
        maxConsecutiveNonDur, splitLongClustersForYield, splitTargetDur, splitMinGapDur, frameDt, ...
        maxSplitIterations);
    clusterResults{groupIdx} = resultStruct;

    durations = resultStruct.clusterDurationSec(:);
    if isempty(durations)
        meanDur = nan;
        medianDur = nan;
        minDurVal = nan;
        maxDurVal = nan;
    else
        meanDur = mean(durations);
        medianDur = median(durations);
        minDurVal = min(durations);
        maxDurVal = max(durations);
    end

    summaryTable(groupIdx, :) = {groupIdx, mat2str(includeLabels), numel(durations), ...
        meanDur, medianDur, minDurVal, maxDurVal};
end

%% ============================= Print Summary =============================
fprintf('\n=== Cluster Summary by includeBhv Group ===\n');
for groupIdx = 1:nGroups
    groupResult = clusterResults{groupIdx};
    nClusters = numel(groupResult.clusterDurationSec);
    fprintf('Group %d | include=%s | nClusters=%d', ...
        groupIdx, mat2str(groupResult.includeLabels), nClusters);
    if nClusters > 0
        fprintf(' | meanDur=%.2fs | medianDur=%.2fs\n', ...
            mean(groupResult.clusterDurationSec), median(groupResult.clusterDurationSec));
    else
        fprintf('\n');
    end
end

%% ============================= Plot Histograms =============================
nHistRows = ceil(nGroups / nHistCols);
figHandle = figure(figureId);
clf(figHandle);
set(figHandle, 'Name', 'Spontaneous Behavior Clusters', 'Color', 'w', 'NumberTitle', 'off');

if isprop(figHandle, 'WindowState')
    figHandle.WindowState = 'maximized';
end

for groupIdx = 1:nGroups
    subplot(nHistRows, nHistCols, groupIdx);
    hold on;

    groupResult = clusterResults{groupIdx};
    durations = groupResult.clusterDurationSec(:);
    if isempty(durations)
        text(0.5, 0.5, 'No clusters', 'HorizontalAlignment', 'center', 'Units', 'normalized');
        axis off;
    else
        histogram(durations, 'FaceColor', [0.3 0.6 0.85], 'EdgeColor', 'k');
        xlabel('Cluster duration (s)');
        ylabel('Count');
        title(sprintf('Group %d: include=%s, n=%d', ...
            groupIdx, mat2str(groupResult.includeLabels), numel(durations)), ...
            'Interpreter', 'none');
        grid on;
    end
    hold off;
end

sgtitle(sprintf('Spontaneous behavior clusters | %s', sessionName), 'Interpreter', 'none');

%% ============================= Output Variables =============================
% clusterResults{g} fields:
%   .groupIdx
%   .includeLabels
%   .clusterIdxWindow      [nClusters x 2] start/end row index in bhvTable
%   .clusterTimeWindow     [nClusters x 2] start/end time in seconds
%   .clusterDurationSec    [nClusters x 1] duration in seconds
%   .clusterLabels         {nClusters x 1} label vectors for each cluster
%   .clusterTimeAxis       {nClusters x 1} time vectors for each cluster
%
% summaryTable:
%   one row per includeBhv group.

%% ============================= Local Functions =============================
function resultStruct = empty_group_result(groupIdx, includeLabels)
% empty_group_result
% Variables:
%   groupIdx - includeBhv group index
%   includeLabels - behavior labels for this group
% Goal:
%   Return an empty output struct with expected fields.

    resultStruct = struct();
    resultStruct.groupIdx = groupIdx;
    resultStruct.includeLabels = includeLabels(:)';
    resultStruct.clusterIdxWindow = zeros(0, 2);
    resultStruct.clusterTimeWindow = zeros(0, 2);
    resultStruct.clusterDurationSec = zeros(0, 1);
    resultStruct.clusterLabels = cell(0, 1);
    resultStruct.clusterTimeAxis = cell(0, 1);
end

function resultStruct = find_clusters_for_include_group(labelsVec, timeAxis, includeLabels, thresholdProp, ...
    minDur, maxConsecutiveNonDur, splitLongClustersForYield, splitTargetDur, splitMinGapDur, frameDt, ...
    maxSplitIterations)
% find_clusters_for_include_group
% Variables:
%   labelsVec - full behavior label vector
%   timeAxis - full behavior time vector (s), same length as labelsVec
%   includeLabels - labels defining "included" behavior for this group
%   thresholdProp - min included proportion within cluster
%   minDur - minimum cluster duration (s)
%   maxConsecutiveNonDur - max consecutive non-included duration (s)
%   splitLongClustersForYield - whether to split long valid clusters
%   splitTargetDur - target duration for split heuristic (s)
%   splitMinGapDur - preferred minimum non-included run duration for split points (s)
%   frameDt - median behavior frame duration (s)
%   maxSplitIterations - max split attempts in split_clusters_for_yield
% Goal:
%   Extract valid clusters for one includeBhv group.

    resultStruct = empty_group_result(0, includeLabels);
    isIncluded = ismember(labelsVec, includeLabels);

    % Candidate blocks are delimited by runs of non-included labels that exceed
    % maxConsecutiveNonDur. These are hard boundaries.
    maxNonFrames = max(0, floor(maxConsecutiveNonDur / frameDt));
    breakMask = get_break_mask_from_nonincluded_runs(isIncluded, maxNonFrames);

    blockEdges = [1; find(breakMask) + 1; numel(labelsVec) + 1];
    candidateWindows = zeros(0, 2);
    for b = 1:(numel(blockEdges) - 1)
        startIdx = blockEdges(b);
        endIdx = blockEdges(b + 1) - 1;
        if endIdx < startIdx
            continue;
        end

        % Enforce "cluster begins/ends with included label"
        while startIdx <= endIdx && ~isIncluded(startIdx)
            startIdx = startIdx + 1;
        end
        while endIdx >= startIdx && ~isIncluded(endIdx)
            endIdx = endIdx - 1;
        end

        if startIdx > endIdx
            continue;
        end

        if is_valid_cluster(startIdx, endIdx, isIncluded, timeAxis, thresholdProp, minDur)
            candidateWindows(end + 1, :) = [startIdx, endIdx]; %#ok<AGROW>
        end
    end

    if splitLongClustersForYield
        candidateWindows = split_clusters_for_yield(candidateWindows, isIncluded, timeAxis, ...
            thresholdProp, minDur, splitTargetDur, splitMinGapDur, frameDt, maxSplitIterations);
    end

    % Build output fields
    nClusters = size(candidateWindows, 1);
    resultStruct.clusterIdxWindow = candidateWindows;
    resultStruct.clusterTimeWindow = zeros(nClusters, 2);
    resultStruct.clusterDurationSec = zeros(nClusters, 1);
    resultStruct.clusterLabels = cell(nClusters, 1);
    resultStruct.clusterTimeAxis = cell(nClusters, 1);

    for c = 1:nClusters
        iStart = candidateWindows(c, 1);
        iEnd = candidateWindows(c, 2);
        tStart = timeAxis(iStart);
        tEnd = timeAxis(iEnd);
        durSec = tEnd - tStart;

        resultStruct.clusterTimeWindow(c, :) = [tStart, tEnd];
        resultStruct.clusterDurationSec(c) = durSec;
        resultStruct.clusterLabels{c} = labelsVec(iStart:iEnd);
        resultStruct.clusterTimeAxis{c} = timeAxis(iStart:iEnd);
    end
end

function breakMask = get_break_mask_from_nonincluded_runs(isIncluded, maxNonFrames)
% get_break_mask_from_nonincluded_runs
% Variables:
%   isIncluded - logical vector marking included labels
%   maxNonFrames - max allowed consecutive non-included frames
% Goal:
%   Return a logical "break after this frame" mask using long non-included runs.

    nFrames = numel(isIncluded);
    breakMask = false(nFrames, 1);
    nonMask = ~isIncluded(:);
    if ~any(nonMask)
        return;
    end

    runStarts = find(diff([0; nonMask]) == 1);
    runEnds = find(diff([nonMask; 0]) == -1);
    runLengths = runEnds - runStarts + 1;

    longRunIdx = find(runLengths > maxNonFrames);
    for i = 1:numel(longRunIdx)
        idx = longRunIdx(i);
        runStart = runStarts(idx);
        runEnd = runEnds(idx);
        if runStart > 1
            breakMask(runStart - 1) = true;
        end
        if runEnd < nFrames
            breakMask(runEnd) = true;
        end
    end
end

function tf = is_valid_cluster(iStart, iEnd, isIncluded, timeAxis, thresholdProp, minDur)
% is_valid_cluster
% Variables:
%   iStart, iEnd - cluster start/end indices
%   isIncluded - logical include mask
%   timeAxis - behavior times
%   thresholdProp - minimum included-label proportion
%   minDur - minimum duration (s)
% Goal:
%   Return whether cluster satisfies start/end include, proportion, and duration.

    if iStart < 1 || iEnd > numel(isIncluded) || iStart >= iEnd
        tf = false;
        return;
    end
    if ~isIncluded(iStart) || ~isIncluded(iEnd)
        tf = false;
        return;
    end

    idxRange = iStart:iEnd;
    propIncluded = mean(isIncluded(idxRange));
    durationSec = timeAxis(iEnd) - timeAxis(iStart);
    tf = (propIncluded >= thresholdProp) && (durationSec >= minDur);
end

function splitWindows = split_clusters_for_yield(candidateWindows, isIncluded, timeAxis, ...
    thresholdProp, minDur, splitTargetDur, splitMinGapDur, frameDt, maxSplitIterations)
% split_clusters_for_yield
% Variables:
%   candidateWindows - [n x 2] valid clusters before splitting
%   isIncluded - include mask over full session labels
%   timeAxis - time vector
%   thresholdProp - minimum included-label proportion
%   minDur - minimum duration
%   splitTargetDur - target duration to encourage splitting long clusters
%   splitMinGapDur - preferred minimum non-included gap duration for split points
%   frameDt - frame duration
%   maxSplitIterations - safety limit on split attempts
% Goal:
%   Repeatedly split valid clusters into two valid sub-clusters whenever
%   try_split_once succeeds, until no further splits are possible (or limit hit).

    splitWindows = zeros(0, 2);
    minGapFrames = max(1, floor(splitMinGapDur / frameDt));

    queue = candidateWindows;
    queueHead = 1;
    splitCount = 0;

    while true
        if queueHead > size(queue, 1)
            break;
        end
        if splitCount >= maxSplitIterations
            warning('split_clusters_for_yield: hit maxSplitIterations=%d; appending remaining queue.', ...
                maxSplitIterations);
            splitWindows = [splitWindows; queue(queueHead:end, :)]; %#ok<AGROW>
            return;
        end

        iStart = queue(queueHead, 1);
        iEnd = queue(queueHead, 2);
        queueHead = queueHead + 1;

        durationSec = timeAxis(iEnd) - timeAxis(iStart);
        if durationSec < 2 * minDur
            splitWindows(end + 1, :) = [iStart, iEnd]; %#ok<AGROW>
            continue;
        end

        [didSplit, leftWin, rightWin] = try_split_once( ...
            iStart, iEnd, isIncluded, timeAxis, thresholdProp, minDur, splitTargetDur, minGapFrames);
        splitCount = splitCount + 1;

        if ~didSplit
            splitWindows(end + 1, :) = [iStart, iEnd]; %#ok<AGROW>
        else
            queue(end + 1, :) = leftWin; %#ok<AGROW>
            queue(end + 1, :) = rightWin; %#ok<AGROW>
        end
    end
end

function [didSplit, leftWin, rightWin] = try_split_once(iStart, iEnd, isIncluded, timeAxis, ...
    thresholdProp, minDur, splitTargetDur, minGapFrames)
% try_split_once
% Variables:
%   iStart, iEnd - bounds of one valid cluster
%   isIncluded - include mask
%   timeAxis - time vector
%   thresholdProp, minDur - validity constraints
%   splitTargetDur - target duration for balanced splitting
%   minGapFrames - preferred minimum non-included run length at split
% Goal:
%   Attempt one split to produce two valid sub-clusters.

    didSplit = false;
    leftWin = [iStart, iEnd];
    rightWin = [iStart, iEnd];

    idxRange = iStart:iEnd;
    localIncluded = isIncluded(idxRange);
    localN = numel(localIncluded);
    if localN < 3
        return;
    end

    targetMidTime = timeAxis(iStart) + splitTargetDur;
    [~, targetIdxGlobal] = min(abs(timeAxis(idxRange) - targetMidTime));
    targetSplitGlobal = idxRange(targetIdxGlobal);

    % Candidate split points: boundaries around non-included runs first.
    nonMask = ~localIncluded(:);
    runStarts = find(diff([0; nonMask]) == 1);
    runEnds = find(diff([nonMask; 0]) == -1);
    splitCandidates = [];
    for r = 1:numel(runStarts)
        runLen = runEnds(r) - runStarts(r) + 1;
        if runLen >= minGapFrames
            leftBoundaryGlobal = idxRange(runStarts(r)) - 1;
            rightBoundaryGlobal = idxRange(runEnds(r)) + 1;
            splitCandidates = [splitCandidates, leftBoundaryGlobal, rightBoundaryGlobal]; %#ok<AGROW>
        end
    end

    % Fallback: also allow any interior point.
    splitCandidates = unique([splitCandidates, (iStart + 1):(iEnd - 1)]);
    if isempty(splitCandidates)
        return;
    end

    % Score candidates by distance to target and validity of both sides.
    bestScore = inf;
    bestSplit = nan;
    for s = 1:numel(splitCandidates)
        splitIdx = splitCandidates(s);
        leftStart = iStart;
        leftEnd = splitIdx;
        rightStart = splitIdx + 1;
        rightEnd = iEnd;

        if ~is_valid_cluster(leftStart, leftEnd, isIncluded, timeAxis, thresholdProp, minDur)
            continue;
        end
        if ~is_valid_cluster(rightStart, rightEnd, isIncluded, timeAxis, thresholdProp, minDur)
            continue;
        end

        score = abs(splitIdx - targetSplitGlobal);
        if score < bestScore
            bestScore = score;
            bestSplit = splitIdx;
        end
    end

    if ~isnan(bestSplit)
        didSplit = true;
        leftWin = [iStart, bestSplit];
        rightWin = [bestSplit + 1, iEnd];
    end
end

function fsBhv = estimate_behavior_fs(timeVec)
% estimate_behavior_fs
% Variables:
%   timeVec - behavior timestamps (seconds)
% Goal:
%   Estimate behavior sampling rate from timestamp differences.

    dt = diff(timeVec(:));
    dt = dt(isfinite(dt) & dt > 0);
    if isempty(dt)
        fsBhv = 60;
        warning('Could not estimate behavior fs from Time. Using default fsBhv=60 Hz.');
        return;
    end
    fsBhv = 1 / median(dt);
end
