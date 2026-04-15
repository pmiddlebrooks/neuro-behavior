%%
% Sequence-aligned peri-event HMM plotting for spontaneous data.
% Requires sequence variables from spontaneous_behavior_sequences.m to
% already exist in the workspace:
%   alignTimes, sequences, sequenceNames, alignOnIdx
%
% Variables:
%   sessionName   - spontaneous session name (e.g. 'ag112321_1')
%   binSize       - HMM bin size used in saved result filename
%   minDur        - HMM min duration used in saved result filename
%   collectStart, collectEnd - (optional) seconds; if collectEnd is set, filename matches
%       hmm_mazz_analysis: ..._start_XX_end_XX.mat before .mat
%   periWindowSec - peri-event window duration (seconds)
%   maxTrial      - max number of aligned events plotted per sequence
%   areasToTest   - area indices to consider from saved HMM results
%   savePlots     - if true, export each figure as .eps to HMM results folder (default false)
%
% Behavior strip: loads behavior_labels*.csv at native rate; each HMM bin maps to the
% nearest CSV Time via interp1 (avoids O(nBins*nWindow) distance matrices that OOM).

%% Parameters
paths = get_paths;

if ~exist('sessionName', 'var') || isempty(sessionName)
    sessionName = 'ag112321_1';
end

if ~exist('savePlots', 'var') || isempty(savePlots)
    savePlots = false;
end

binSize = 0.005;
minDur = 0.05;
collectStart = [];
collectEnd = []; % set with collectStart for windowed HMM saves; [] = full session filename
collectEnd = 60*60; % set with collectStart for windowed HMM saves; [] = full session filename
periWindowSec = 16;
maxTrial = 100;
areasToTest = 1:4;
probMetric = 'entropy'; % 'maxProb' | 'dominance' | 'margin' | 'entropy'

% Validate required sequence variables (manual workflow only)
hasSequenceVars = exist('alignTimes', 'var') && exist('sequences', 'var') && ...
    exist('sequenceNames', 'var') && exist('alignOnIdx', 'var');

if ~hasSequenceVars
    error(['Missing sequence variables in workspace. Run spontaneous_behavior_sequences.m first to define ', ...
        'alignTimes, sequences, sequenceNames, and alignOnIdx.']);
end

numSequences = numel(alignTimes);
if numSequences == 0
    error('No sequences available in alignTimes.');
end

% Load saved HMM analysis results (no legacy fallback)
[~, sessionBaseName, ~] = fileparts(sessionName);
timeWindowSuffix = '';
if exist('collectEnd', 'var') && ~isempty(collectEnd)
    collectStartSec = 0;
    if exist('collectStart', 'var') && ~isempty(collectStart)
        collectStartSec = collectStart;
    end
    timeWindowSuffix = sprintf('_start_%g_end_%g', collectStartSec, collectEnd);
end
resultsPath = fullfile(paths.spontaneousResultsPath, sessionBaseName, ...
    sprintf('hmm_mazz_spontaneous_bin%.3f_minDur%.3f%s.mat', binSize, minDur, timeWindowSuffix));

if ~exist(resultsPath, 'file')
    error('HMM result file not found: %s', resultsPath);
end

fprintf('Loading HMM analysis from: %s\n', resultsPath);
loadedResults = load(resultsPath);
results = loadedResults.results;
areas = results.areas;
hmmResults = results.hmm_results;
hmmdir = fileparts(resultsPath);

% Load behavior labels at native CSV / B-SOiD rate (same as spontaneous_behavior_sequences)
fprintf('Loading behavior labels (behavior_labels*.csv, fsBhv) for session %s...\n', sessionName);
pathParts = strsplit(sessionName, filesep);
subDir = pathParts{1}(1:min(2, numel(pathParts{1})));
dataPathBhv = fullfile(paths.spontaneousDataPath, subDir);
sessionFolderBhv = fullfile(dataPathBhv, sessionBaseName);

csvFiles = dir(fullfile(sessionFolderBhv, 'behavior_labels*.csv'));
if isempty(csvFiles)
    error('No CSV file starting with "behavior_labels" found in %s', sessionFolderBhv);
elseif numel(csvFiles) > 1
    warning('hmm_mazz_peri_nat:multipleCSV', ...
        'Multiple behavior_labels CSV files found. Using: %s', csvFiles(1).name);
end
dataFull = readtable(fullfile(sessionFolderBhv, csvFiles(1).name));
if isempty(dataFull) || height(dataFull) == 0
    error('Behavior CSV is empty: %s', csvFiles(1).name);
end

opts = neuro_behavior_options;

bhvID = dataFull.Code(:);
bhvTimeVec = dataFull.Time(:);
nBhvBins = numel(bhvID);
% interp1(...,'nearest') requires monotonic X; sort once (cheap vs OOM below).
if nBhvBins > 1 && ~issorted(bhvTimeVec)
    [bhvTimeVec, sortOrd] = sort(bhvTimeVec);
    bhvID = bhvID(sortOrd);
end
fprintf('  Loaded %d behavior samples (native fsBhv=%g Hz; one row per CSV sample).\n', ...
    nBhvBins, opts.fsBhv);

% Build alignment event times per sequence
eventTimesPerSequence = cell(1, numSequences);
for seqIdx = 1:numSequences
    seqOccurrences = alignTimes{seqIdx};
    if isempty(seqOccurrences)
        eventTimesPerSequence{seqIdx} = [];
        continue;
    end
    
    if iscell(seqOccurrences)
        sequenceEventTimes = nan(1, numel(seqOccurrences));
        for occIdx = 1:numel(seqOccurrences)
            timeVec = seqOccurrences{occIdx};
            if ~isempty(timeVec) && alignOnIdx <= numel(timeVec)
                sequenceEventTimes(occIdx) = timeVec(alignOnIdx);
            end
        end
        eventTimesPerSequence{seqIdx} = sequenceEventTimes(~isnan(sequenceEventTimes));
    else
        eventTimesPerSequence{seqIdx} = seqOccurrences(:)';
    end
end

% Prepare color maps
% HMM states: distinguishable colors from figure_tools
figureToolsPath = '/Users/paulmiddlebrooks/Projects/figure_tools';
if exist(figureToolsPath, 'dir')
    addpath(figureToolsPath);
end

% Behavior labels: fixed mapping from colors_for_behaviors
behaviorIdList = -1:15;
behaviorColors = colors_for_behaviors(behaviorIdList);

%% Plot one figure per area
for areaIdx = areasToTest
    if areaIdx > numel(hmmResults)
        continue;
    end
    
    hmmRes = hmmResults{areaIdx};
    if isempty(hmmRes) || ~isfield(hmmRes, 'metadata') || ...
            ~isfield(hmmRes.metadata, 'analysis_status') || ...
            ~strcmp(hmmRes.metadata.analysis_status, 'SUCCESS')
        fprintf('Skipping area %d (no successful HMM result).\n', areaIdx);
        continue;
    end
    
    continuousSequence = hmmRes.continuous_results.sequence(:);
    if isfield(hmmRes.continuous_results, 'pStates') && ~isempty(hmmRes.continuous_results.pStates)
        continuousPStates = hmmRes.continuous_results.pStates;
    else
        warning('Area %d missing continuous_results.pStates; skipping.', areaIdx);
        continue;
    end
    hmmBinSize = hmmRes.HmmParam.BinSize;
    nStates = hmmRes.best_model.num_states;
    totalTimeBins = numel(continuousSequence);
    
    if exist('distinguishable_colors', 'file')
        stateColors = distinguishable_colors(max(1, nStates), [1 1 1]);
    else
        stateColors = lines(max(1, nStates));
    end
    stateCmap = [1 1 1; stateColors]; % state 0/undefined = white
    
    windowBins = ceil(periWindowSec / hmmBinSize);
    halfWindowBins = floor(windowBins / 2);
    nBinsWindow = windowBins + 1;
    timeAxis = (-halfWindowBins:halfWindowBins) * hmmBinSize;
    
    figure(5000 + areaIdx); clf;
    monitorPositions = get(0, 'MonitorPositions');
    if size(monitorPositions, 1) >= 2
        set(gcf, 'Position', monitorPositions(end, :));
    else
        set(gcf, 'Position', monitorPositions(1, :));
    end
    
    if exist('tight_subplot', 'file')
        ha = tight_subplot(3, numSequences, [0.08 0.03], [0.08 0.08], [0.05 0.02]);
    else
        ha = zeros(3 * numSequences, 1);
        for axisIdx = 1:(3 * numSequences)
            ha(axisIdx) = subplot(3, numSequences, axisIdx);
        end
    end
    
    for seqIdx = 1:numSequences
        seqName = sequenceNames{seqIdx};
        seqEvents = eventTimesPerSequence{seqIdx};
        
        stateWindows = nan(0, nBinsWindow);
        probMetricWindows = nan(0, nBinsWindow);
        behaviorWindows = nan(0, nBinsWindow);
        
        for evIdx = 1:numel(seqEvents)
            eventTime = seqEvents(evIdx);
            centerIdx = round(eventTime / hmmBinSize) + 1;
            winStart = centerIdx - halfWindowBins;
            winEnd = centerIdx + halfWindowBins;
            
            if winStart < 1 || winEnd > totalTimeBins
                continue;
            end
            
            thisStateWindow = continuousSequence(winStart:winEnd)';
            thisPStateWindow = continuousPStates(winStart:winEnd, :);
            thisMetricWindow = compute_prob_metric(thisPStateWindow, probMetric);
            
            % HMM bin k -> absolute session time (same convention as hmm_mazz_analysis:
            % winTrain starts at 0). Nearest CSV row: use interp1 — NOT broadcast
            % abs(bhvTimeVec - hmmAbsTimes), which is nBhvBins x nBinsWindow (~GB+).
            hmmAbsTimes = ((winStart:winEnd) - 1) * hmmBinSize;
            bhvIdx = interp1(bhvTimeVec, (1:nBhvBins)', hmmAbsTimes(:), 'nearest', 'extrap');
            bhvIdx = round(bhvIdx);
            bhvIdx = max(1, min(nBhvBins, bhvIdx(:)'));
            thisBehaviorWindow = bhvID(bhvIdx)';
            
            stateWindows(end+1, :) = thisStateWindow; %#ok<AGROW>
            probMetricWindows(end+1, :) = thisMetricWindow; %#ok<AGROW>
            behaviorWindows(end+1, :) = thisBehaviorWindow; %#ok<AGROW>
            
            if size(stateWindows, 1) >= maxTrial
                break;
            end
        end
        
        nTrialsPlot = size(stateWindows, 1);
        
        % Row 1: aligned HMM states
        axState = ha(seqIdx);
        axes(axState); %#ok<LAXES>
        hold on;
        if nTrialsPlot > 0
            imagesc(timeAxis, 1:nTrialsPlot, stateWindows);
            ylim([0.5, nTrialsPlot + 0.5]);
        else
            imagesc(timeAxis, [1 1], nan(1, nBinsWindow));
            ylim([0.5, 1.5]);
        end
        colormap(axState, stateCmap);
        caxis(axState, [0, max(1, nStates)]);
        plot([0 0], ylim, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
        title(sprintf('%s\n%s (%d trials)', areas{areaIdx}, seqName, nTrialsPlot), ...
            'Interpreter', 'none', 'FontSize', 10);
        ylabel('Trial');
        set(gca, 'FontSize', 9);
        grid on;
        
        % Row 2: aligned posterior metric from pStates
        axMetric = ha(numSequences + seqIdx);
        axes(axMetric); %#ok<LAXES>
        hold on;
        if nTrialsPlot > 0
            imagesc(timeAxis, 1:nTrialsPlot, probMetricWindows);
            ylim([0.5, nTrialsPlot + 0.5]);
        else
            imagesc(timeAxis, [1 1], nan(1, nBinsWindow));
            ylim([0.5, 1.5]);
        end
        colormap(axMetric, parula(256));

            caxis(axMetric, [0, 1]);
        plot([0 0], ylim, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
        ylabel(sprintf('%s', probMetric));
        set(gca, 'FontSize', 9);
        grid on;
        colorbar('eastoutside');

        % Row 3: aligned behavior labels
        axBhv = ha(2 * numSequences + seqIdx);
        axes(axBhv); %#ok<LAXES>
        hold on;
        if nTrialsPlot > 0
            rgbImg = zeros(nTrialsPlot, nBinsWindow, 3);
            for rowIdx = 1:nTrialsPlot
                for colIdx = 1:nBinsWindow
                    labelVal = behaviorWindows(rowIdx, colIdx);
                    mapColorIdx = find(behaviorIdList == labelVal, 1);
                    if isempty(mapColorIdx)
                        rgbImg(rowIdx, colIdx, :) = [1 1 1];
                    else
                        rgbImg(rowIdx, colIdx, :) = behaviorColors(mapColorIdx, :);
                    end
                end
            end
            image(timeAxis, 1:nTrialsPlot, rgbImg);
            set(gca, 'YDir', 'normal');
            ylim([0.5, nTrialsPlot + 0.5]);
        else
            image(timeAxis, [1 1], ones(1, nBinsWindow, 3));
            set(gca, 'YDir', 'normal');
            ylim([0.5, 1.5]);
        end
        plot([0 0], ylim, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
        xlabel('Time relative to aligned behavior (s)');
        ylabel('Trial');
        set(gca, 'FontSize', 9);
        grid on;
    end
    
    sgtitle(sprintf(['%s - %s sequence-aligned HMM states (alignOnIdx=%d, window=%.1fs)\n' ...
        'Middle row metric from pStates: %s'], ...
        sessionBaseName, areas{areaIdx}, alignOnIdx, periWindowSec, probMetric), ...
        'Interpreter', 'none', 'FontSize', 14);
    
    if savePlots
        savePath = fullfile(hmmdir, sprintf('%s_sequence_aligned_hmm_states_area_%s_win%.1f.eps', ...
            sessionBaseName, areas{areaIdx}, periWindowSec));
        exportgraphics(gcf, savePath, 'ContentType', 'vector');
        fprintf('Saved area %s plot to: %s\n', areas{areaIdx}, savePath);
    end
end

function metricVals = compute_prob_metric(pStateWindow, probMetric)
% COMPUTE_PROB_METRIC Compute aligned scalar confidence metric from pStates.
%
% Variables:
%   pStateWindow - [nBins x nStates] posterior probabilities in peri window.
%   probMetric   - String selector:
%       'maxProb'    : max posterior per bin (confidence of winning state).
%       'dominance'  : 1 - normalized entropy (high = concentrated posterior).
%       'margin'     : top1 - top2 posterior gap per bin.
%       'entropy'    : normalized entropy (high = uncertainty).
%
% Goal:
%   Reduce per-bin posterior vectors into one interpretable trace that can
%   be visualized trial-by-trial alongside state and behavior rasters.

if nargin < 2 || isempty(probMetric)
    probMetric = 'maxProb';
end

if isempty(pStateWindow)
    metricVals = [];
    return;
end

[nBins, nStates] = size(pStateWindow);
[maxProb, ~] = max(pStateWindow, [], 2);

switch lower(probMetric)
    case 'maxprob'
        metricVals = maxProb;
    case 'margin'
        if nStates < 2
            metricVals = ones(nBins, 1);
        else
            sortedProbs = sort(pStateWindow, 2, 'descend');
            metricVals = sortedProbs(:, 1) - sortedProbs(:, 2);
        end
    case 'entropy'
        epsVal = 1e-12;
        entropyVals = -sum(pStateWindow .* log(pStateWindow + epsVal), 2);
        if nStates > 1
            metricVals = entropyVals ./ log(nStates);
        else
            metricVals = zeros(nBins, 1);
        end
    case 'dominance'
        epsVal = 1e-12;
        entropyVals = -sum(pStateWindow .* log(pStateWindow + epsVal), 2);
        if nStates > 1
            metricVals = 1 - (entropyVals ./ log(nStates));
        else
            metricVals = ones(nBins, 1);
        end
    otherwise
        warning('Unknown probMetric "%s"; using maxProb.', probMetric);
        metricVals = maxProb;
end

metricVals = metricVals(:)';

end
