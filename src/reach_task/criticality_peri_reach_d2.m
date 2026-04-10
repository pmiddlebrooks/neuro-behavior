%%
% criticality_peri_reach_d2
% Peri-reach sliding-window d2: how criticality changes before vs after reach onsets.
%
% Align all windows to reach start times (seconds). Two analyses:
%   Pre  — centers in (-prePostSec+W/2, -W/2) so winEnd < reach (no reach in window).
%   Post — centers in (W/2, prePostSec-W/2) so winStart > reach (no reach in window).
%
% Variables:
%   sessionType, sessionName — same as criticality_reach_intertrial_d2.m
%   minITI — include a reach for pre only if (reach - prevReach) >= minITI;
%            for post only if (nextReach - reach) >= minITI.
%   prePostSec — extent of sliding coverage before/after each reach (seconds).
%   slidingWindowSize, stepSize, windowBuffer — windowing (seconds).
%   pOrder, critType, normalizeD2, nShuffles — d2
%   useLog10D2 — if true, plot/analyze log10(d2), d2<=0 -> NaN
%
% Reach-free windows: d2 is computed only if (i) [winStart,winEnd] contains no
% reach onset in the session, (ii) the window lies fully inside the recording
% (with windowBuffer), and (iii) the binned window is full-length (indices not
% clamped). Mean/SEM at each slide position use only finite d2 (reach-free trials).
%
% Plots: one figure, row 1 = pre, row 2 = post, columns = areas.
%   Mean d2 vs relative time (window center - reach), SEM error bars.
%   Vertical dashed lines at -W/2 and +W/2: reference distance from reach onset
%   (nearest slide centers lie just inside these bounds).
%   Faint gray xlines: per-reach relative times of previous and next reach (distribution).

%% ============================= Data loading =============================
fprintf('\n=== criticality_peri_reach_d2: load reach spikes ===\n');

if ~exist('sessionType', 'var')
    error('sessionType must be defined. Run choose_task_and_session.m first or set sessionType in workspace.');
end
if ~exist('sessionName', 'var')
    error('sessionName must be defined. Run choose_task_and_session.m first or set sessionName in workspace.');
end

opts = neuro_behavior_options;
opts.frameSize = 0.001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = [];
opts.minFiringRate = 0.1;
opts.maxFiringRate = 100;

dataStruct = load_sliding_window_data(sessionType, 'spikes', ...
    'sessionName', sessionName, 'opts', opts);

if ~isfield(dataStruct, 'dataR')
    error('dataStruct.dataR missing; need reach times.');
end

dataR = dataStruct.dataR;
reachStart = dataR.R(:, 1) / 1000;  % ms -> s
nReach = numel(reachStart);

areas = dataStruct.areas;
numAreas = numel(areas);
if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:numAreas;
end

saveDir = dataStruct.saveDir;
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'sliding_window_prep', 'utils'));

if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
    timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
else
    timeRange = [0, max(dataStruct.spikeTimes)];
end
tRecStart = timeRange(1);
tRecEnd = timeRange(2);

%% ============================= Configuration =============================
minITI = 1.5;           % s; pre requires gap from prev reach, post requires gap to next
prePostSec = 8;         % s; sliding extent before / after reach
slidingWindowSize = 2;  % s
stepSize = 0.25;        % s
windowBuffer = 0.5;     % s inside recording bounds

pOrder = 10;
critType = 2;
normalizeD2 = false;
nShuffles = 3;

useOptimalBinWindowFunction = false;
binSizeManual = 0.025;
minSpikesPerBin = 3;
minBinsPerWindow = 1000;

pcaFlag = 0;

useLog10D2 = true;

makePlots = true;

halfW = slidingWindowSize / 2;

%% Relative sliding positions (window center relative to reach onset at 0)
% Require strict separation from aligned reach: pre has winEnd < reach, post has winStart > reach.
relCentersPre = (-prePostSec + halfW):stepSize:(-halfW);
relCentersPre = relCentersPre(relCentersPre < -halfW);
relCentersPost = (halfW):stepSize:(prePostSec - halfW);
relCentersPost = relCentersPost(relCentersPost > halfW);
if isempty(relCentersPre)
    error('prePostSec too small for slidingWindowSize (pre side empty after reach-free filter).');
end
if isempty(relCentersPost)
    error('prePostSec too small for slidingWindowSize (post side empty after reach-free filter).');
end

nPre = numel(relCentersPre);
nPost = numel(relCentersPost);

% First relative center where window does NOT contain reach at 0 (point reach)
% Containment: center in [-halfW, halfW]. Pre side: first rel < -halfW when scanning
% from 0 backward is boundary at -halfW. Post: +halfW.
boundaryPreRel = -halfW;
boundaryPostRel = halfW;

%% Reach eligibility
eligiblePre = false(nReach, 1);
eligiblePost = false(nReach, 1);
itiPrev = nan(nReach, 1);
itiNext = nan(nReach, 1);

for r = 1:nReach
    if r > 1
        itiPrev(r) = reachStart(r) - reachStart(r - 1);
        if itiPrev(r) >= minITI
            eligiblePre(r) = true;
        end
    end
    if r < nReach
        itiNext(r) = reachStart(r + 1) - reachStart(r);
        if itiNext(r) >= minITI
            eligiblePost(r) = true;
        end
    end
end

fprintf('Reaches: %d | pre-eligible (minITI from prev): %d | post-eligible (minITI to next): %d\n', ...
    nReach, sum(eligiblePre), sum(eligiblePost));

%% Bin sizes
binSize = zeros(1, numAreas);
if useOptimalBinWindowFunction
    for a = areasToTest
        neuronIDs = dataStruct.idLabel{a};
        fr = calculate_firing_rate_from_spikes(dataStruct.spikeTimes, ...
            dataStruct.spikeClusters, neuronIDs, timeRange);
        [binSize(a), ~] = find_optimal_bin_and_window(fr, minSpikesPerBin, minBinsPerWindow);
    end
else
    binSize(:) = binSizeManual;
end

%% Storage: d2Pre{d2Post}{a} = [nPos x nReach] then mask NaN
d2Pre = cell(1, numAreas);
d2Post = cell(1, numAreas);
for a = areasToTest
    d2Pre{a} = nan(nPre, nReach);
    d2Post{a} = nan(nPost, nReach);
end

%% ============================= Per-area loop =============================
for a = areasToTest
    fprintf('\nArea %s: peri-reach d2...\n', areas{a});
    neuronIDs = dataStruct.idLabel{a};
    aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange, binSize(a));
    nT = size(aDataMat, 1);

    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(aDataMat);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, min(6, forDim));
        nDimUse = 1:forDim;
        aDataMat = score(:, nDimUse) * coeff(:, nDimUse)' + mu;
    end

    for r = 1:nReach
        tReach = reachStart(r);

        %% Pre
        if eligiblePre(r)
            for k = 1:nPre
                relC = relCentersPre(k);
                centerTime = tReach + relC;
                winStart = centerTime - halfW;
                winEnd = centerTime + halfW;
                if winStart < tRecStart + windowBuffer || winEnd > tRecEnd - windowBuffer
                    continue;
                end
                if window_contains_any_reach(winStart, winEnd, reachStart)
                    continue;
                end

                [i0, i1, idxOk] = window_indices_strict(centerTime, slidingWindowSize, ...
                    binSize(a), nT);
                if ~idxOk
                    continue;
                end
                wMat = aDataMat(i0:i1, :);
                d2v = compute_d2_for_window_matrix_local(wMat, pOrder, critType, ...
                    normalizeD2, nShuffles);
                d2Pre{a}(k, r) = d2v;
            end
        end

        %% Post
        if eligiblePost(r)
            for k = 1:nPost
                relC = relCentersPost(k);
                centerTime = tReach + relC;
                winStart = centerTime - halfW;
                winEnd = centerTime + halfW;
                if winStart < tRecStart + windowBuffer || winEnd > tRecEnd - windowBuffer
                    continue;
                end
                if window_contains_any_reach(winStart, winEnd, reachStart)
                    continue;
                end

                [i0, i1, idxOk] = window_indices_strict(centerTime, slidingWindowSize, ...
                    binSize(a), nT);
                if ~idxOk
                    continue;
                end
                wMat = aDataMat(i0:i1, :);
                d2v = compute_d2_for_window_matrix_local(wMat, pOrder, critType, ...
                    normalizeD2, nShuffles);
                d2Post{a}(k, r) = d2v;
            end
        end
    end
end

%% Optional log10
if useLog10D2
    for a = areasToTest
        d2Pre{a} = apply_log10_safe_matrix(d2Pre{a});
        d2Post{a} = apply_log10_safe_matrix(d2Post{a});
    end
end

if useLog10D2
    yLabelStr = 'log_{10}(d2)';
else
    yLabelStr = 'd2';
end

%% ============================= Summary + plot =============================
% Per slide position: nanmean / SEM use only finite d2 (valid, reach-free, full-bin
% windows). NaNs at (k,r) are omitted from n; ineligible reaches are excluded by mask.
meanPre = nan(nPre, numAreas);
semPre = nan(nPre, numAreas);
meanPost = nan(nPost, numAreas);
semPost = nan(nPost, numAreas);

for a = areasToTest
    for k = 1:nPre
        rowv = d2Pre{a}(k, eligiblePre);
        meanPre(k, a) = nanmean(rowv);
        nn = sum(isfinite(rowv));
        if nn > 1
            semPre(k, a) = nanstd(rowv) / sqrt(nn);
        end
    end
    for k = 1:nPost
        rowv = d2Post{a}(k, eligiblePost);
        meanPost(k, a) = nanmean(rowv);
        nn = sum(isfinite(rowv));
        if nn > 1
            semPost(k, a) = nanstd(rowv) / sqrt(nn);
        end
    end
end

relPrevAll = nan(nReach, 1);
relNextAll = nan(nReach, 1);
for r = 1:nReach
    if r > 1
        relPrevAll(r) = reachStart(r - 1) - reachStart(r);
    end
    if r < nReach
        relNextAll(r) = reachStart(r + 1) - reachStart(r);
    end
end

if makePlots
    nCol = numel(areasToTest);
    figure(4101);
    clf;
    set(gcf, 'Color', 'w', 'Name', 'Peri-reach d2', 'NumberTitle', 'off');
    monitorPositions = get(0, 'MonitorPositions');
    if size(monitorPositions, 1) >= 2
        set(gcf, 'Position', monitorPositions(end, :));
    else
        set(gcf, 'Position', monitorPositions(1, :));
    end

    useTight = exist('tight_subplot', 'file');
    if useTight
        ha = tight_subplot(2, nCol, [0.07 0.05], [0.1 0.08], [0.06 0.04]);
    else
        ha = gobjects(2 * nCol, 1);
        for ii = 1:(2 * nCol)
            ha(ii) = subplot(2, nCol, ii);
        end
    end

    for colIdx = 1:nCol
        a = areasToTest(colIdx);
        axPre = ha(colIdx);
        axPost = ha(nCol + colIdx);
        %% Pre panel
        axes(axPre);
        hold on;
        plot_gray_reach_marks(relPrevAll(eligiblePre), relNextAll(eligiblePre), relCentersPre);
        errorbar(relCentersPre, meanPre(:, a), semPre(:, a), 'k.', 'LineWidth', 1);
        plot(relCentersPre, meanPre(:, a), 'b-', 'LineWidth', 1.2);
        xline(boundaryPreRel, 'k--', 'LineWidth', 1.5);
        xline(0, 'k:', 'LineWidth', 1);
        xlabel('Window center - reach (s)');
        ylabel(yLabelStr);
        title(sprintf('%s pre-reach', areas{a}), 'Interpreter', 'none');
        grid on;
        hold off;

        %% Post panel
        axes(axPost);
        hold on;
        plot_gray_reach_marks(relPrevAll(eligiblePost), relNextAll(eligiblePost), relCentersPost);
        errorbar(relCentersPost, meanPost(:, a), semPost(:, a), 'k.', 'LineWidth', 1);
        plot(relCentersPost, meanPost(:, a), 'r-', 'LineWidth', 1.2);
        xline(boundaryPostRel, 'k--', 'LineWidth', 1.5);
        xline(0, 'k:', 'LineWidth', 1);
        xlabel('Window center - reach (s)');
        ylabel(yLabelStr);
        title(sprintf('%s post-reach', areas{a}), 'Interpreter', 'none');
        grid on;
        hold off;
    end

    sgtitle(sprintf(['Peri-reach d2 | %s | W=%.2fs step=%.2fs minITI=%.2fs prePost=%.2fs'], ...
        sessionName, slidingWindowSize, stepSize, minITI, prePostSec), 'Interpreter', 'none');

    outPng = fullfile(saveDir, sprintf('criticality_peri_reach_d2_%s_W%.1f.png', sessionName, slidingWindowSize));
    exportgraphics(gcf, outPng, 'Resolution', 300);
    fprintf('Saved figure: %s\n', outPng);
end

%% Save
results = struct();
results.sessionName = sessionName;
results.areas = areas;
results.areasToTest = areasToTest;
results.reachStart = reachStart;
results.minITI = minITI;
results.prePostSec = prePostSec;
results.slidingWindowSize = slidingWindowSize;
results.stepSize = stepSize;
results.windowBuffer = windowBuffer;
results.relCentersPre = relCentersPre;
results.relCentersPost = relCentersPost;
results.boundaryPreRel = boundaryPreRel;
results.boundaryPostRel = boundaryPostRel;
results.eligiblePre = eligiblePre;
results.eligiblePost = eligiblePost;
results.d2Pre = d2Pre;
results.d2Post = d2Post;
results.meanPre = meanPre;
results.semPre = semPre;
results.meanPost = meanPost;
results.semPost = semPost;
results.binSize = binSize;
results.params.pOrder = pOrder;
results.params.critType = critType;
results.params.normalizeD2 = normalizeD2;
results.params.useLog10D2 = useLog10D2;

resultsPath = fullfile(saveDir, sprintf('criticality_peri_reach_d2_W%.1f_step%.2f.mat', ...
    slidingWindowSize, stepSize));
save(resultsPath, 'results');
fprintf('Saved results: %s\n', resultsPath);

%% ============================= Local functions =============================
function tf = window_contains_any_reach(winStart, winEnd, reachStartAll)
% window_contains_any_reach
% Goal: true if any reach onset lies in [winStart, winEnd] (closed interval).
% Variables: winStart, winEnd (s); reachStartAll — column vector of reach times (s).

    tf = any(reachStartAll >= winStart & reachStartAll <= winEnd);
end

function [startIdx, endIdx, ok] = window_indices_strict(centerTime, slidingWindowSize, binSize, numTimePoints)
% window_indices_strict
% Goal: same index mapping as calculate_window_indices_from_center, but ok is false
% if the window would be clamped (incomplete / off-center span at recording edges).
% Variables: centerTime, slidingWindowSize, binSize (s), numTimePoints (bins).
% Returns: startIdx, endIdx (1-based), ok — logical.

    centerIdx = round(centerTime / binSize) + 1;
    winSamples = round(slidingWindowSize / binSize);
    if winSamples < 1
        winSamples = 1;
    end
    halfWin = round(winSamples / 2);
    startIdx = centerIdx - halfWin + 1;
    endIdx = startIdx + winSamples - 1;
    ok = startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx;
end

function plot_gray_reach_marks(relPrevVec, relNextVec, xLimRef)
% plot_gray_reach_marks
% Goal: show when prior/next reaches occur relative to alignment (light xlines).
% Variables: relPrevVec, relNextVec — relative times (s); xLimRef — axis span for clipping.

    if nargin < 3 || isempty(xLimRef)
        return;
    end
    xl = [min(xLimRef), max(xLimRef)];
    for u = 1:numel(relPrevVec)
        v = relPrevVec(u);
        if isfinite(v) && v >= xl(1) && v <= xl(2)
            xline(v, 'Color', [0.88 0.88 0.88], 'LineWidth', 0.5, ...
                'HandleVisibility', 'off');
        end
    end
    for u = 1:numel(relNextVec)
        v = relNextVec(u);
        if isfinite(v) && v >= xl(1) && v <= xl(2)
            xline(v, 'Color', [0.88 0.88 0.88], 'LineWidth', 0.5, ...
                'HandleVisibility', 'off');
        end
    end
end

function Mout = apply_log10_safe_matrix(Min)
% apply_log10_safe_matrix — log10(Min) with nonpositive/nonfinite -> NaN.

    Mout = nan(size(Min));
    ok = isfinite(Min) & Min > 0;
    Mout(ok) = log10(Min(ok));
end

function d2Val = compute_d2_for_window_matrix_local(wDataMat, pOrder, critType, normalizeD2, nShuffles)
% compute_d2_for_window_matrix_local
% Goal: d2 from one window of binned population activity [timeBins x neurons].
% Variables:
%   wDataMat — binned spikes (or PCA-reconstructed activity) for the window
%   pOrder, critType — passed to myYuleWalker3 / getFixedPointDistance2
%   normalizeD2 — if true, divide raw d2 by mean d2 from circularly shuffled controls
%   nShuffles — number of shuffle draws for normalization
% Returns:
%   d2Val — scalar d2 (or NaN)

    d2Val = nan;
    if isempty(wDataMat) || size(wDataMat, 1) < pOrder + 2
        return;
    end

    wPopActivity = mean(wDataMat, 2);

    try
        [varphi, ~] = myYuleWalker3(double(wPopActivity), pOrder);
        d2Raw = getFixedPointDistance2(pOrder, critType, varphi);
    catch
        d2Raw = nan;
    end

    if ~normalizeD2
        d2Val = d2Raw;
        return;
    end
    if isnan(d2Raw)
        return;
    end

    nNeu = size(wDataMat, 2);
    nBins = size(wDataMat, 1);
    d2Sh = nan(1, nShuffles);
    for sh = 1:nShuffles
        permutedMat = zeros(size(wDataMat));
        for n = 1:nNeu
            permutedMat(:, n) = circshift(wDataMat(:, n), randi(nBins));
        end
        permPop = mean(permutedMat, 2);
        try
            [varphiPerm, ~] = myYuleWalker3(double(permPop), pOrder);
            d2Sh(sh) = getFixedPointDistance2(pOrder, critType, varphiPerm);
        catch
            d2Sh(sh) = nan;
        end
    end
    meanSh = nanmean(d2Sh);
    if isfinite(meanSh) && meanSh > 0
        d2Val = d2Raw / meanSh;
    end
end
