%%
% criticality_peri_reach_d2
% Peri-reach sliding-window d2: how criticality changes before vs after reach onsets.
%
% Align all windows to reach start times (seconds). Two analyses:
%   Pre  — strict: centers < -W/2 so winEnd < reach (no reach in window). Additional
%         overlap bins: centers in [-W/2, 0] so the sliding window can include reach
%         onset (d2 and pop activity still exclude other reaches; aligned reach allowed).
%   Post — strict: centers > W/2 so winStart > reach. Additional overlap bins: centers
%         in (0, W/2] straddling or ending past reach onset.
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
% Valid windows: recording bounds + windowBuffer, full binned length, and corridor
% between neighbors. Strict bins also require no reach onset in the window. Overlap
% bins include the aligned reach onset but exclude any other reach; pre overlap
% requires winEnd < next reach; post overlap requires winStart > prev reach (when
% it exists). Extreme slide positions can still drop trials, so mean/SEM n can fall
% near corridor edges.
%
% Plots: one figure, row 1 = pre, row 2 = post, columns = areas.
%   Left y-axis: mean d2 vs relative time (window center - reach), SEM error bars.
%   Right y-axis: (1) mean population activity in each full window (mean over bins of
%   summed counts per bin); (2) raw summed population activity at the single bin aligned
%   with each window center (same x = slide centers).
%   Vertical dashed lines at -W/2 and +W/2: last slide centers where the window still
%   touches/overlaps reach onset (strict reach-free bins lie outside these bounds).
%   Faint gray xlines: pre row = previous reach only (itiPrev >= minITI); post row =
%   next reach only (itiNext >= minITI) — not both neighbors on the same panel.

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
minITI = 6;           % s; pre requires gap from prev reach, post requires gap to next
prePostSec = 15;         % s; sliding extent before / after reach
slidingWindowSize = 5;  % s
stepSize = 0.25;        % s
windowBuffer = 0.5;     % s inside recording bounds

pOrder = 10;
critType = 2;
normalizeD2 = false;
nShuffles = 10;
meanSubtract = false;   % If true, subtract bin-wise mean across windows at each slide position

useOptimalBinWindowFunction = false;
binSizeManual = 0.025;
minSpikesPerBin = 3;
minBinsPerWindow = 1000;

pcaFlag = 0;

useLog10D2 = true;

makePlots = true;
plotJoystick = true;  % If true, overlay peri-reach joystick metrics on right y-axis

halfW = slidingWindowSize / 2;

% Relative sliding positions (window center relative to reach onset at 0).
% Strict bins: no aligned reach inside the window. Overlap bins: window contains reach.
relCentersPreStrict = (-prePostSec + halfW):stepSize:(-halfW);
relCentersPreStrict = relCentersPreStrict(relCentersPreStrict < -halfW);
relCentersPreOverlap = (-halfW):stepSize:0;
relCentersPre = sort(unique([relCentersPreStrict(:); relCentersPreOverlap(:)]'));

relCentersPostStrict = (halfW):stepSize:(prePostSec - halfW);
relCentersPostStrict = relCentersPostStrict(relCentersPostStrict > halfW);
relCentersPostOverlap = stepSize:stepSize:halfW;
relCentersPost = sort(unique([relCentersPostOverlap(:); relCentersPostStrict(:)]'));

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

% Reach eligibility
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

% Bin sizes
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

% Storage: d2Pre/d2Post, popActPre/Post (window-mean pop), popCenterRawPre/Post (summed
% population count at the bin aligned with each window center) — all [nPos x nReach].
d2Pre = cell(1, numAreas);
d2Post = cell(1, numAreas);
popActPre = cell(1, numAreas);
popActPost = cell(1, numAreas);
popCenterRawPre = cell(1, numAreas);
popCenterRawPost = cell(1, numAreas);
jsCenterRawPre = cell(1, numAreas);
jsCenterRawPost = cell(1, numAreas);
for a = areasToTest
    d2Pre{a} = nan(nPre, nReach);
    d2Post{a} = nan(nPost, nReach);
    popActPre{a} = nan(nPre, nReach);
    popActPost{a} = nan(nPost, nReach);
    popCenterRawPre{a} = nan(nPre, nReach);
    popCenterRawPost{a} = nan(nPost, nReach);
    jsCenterRawPre{a} = nan(nPre, nReach);
    jsCenterRawPost{a} = nan(nPost, nReach);
end

jsSampleRate = 1000;  % Hz; NIDATA is sampled at 1 kHz in reach task files
jsSmoothWindow = 61;  % samples; matches explore_session.m
hasJoystickTrace = isfield(dataR, 'NIDATA') && size(dataR.NIDATA, 1) >= 8 && size(dataR.NIDATA, 2) > 1;
if plotJoystick && hasJoystickTrace
    jsXIdx = 7;
    jsYIdx = 8;
    jsAmpTrace = sqrt(dataR.NIDATA(jsXIdx, 2:end).^2 + dataR.NIDATA(jsYIdx, 2:end).^2);
    jsAmpTrace = movmean(jsAmpTrace, jsSmoothWindow);
    jsAmpTrace = zscore(jsAmpTrace);
else
    jsAmpTrace = [];
    if plotJoystick
        warning('plotJoystick is true but NIDATA joystick channels are unavailable. Skipping joystick overlay.');
    end
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

    prePopTraceForD2 = cell(nPre, nReach);    % each entry: [timeBins x 1], population mean trace
    postPopTraceForD2 = cell(nPost, nReach);  % each entry: [timeBins x 1], population mean trace
    preWindowMatForShuffle = cell(nPre, nReach);   % each entry: [timeBins x neurons]
    postWindowMatForShuffle = cell(nPost, nReach); % each entry: [timeBins x neurons]
    preCenterRowForD2 = cell(nPre, nReach);   % each entry: center-bin row index for the window
    postCenterRowForD2 = cell(nPost, nReach); % each entry: center-bin row index for the window

    for r = 1:nReach
        tReach = reachStart(r);

        % Pre
        if eligiblePre(r)
            tPrevReach = reachStart(r - 1);
            tNextReach = inf;
            if r < nReach
                tNextReach = reachStart(r + 1);
            end
            for k = 1:nPre
                relC = relCentersPre(k);
                centerTime = tReach + relC;
                winStart = centerTime - halfW;
                winEnd = centerTime + halfW;
                if winStart < tRecStart + windowBuffer || winEnd > tRecEnd - windowBuffer
                    continue;
                end
                isStrictPreBin = relC < -halfW;
                if isStrictPreBin
                    if ~(winStart > tPrevReach && winEnd < tReach)
                        continue;
                    end
                    if window_contains_any_reach(winStart, winEnd, reachStart)
                        continue;
                    end
                else
                    if ~(winStart > tPrevReach && winEnd < tNextReach)
                        continue;
                    end
                    if ~(winEnd >= tReach && winStart < tReach)
                        continue;
                    end
                    if window_contains_other_reaches(winStart, winEnd, reachStart, r)
                        continue;
                    end
                end

                [i0, i1, idxOk] = window_indices_strict(centerTime, slidingWindowSize, ...
                    binSize(a), nT);
                if ~idxOk
                    continue;
                end
                wMat = aDataMat(i0:i1, :);
                popTraceCol = sum(wMat, 2);
                centerIdx = round(centerTime / binSize(a)) + 1;
                cRow = centerIdx - i0 + 1;
                if cRow < 1 || cRow > size(wMat, 1)
                    continue;
                end
                popActPre{a}(k, r) = mean(popTraceCol);
                popCenterRawPre{a}(k, r) = sum(wMat(cRow, :));
                prePopTraceForD2{k, r} = mean(wMat, 2);
                preWindowMatForShuffle{k, r} = wMat;
                preCenterRowForD2{k, r} = cRow;
                if plotJoystick && ~isempty(jsAmpTrace)
                    [jsCenterVal, jsOk] = trace_center_value_local(jsAmpTrace, centerTime, jsSampleRate, 0);
                    if jsOk
                        jsCenterRawPre{a}(k, r) = jsCenterVal;
                    end
                end
            end
        end

        % Post
        if eligiblePost(r)
            tNextReach = reachStart(r + 1);
            for k = 1:nPost
                relC = relCentersPost(k);
                centerTime = tReach + relC;
                winStart = centerTime - halfW;
                winEnd = centerTime + halfW;
                if winStart < tRecStart + windowBuffer || winEnd > tRecEnd - windowBuffer
                    continue;
                end
                isStrictPostBin = relC > halfW;
                if isStrictPostBin
                    if ~(winStart > tReach && winEnd < tNextReach)
                        continue;
                    end
                    if window_contains_any_reach(winStart, winEnd, reachStart)
                        continue;
                    end
                else
                    if r > 1 && winStart <= reachStart(r - 1)
                        continue;
                    end
                    if ~(winStart > tRecStart + windowBuffer && winEnd < tNextReach)
                        continue;
                    end
                    if ~(winStart <= tReach && winEnd >= tReach)
                        continue;
                    end
                    if window_contains_other_reaches(winStart, winEnd, reachStart, r)
                        continue;
                    end
                end

                [i0, i1, idxOk] = window_indices_strict(centerTime, slidingWindowSize, ...
                    binSize(a), nT);
                if ~idxOk
                    continue;
                end
                wMat = aDataMat(i0:i1, :);
                popTraceCol = sum(wMat, 2);
                centerIdx = round(centerTime / binSize(a)) + 1;
                cRow = centerIdx - i0 + 1;
                if cRow < 1 || cRow > size(wMat, 1)
                    continue;
                end
                popActPost{a}(k, r) = mean(popTraceCol);
                popCenterRawPost{a}(k, r) = sum(wMat(cRow, :));
                postPopTraceForD2{k, r} = mean(wMat, 2);
                postWindowMatForShuffle{k, r} = wMat;
                postCenterRowForD2{k, r} = cRow;
                if plotJoystick && ~isempty(jsAmpTrace)
                    [jsCenterVal, jsOk] = trace_center_value_local(jsAmpTrace, centerTime, jsSampleRate, 0);
                    if jsOk
                        jsCenterRawPost{a}(k, r) = jsCenterVal;
                    end
                end
            end
        end
    end

    % Compute d2 per slide position after collecting all window traces so that
    % optional mean subtraction uses the bin-wise mean across windows.
    d2Pre{a} = compute_d2_by_position_local(prePopTraceForD2, preWindowMatForShuffle, ...
        pOrder, critType, normalizeD2, nShuffles, meanSubtract);
    d2Post{a} = compute_d2_by_position_local(postPopTraceForD2, postWindowMatForShuffle, ...
        pOrder, critType, normalizeD2, nShuffles, meanSubtract);

    % If mean subtraction is active, plot population metrics from mean-subtracted
    % traces (window-mean and center-bin values) instead of raw traces.
    if meanSubtract
        [popActPre{a}, popCenterRawPre{a}] = compute_mean_subtracted_pop_metrics_local( ...
            prePopTraceForD2, preCenterRowForD2);
        [popActPost{a}, popCenterRawPost{a}] = compute_mean_subtracted_pop_metrics_local( ...
            postPopTraceForD2, postCenterRowForD2);
    end
end

% Optional log10
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
% Per slide position: nanmean / SEM use only finite d2 at (k,r) after all window
% gates (recording bounds, strict inter-reach corridor, no reach in window, strict
% bin indices). NaNs omit that trial from n at that k; ineligible reaches never enter the loop.
meanPre = nan(nPre, numAreas);
semPre = nan(nPre, numAreas);
meanPost = nan(nPost, numAreas);
semPost = nan(nPost, numAreas); 
meanPopPre = nan(nPre, numAreas);
semPopPre = nan(nPre, numAreas);
meanPopPost = nan(nPost, numAreas);
semPopPost = nan(nPost, numAreas);
meanPopCenterRawPre = nan(nPre, numAreas);
semPopCenterRawPre = nan(nPre, numAreas);
meanPopCenterRawPost = nan(nPost, numAreas);
semPopCenterRawPost = nan(nPost, numAreas);
meanJsCenterRawPre = nan(nPre, numAreas);
semJsCenterRawPre = nan(nPre, numAreas);
meanJsCenterRawPost = nan(nPost, numAreas);
semJsCenterRawPost = nan(nPost, numAreas);

for a = areasToTest
    for k = 1:nPre
        rowv = d2Pre{a}(k, eligiblePre);
        meanPre(k, a) = nanmean(rowv);
        nn = sum(isfinite(rowv));
        if nn > 1
            semPre(k, a) = nanstd(rowv) / sqrt(nn);
        end
        rowPop = popActPre{a}(k, eligiblePre);
        meanPopPre(k, a) = nanmean(rowPop);
        nnPop = sum(isfinite(rowPop));
        if nnPop > 1
            semPopPre(k, a) = nanstd(rowPop) / sqrt(nnPop);
        end
        rowCenter = popCenterRawPre{a}(k, eligiblePre);
        meanPopCenterRawPre(k, a) = nanmean(rowCenter);
        nnC = sum(isfinite(rowCenter));
        if nnC > 1
            semPopCenterRawPre(k, a) = nanstd(rowCenter) / sqrt(nnC);
        end
        rowJsCenter = jsCenterRawPre{a}(k, eligiblePre);
        meanJsCenterRawPre(k, a) = nanmean(rowJsCenter);
        nnJsC = sum(isfinite(rowJsCenter));
        if nnJsC > 1
            semJsCenterRawPre(k, a) = nanstd(rowJsCenter) / sqrt(nnJsC);
        end
    end
    for k = 1:nPost
        rowv = d2Post{a}(k, eligiblePost);
        meanPost(k, a) = nanmean(rowv);
        nn = sum(isfinite(rowv));
        if nn > 1
            semPost(k, a) = nanstd(rowv) / sqrt(nn);
        end
        rowPop = popActPost{a}(k, eligiblePost);
        meanPopPost(k, a) = nanmean(rowPop);
        nnPop = sum(isfinite(rowPop));
        if nnPop > 1
            semPopPost(k, a) = nanstd(rowPop) / sqrt(nnPop);
        end
        rowCenter = popCenterRawPost{a}(k, eligiblePost);
        meanPopCenterRawPost(k, a) = nanmean(rowCenter);
        nnC = sum(isfinite(rowCenter));
        if nnC > 1
            semPopCenterRawPost(k, a) = nanstd(rowCenter) / sqrt(nnC);
        end
        rowJsCenter = jsCenterRawPost{a}(k, eligiblePost);
        meanJsCenterRawPost(k, a) = nanmean(rowJsCenter);
        nnJsC = sum(isfinite(rowJsCenter));
        if nnJsC > 1
            semJsCenterRawPost(k, a) = nanstd(rowJsCenter) / sqrt(nnJsC);
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

    % Shared y-limits: mean ± SEM across areas under test (pre + post).
    areaCols = areasToTest(:);
    bandLo = [meanPre(:, areaCols) - semPre(:, areaCols); ...
        meanPost(:, areaCols) - semPost(:, areaCols)];
    bandHi = [meanPre(:, areaCols) + semPre(:, areaCols); ...
        meanPost(:, areaCols) + semPost(:, areaCols)];
    yMinData = nanmin(bandLo(:));
    yMaxData = nanmax(bandHi(:));
    if ~isfinite(yMinData) || ~isfinite(yMaxData)
        yLimGlobal = [0, 1];
    elseif yMaxData <= yMinData
        yLimGlobal = [yMinData - 0.05, yMaxData + 0.05];
    else
        yPad = 0.05 * (yMaxData - yMinData);
        yLimGlobal = [yMinData - yPad, yMaxData + yPad];
    end

    % X includes 0 on both rows so the reach-onset reference is on every axis.
    xPreLim = [min(relCentersPre), 0];
    xPostLim = [0, max(relCentersPost)];
    xLabelStr = 'Window center - reach onset (s); 0 = reach start';
    if useLog10D2
        yLabInterpreter = 'tex';
    else
        yLabInterpreter = 'none';
    end

    nTickX = 6;
    xTicksPre = linspace(xPreLim(1), xPreLim(2), nTickX);
    xTicksPre(end) = 0;
    xTickLabelsPre = cell(nTickX, 1);
    for tt = 1:nTickX
        if abs(xTicksPre(tt)) < 1e-10
            xTickLabelsPre{tt} = '0';
        else
            xTickLabelsPre{tt} = sprintf('%.1f', xTicksPre(tt));
        end
    end
    xTicksPost = linspace(xPostLim(1), xPostLim(2), nTickX);
    xTicksPost(1) = 0;
    xTickLabelsPost = cell(nTickX, 1);
    for tt = 1:nTickX
        if abs(xTicksPost(tt)) < 1e-10
            xTickLabelsPost{tt} = '0';
        else
            xTickLabelsPost{tt} = sprintf('%.1f', xTicksPost(tt));
        end
    end

    nTickY = 5;
    yTicksGlobal = linspace(yLimGlobal(1), yLimGlobal(2), nTickY);
    yTickLabelsGlobal = cell(nTickY, 1);
    for tt = 1:nTickY
        yTickLabelsGlobal{tt} = sprintf('%.2f', yTicksGlobal(tt));
    end

    % Right axis: mean population activity (mean ± SEM across tested areas, pre + post).
    bandPopLo = [meanPopPre(:, areaCols) - semPopPre(:, areaCols); ...
        meanPopPost(:, areaCols) - semPopPost(:, areaCols)];
    bandPopHi = [meanPopPre(:, areaCols) + semPopPre(:, areaCols); ...
        meanPopPost(:, areaCols) + semPopPost(:, areaCols)];
    bandCenterLo = [meanPopCenterRawPre(:, areaCols) - semPopCenterRawPre(:, areaCols); ...
        meanPopCenterRawPost(:, areaCols) - semPopCenterRawPost(:, areaCols)];
    bandCenterHi = [meanPopCenterRawPre(:, areaCols) + semPopCenterRawPre(:, areaCols); ...
        meanPopCenterRawPost(:, areaCols) + semPopCenterRawPost(:, areaCols)];
    bandAllLo = [bandPopLo(:); bandCenterLo(:)];
    bandAllHi = [bandPopHi(:); bandCenterHi(:)];
    if plotJoystick && ~isempty(jsAmpTrace)
        bandJsCenterLo = [meanJsCenterRawPre(:, areaCols) - semJsCenterRawPre(:, areaCols); ...
            meanJsCenterRawPost(:, areaCols) - semJsCenterRawPost(:, areaCols)];
        bandJsCenterHi = [meanJsCenterRawPre(:, areaCols) + semJsCenterRawPre(:, areaCols); ...
            meanJsCenterRawPost(:, areaCols) + semJsCenterRawPost(:, areaCols)];
        bandAllLo = [bandAllLo; bandJsCenterLo(:)];
        bandAllHi = [bandAllHi; bandJsCenterHi(:)];
    end
    yMinPop = nanmin(bandAllLo);
    yMaxPop = nanmax(bandAllHi);
    if ~isfinite(yMinPop) || ~isfinite(yMaxPop)
        yLimPopGlobal = [0, 1];
    elseif yMaxPop <= yMinPop
        yLimPopGlobal = [yMinPop - 0.05, yMaxPop + 0.05];
    else
        yPadPop = 0.05 * (yMaxPop - yMinPop);
        yLimPopGlobal = [yMinPop - yPadPop, yMaxPop + yPadPop];
    end
    yTicksPopGlobal = linspace(yLimPopGlobal(1), yLimPopGlobal(2), nTickY);
    yTickLabelsPopGlobal = cell(nTickY, 1);
    for tt = 1:nTickY
        yTickLabelsPopGlobal{tt} = sprintf('%.2f', yTicksPopGlobal(tt));
    end
    if plotJoystick && ~isempty(jsAmpTrace)
        popActYLabelStr = 'Population / joystick activity';
    else
        popActYLabelStr = 'Population activity (spks/bin)';
    end

    figure(4102);
    clf;
    set(gcf, 'Color', 'w', 'Name', 'Peri-reach d2', 'NumberTitle', 'off');
    monitorPositions = get(0, 'MonitorPositions');
    if size(monitorPositions, 1) >= 2
        set(gcf, 'Position', monitorPositions(end, :));
    else
        set(gcf, 'Position', monitorPositions(1, :));
    end

    useTight = exist('tight_subplot', 'file');
    nRowFig = 2;
    if useTight
        ha = tight_subplot(nRowFig, nCol, [0.075 0.075], [0.1 0.08], [0.06 0.04]);
    else
        ha = gobjects(nRowFig * nCol, 1);
        for ii = 1:(nRowFig * nCol)
            ha(ii) = subplot(nRowFig, nCol, ii);
        end
    end

    for colIdx = 1:nCol
        a = areasToTest(colIdx);
        axPre = ha(colIdx);
        axPost = ha(nCol + colIdx);
        % Pre panel (left: d2; right: mean population activity in each window)
        axes(axPre);
        yyaxis left;
        hold on;
        % Only previous-reach times: eligiblePre enforces itiPrev >= minITI, not itiNext.
        plot_gray_reach_marks(relPrevAll(eligiblePre), [], relCentersPre);
        errorbar(relCentersPre, meanPre(:, a), semPre(:, a), 'k.', 'LineWidth', 1);
        plot(relCentersPre, meanPre(:, a), 'b-', 'LineWidth', 1.2);
        xline(boundaryPreRel, 'k--', 'LineWidth', 1.5);
        xline(0, 'k:', 'LineWidth', 1);
        yyaxis right;
        hold on;
        errorbar(relCentersPre, meanPopPre(:, a), semPopPre(:, a), '.', 'Color', [0.55 0.55 0.55], ...
            'LineWidth', 0.9, 'MarkerSize', 10, 'CapSize', 3);
        plot(relCentersPre, meanPopPre(:, a), '-', 'Color', [0.85 0.4 0.08], 'LineWidth', 1.2);
        errorbar(relCentersPre, meanPopCenterRawPre(:, a), semPopCenterRawPre(:, a), '.', ...
            'Color', [0.15 0.55 0.25], 'LineWidth', 0.85, 'MarkerSize', 8, 'CapSize', 2);
        plot(relCentersPre, meanPopCenterRawPre(:, a), '-', 'Color', [0.1 0.45 0.2], ...
            'LineWidth', 1.1);
        if plotJoystick && ~isempty(jsAmpTrace)
            errorbar(relCentersPre, meanJsCenterRawPre(:, a), semJsCenterRawPre(:, a), '.', ...
                'Color', [0.7 0.55 0.9], 'LineWidth', 0.8, 'MarkerSize', 7, 'CapSize', 2);
            plot(relCentersPre, meanJsCenterRawPre(:, a), '-', 'Color', [0.58 0.42 0.82], ...
                'LineWidth', 1.0);
        end
        hold off;
        yyaxis left;
        title(sprintf('%s pre-reach', areas{a}), 'Interpreter', 'none');
        grid on;
        hold off;
        xlabel(axPre, xLabelStr, 'Interpreter', 'none');
        xlim(axPre, xPreLim);
        set(axPre, 'XTick', xTicksPre, 'XTickLabel', xTickLabelsPre, ...
            'TickLabelInterpreter', 'none', 'XTickLabelMode', 'manual');
        yyaxis left;
        ylabel(axPre, yLabelStr, 'Interpreter', yLabInterpreter);
        ylim(axPre, yLimGlobal);
        set(axPre, 'YTick', yTicksGlobal, 'YTickLabel', yTickLabelsGlobal, ...
            'TickLabelInterpreter', 'none', 'YTickLabelMode', 'manual');
        yyaxis right;
        ylabel(axPre, popActYLabelStr, 'Interpreter', 'none');
        ylim(axPre, yLimPopGlobal);
        set(axPre, 'YTick', yTicksPopGlobal, 'YTickLabel', yTickLabelsPopGlobal, ...
            'TickLabelInterpreter', 'none', 'YTickLabelMode', 'manual');

        % Post panel
        axes(axPost);
        yyaxis left;
        hold on;
        % Only next-reach times: eligiblePost enforces itiNext >= minITI, not itiPrev.
        plot_gray_reach_marks([], relNextAll(eligiblePost), relCentersPost);
        errorbar(relCentersPost, meanPost(:, a), semPost(:, a), 'k.', 'LineWidth', 1);
        plot(relCentersPost, meanPost(:, a), 'r-', 'LineWidth', 1.2);
        xline(boundaryPostRel, 'k--', 'LineWidth', 1.5);
        xline(0, 'k:', 'LineWidth', 1);
        yyaxis right;
        hold on;
        errorbar(relCentersPost, meanPopPost(:, a), semPopPost(:, a), '.', 'Color', [0.55 0.55 0.55], ...
            'LineWidth', 0.9, 'MarkerSize', 10, 'CapSize', 3);
        plot(relCentersPost, meanPopPost(:, a), '-', 'Color', [0.85 0.4 0.08], 'LineWidth', 1.2);
        errorbar(relCentersPost, meanPopCenterRawPost(:, a), semPopCenterRawPost(:, a), '.', ...
            'Color', [0.15 0.55 0.25], 'LineWidth', 0.85, 'MarkerSize', 8, 'CapSize', 2);
        plot(relCentersPost, meanPopCenterRawPost(:, a), '-', 'Color', [0.1 0.45 0.2], ...
            'LineWidth', 1.1);
        if plotJoystick && ~isempty(jsAmpTrace)
            errorbar(relCentersPost, meanJsCenterRawPost(:, a), semJsCenterRawPost(:, a), '.', ...
                'Color', [0.7 0.55 0.9], 'LineWidth', 0.8, 'MarkerSize', 7, 'CapSize', 2);
            plot(relCentersPost, meanJsCenterRawPost(:, a), '-', 'Color', [0.58 0.42 0.82], ...
                'LineWidth', 1.0);
        end
        hold off;
        yyaxis left;
        title(sprintf('%s post-reach', areas{a}), 'Interpreter', 'none');
        grid on;
        hold off;
        xlabel(axPost, xLabelStr, 'Interpreter', 'none');
        xlim(axPost, xPostLim);
        set(axPost, 'XTick', xTicksPost, 'XTickLabel', xTickLabelsPost, ...
            'TickLabelInterpreter', 'none', 'XTickLabelMode', 'manual');
        yyaxis left;
        ylabel(axPost, yLabelStr, 'Interpreter', yLabInterpreter);
        ylim(axPost, yLimGlobal);
        set(axPost, 'YTick', yTicksGlobal, 'YTickLabel', yTickLabelsGlobal, ...
            'TickLabelInterpreter', 'none', 'YTickLabelMode', 'manual');
        yyaxis right;
        ylabel(axPost, popActYLabelStr, 'Interpreter', 'none');
        ylim(axPost, yLimPopGlobal);
        set(axPost, 'YTick', yTicksPopGlobal, 'YTickLabel', yTickLabelsPopGlobal, ...
            'TickLabelInterpreter', 'none', 'YTickLabelMode', 'manual');
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
results.popActPre = popActPre;
results.popActPost = popActPost;
results.popCenterRawPre = popCenterRawPre;
results.popCenterRawPost = popCenterRawPost;
results.jsCenterRawPre = jsCenterRawPre;
results.jsCenterRawPost = jsCenterRawPost;
results.meanPopCenterRawPre = meanPopCenterRawPre;
results.semPopCenterRawPre = semPopCenterRawPre;
results.meanPopCenterRawPost = meanPopCenterRawPost;
results.semPopCenterRawPost = semPopCenterRawPost;
results.meanPopPre = meanPopPre;
results.semPopPre = semPopPre;
results.meanPopPost = meanPopPost;
results.semPopPost = semPopPost;
results.meanJsCenterRawPre = meanJsCenterRawPre;
results.semJsCenterRawPre = semJsCenterRawPre;
results.meanJsCenterRawPost = meanJsCenterRawPost;
results.semJsCenterRawPost = semJsCenterRawPost;
results.meanPre = meanPre;
results.semPre = semPre;
results.meanPost = meanPost;
results.semPost = semPost;
results.binSize = binSize;
results.params.pOrder = pOrder;
results.params.critType = critType;
results.params.normalizeD2 = normalizeD2;
results.params.useLog10D2 = useLog10D2;
results.params.meanSubtract = meanSubtract;
results.params.plotJoystick = plotJoystick;
results.params.jsSampleRate = jsSampleRate;
results.params.jsSmoothWindow = jsSmoothWindow;

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

function tf = window_contains_other_reaches(winStart, winEnd, reachStartAll, idxExcept)
% window_contains_other_reaches
% Goal: true if any reach onset other than idxExcept lies in [winStart, winEnd].
% Variables: winStart, winEnd (s); reachStartAll — reach onsets (s); idxExcept — index
%   of the aligned reach (allowed inside the window for overlap-bin analysis).

    reachCol = reachStartAll(:);
    nR = numel(reachCol);
    mask = true(nR, 1);
    if idxExcept >= 1 && idxExcept <= nR
        mask(idxExcept) = false;
    end
    rs = reachCol(mask);
    tf = any(rs >= winStart & rs <= winEnd);
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

function d2Mat = compute_d2_by_position_local(popTraceCell, windowMatCell, pOrder, critType, ...
    normalizeD2, nShuffles, meanSubtract)
% compute_d2_by_position_local
% Goal: compute d2 matrix [nPos x nReach] from per-window population traces.
% Variables:
%   popTraceCell — cell [nPos x nReach], each entry [timeBins x 1] pop trace
%   windowMatCell — cell [nPos x nReach], each entry [timeBins x neurons] for shuffles
%   pOrder, critType — AR/d2 parameters
%   normalizeD2 — if true, divide each window d2 by mean shuffled d2 at same window
%   nShuffles — number of circular-shift shuffles
%   meanSubtract — if true, subtract bin-wise mean across windows at each slide position
% Returns:
%   d2Mat — numeric [nPos x nReach] d2 values.

    [nPos, nReachLocal] = size(popTraceCell);
    d2Mat = nan(nPos, nReachLocal);

    for k = 1:nPos
        validMask = false(1, nReachLocal);
        nBinsRef = nan;
        for r = 1:nReachLocal
            tr = popTraceCell{k, r};
            if ~isempty(tr)
                if isnan(nBinsRef)
                    nBinsRef = numel(tr);
                end
                if numel(tr) == nBinsRef
                    validMask(r) = true;
                end
            end
        end
        if ~any(validMask)
            continue;
        end

        validIdx = find(validMask);
        nValid = numel(validIdx);
        popMat = nan(nValid, nBinsRef);
        for ii = 1:nValid
            popMat(ii, :) = popTraceCell{k, validIdx(ii)}(:)';
        end
        if meanSubtract
            meanPerBin = nanmean(popMat, 1);
            popMat = popMat - meanPerBin;
        end

        d2Raw = nan(nValid, 1);
        for ii = 1:nValid
            d2Raw(ii) = compute_d2_from_pop_trace_local(popMat(ii, :)', pOrder, critType);
        end

        if ~normalizeD2
            d2Mat(k, validIdx) = d2Raw;
            continue;
        end

        d2Shuffled = nan(nValid, nShuffles);
        for sh = 1:nShuffles
            shuffledPopMat = nan(nValid, nBinsRef);
            for ii = 1:nValid
                wMat = windowMatCell{k, validIdx(ii)};
                if isempty(wMat)
                    continue;
                end
                nBins = size(wMat, 1);
                nNeu = size(wMat, 2);
                permutedMat = zeros(size(wMat));
                for n = 1:nNeu
                    permutedMat(:, n) = circshift(wMat(:, n), randi(nBins));
                end
                shuffledPopMat(ii, :) = mean(permutedMat, 2)';
            end

            if meanSubtract
                shMeanPerBin = nanmean(shuffledPopMat, 1);
                shuffledPopMat = shuffledPopMat - shMeanPerBin;
            end

            for ii = 1:nValid
                shuffledTrace = shuffledPopMat(ii, :)';
                d2Shuffled(ii, sh) = compute_d2_from_pop_trace_local(shuffledTrace, pOrder, critType);
            end
        end

        meanShuffled = nanmean(d2Shuffled, 2);
        d2Norm = nan(nValid, 1);
        for ii = 1:nValid
            if isfinite(d2Raw(ii)) && isfinite(meanShuffled(ii)) && meanShuffled(ii) > 0
                d2Norm(ii) = d2Raw(ii) / meanShuffled(ii);
            end
        end
        d2Mat(k, validIdx) = d2Norm;
    end
end

function d2Val = compute_d2_from_pop_trace_local(popTrace, pOrder, critType)
% compute_d2_from_pop_trace_local
% Goal: d2 from one population activity trace [timeBins x 1].
% Variables:
%   popTrace — population activity for one window (typically mean across neurons)
%   pOrder, critType — passed to myYuleWalker3 / getFixedPointDistance2
% Returns:
%   d2Val — scalar d2 (or NaN if estimation fails).

    d2Val = nan;
    if isempty(popTrace) || numel(popTrace) < pOrder + 2
        return;
    end
    try
        [varphi, ~] = myYuleWalker3(double(popTrace), pOrder);
        d2Val = getFixedPointDistance2(pOrder, critType, varphi);
    catch
        d2Val = nan;
    end
end

function [popWindowMeanMat, popCenterMat] = compute_mean_subtracted_pop_metrics_local(popTraceCell, centerRowCell)
% compute_mean_subtracted_pop_metrics_local
% Goal: build population-activity plot matrices from mean-subtracted traces.
% Variables:
%   popTraceCell — cell [nPos x nReach], each entry [timeBins x 1] pop trace
%   centerRowCell — cell [nPos x nReach], each entry scalar center-bin row index
% Returns:
%   popWindowMeanMat — [nPos x nReach], mean over bins of mean-subtracted trace
%   popCenterMat — [nPos x nReach], center-bin value of mean-subtracted trace.

    [nPos, nReachLocal] = size(popTraceCell);
    popWindowMeanMat = nan(nPos, nReachLocal);
    popCenterMat = nan(nPos, nReachLocal);

    for k = 1:nPos
        validMask = false(1, nReachLocal);
        nBinsRef = nan;
        for r = 1:nReachLocal
            tr = popTraceCell{k, r};
            cRow = centerRowCell{k, r};
            if isempty(tr) || isempty(cRow)
                continue;
            end
            if isnan(nBinsRef)
                nBinsRef = numel(tr);
            end
            if numel(tr) == nBinsRef && cRow >= 1 && cRow <= numel(tr)
                validMask(r) = true;
            end
        end
        if ~any(validMask)
            continue;
        end

        validIdx = find(validMask);
        nValid = numel(validIdx);
        traceMat = nan(nValid, nBinsRef);
        centerRows = nan(nValid, 1);
        for ii = 1:nValid
            r = validIdx(ii);
            traceMat(ii, :) = popTraceCell{k, r}(:)';
            centerRows(ii) = centerRowCell{k, r};
        end

        meanPerBin = nanmean(traceMat, 1);
        traceMatSub = traceMat - meanPerBin;

        for ii = 1:nValid
            r = validIdx(ii);
            cRow = centerRows(ii);
            popWindowMeanMat(k, r) = mean(traceMatSub(ii, :), 'omitnan');
            popCenterMat(k, r) = traceMatSub(ii, cRow);
        end
    end
end

function [centerVal, ok] = trace_center_value_local(traceVec, centerTime, sampleRate, traceStartSec)
% trace_center_value_local
% Goal: extract center-sample value from a continuous trace.
% Variables:
%   traceVec — 1 x n or n x 1 continuous signal
%   centerTime — window center time (s)
%   sampleRate — trace sampling rate (Hz)
%   traceStartSec — start time (s) corresponding to traceVec(1)
% Returns:
%   centerVal — value at the center sample
%   ok — true if requested index is in bounds.

    centerVal = nan;
    ok = false;
    if isempty(traceVec) || ~isfinite(centerTime) || sampleRate <= 0
        return;
    end
    tr = traceVec(:);
    ic = round((centerTime - traceStartSec) * sampleRate) + 1;
    if ic < 1 || ic > numel(tr)
        return;
    end
    centerVal = tr(ic);
    ok = isfinite(centerVal);
end
