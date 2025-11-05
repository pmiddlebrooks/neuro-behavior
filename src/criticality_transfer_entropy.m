%%
% Criticality Mutual Information Analysis
% Calculates mutual information between brain areas for different criticality measures
% Uses sliding window analysis on Mark's reach data and naturalistic data
% Tests hypothesis that MI is maximal near criticality and minimal far from criticality


%% ==============================================     Data Loading     ==============================================

paths = get_paths;
load(fullfile(paths.dropPath, 'criticality_compare_results.mat'), 'results');

% Get optimal bin sizes and window/step sizes
binSizeNat = results.naturalistic.unifiedBinSize;
binSizeRea = results.reach.unifiedBinSize;
% binSizeNat = .05;
% binSizeRea = .05;
windowSizeNat = results.naturalistic.unifiedWindowSize;
windowSizeRea = results.reach.unifiedWindowSize;
stepSize = results.params.stepSize;

areas = results.areas;
areasToTest = 2:3; % M56, DS, VS (as in criticality_compare.m)
areaPairs = perms(areasToTest); % all ordered pairs
areaPairs = areaPairs(areaPairs(:,1) ~= areaPairs(:,2), :); % remove self-pairs
numPairs = size(areaPairs, 1);

%% Optional Load original data
% Load naturalistic data (as in criticality_compare.m)
getDataType = 'spikes';
opts = neuro_behavior_options;
opts.firingRateCheckTime = 5 * 60;
opts.collectEnd = 30 * 60; % seconds
get_standard_data
idListNat = {idM23, idM56, idDS, idVS};

% Load reach data (as in criticality_compare.m)
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
[dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);
idM23R = find(strcmp(areaLabels, 'M23'));
idM56R = find(strcmp(areaLabels, 'M56'));
idDSR = find(strcmp(areaLabels, 'DS'));
idVSR = find(strcmp(areaLabels, 'VS'));
idListRea = {idM23R, idM56R, idDSR, idVSR};


%% ==============================================     Binning Spiking Data     ==============================================

% Bin naturalistic data for each area
binnedNat = cell(1, length(areas));
for a = areasToTest
    aID = idListNat{a};
    binnedNat{a} = neural_matrix_ms_to_frames(dataMat(:, aID), binSizeNat);
end

% Bin reach data for each area
binnedRea = cell(1, length(areas));
for a = areasToTest
    aID = idListRea{a};
    binnedRea{a} = neural_matrix_ms_to_frames(dataMatR(:, aID), binSizeRea);
end

% Get PCA options from results
pcaFlag = results.params.pcaFlag;
pcaFirstFlag = results.params.pcaFirstFlag;
nDim = results.params.nDim;

if pcaFlag
    % Naturalistic
    for a = areasToTest
        X = binnedNat{a};
        [coeff, score, ~, ~, ~, mu] = pca(X);
        if pcaFirstFlag
            pcs = 1:nDim;
        else
            pcs = size(score,2)-nDim+1:size(score,2);
        end
        X_recon = score(:,pcs) * coeff(:,pcs)' + mu;
        binnedNat{a} = X_recon;
    end
    % Reach
    for a = areasToTest
        X = binnedRea{a};
        [coeff, score, ~, ~, ~, mu] = pca(X);
        if pcaFirstFlag
            pcs = 1:nDim;
        else
            pcs = size(score,2)-nDim+1:size(score,2);
        end
        X_recon = score(:,pcs) * coeff(:,pcs)' + mu;
        binnedRea{a} = X_recon;
    end
end




%% ==============================================     Sliding Window Transfer Entropy     ==============================================

% Add option to plot TE as a function of lag
plotTeFlag = true; % Set to false to disable TE vs lag plots

maxLag = 12;
delayRange = 1:maxLag;
nTeBins = 20;

stepSamplesNat = round(stepSize / binSizeNat);
winSamplesNat = round(windowSizeNat / binSizeNat);
stepSamplesRea = round(stepSize / binSizeRea);
winSamplesRea = round(windowSizeRea / binSizeRea);

% For each area pair, store TE for all lags and all windows
teValsNatAllLags = cell(numPairs, 1);
teValsReaAllLags = cell(numPairs, 1);
bestLagNat = zeros(numPairs, 1);
bestLagRea = zeros(numPairs, 1);
bestTEVecNat = cell(numPairs, 1);
bestTEVecRea = cell(numPairs, 1);



%% =============================================      TEST HOW MANY BINS TO USE FOR TE HISTOGRAM    =====================
testBinMode = true; % Set to true to test nTeBins sensitivity
if testBinMode
    nTeBinsList = [5, 8, 10, 15, 20, 30];
    nTeBinsList = [10, 20, 30, 40, 50, 60, 70];
    nTestWindows = 10;
    for p = 1:numPairs
        a1 = areaPairs(p, 1);
        a2 = areaPairs(p, 2);
        % Naturalistic
        X = binnedNat{a1};
        Y = binnedNat{a2};
        test_te_bins(X, Y, nTeBinsList, winSamplesNat, stepSamplesNat, nTestWindows, sprintf('Nat: %s→%s', areas{a1}, areas{a2}));
        % Reach
        Xr = binnedRea{a1};
        Yr = binnedRea{a2};
        test_te_bins(Xr, Yr, nTeBinsList, winSamplesRea, stepSamplesRea, nTestWindows, sprintf('Reach: %s→%s', areas{a1}, areas{a2}));
    end
end


%%
% NATURALISTIC
for p = 1:numPairs
    a1 = areaPairs(p, 1);
    a2 = areaPairs(p, 2);
    x = sum(binnedNat{a1}, 2);
    y = sum(binnedNat{a2}, 2);
    nTime = length(x);
    nWindows = floor((nTime - winSamplesNat) / stepSamplesNat) + 1;
    teMat = nan(length(delayRange), nWindows);
    for d = 1:length(delayRange)
        delay = delayRange(d);
        for w = 1:nWindows
            startIdx = (w - 1) * stepSamplesNat + 1;
            endIdx = startIdx + winSamplesNat - 1;
            centerIdx = startIdx + floor((endIdx - startIdx)/2);
            xWin = x(startIdx:endIdx);
            yWin = y(startIdx:endIdx);
            teMat(d, w) = calculate_transfer_entropy(xWin, yWin, delay, nTeBins);
        end
    end
    teValsNatAllLags{p} = teMat;
    % Find lag with max average TE (across windows)
    meanTE = nanmean(teMat, 2);
    [~, bestIdx] = max(meanTE);
    bestLagNat(p) = delayRange(bestIdx);
    bestTEVecNat{p} = teMat(bestIdx, :);
    fprintf('NAT: Area pair %s -> %s, best lag = %d bins, %.2f sec\n', areas{a2}, areas{a1}, bestLagNat(p), bestLagNat(p)*binSizeNat);
end

% REACH
for p = 1:numPairs
    a1 = areaPairs(p, 1);
    a2 = areaPairs(p, 2);
    x = sum(binnedRea{a1}, 2);
    y = sum(binnedRea{a2}, 2);
    nTime = length(x);
    nWindows = floor((nTime - winSamplesRea) / stepSamplesRea) + 1;
    teMat = nan(length(delayRange), nWindows);
    for d = 1:length(delayRange)
        delay = delayRange(d);
        for w = 1:nWindows
            startIdx = (w - 1) * stepSamplesRea + 1;
            endIdx = startIdx + winSamplesRea - 1;
            centerIdx = startIdx + floor((endIdx - startIdx)/2);
            xWin = x(startIdx:endIdx);
            yWin = y(startIdx:endIdx);
            teMat(d, w) = calculate_transfer_entropy(xWin, yWin, delay, nTeBins);
        end
    end
    teValsReaAllLags{p} = teMat;
    % Find lag with max average TE (across windows)
    meanTE = nanmean(teMat, 2);
    [~, bestIdx] = max(meanTE);
    bestLagRea(p) = delayRange(bestIdx);
    bestTEVecRea{p} = teMat(bestIdx, :);
    fprintf('REACH: Area pair %s -> %s, best lag = %d bins, %.2f sec\n', areas{a2}, areas{a1}, bestLagRea(p), bestLagRea(p)*binSizeNat);
end


if plotTeFlag
    for p = 1:numPairs
        a1 = areaPairs(p, 1);
        a2 = areaPairs(p, 2);
        % Naturalistic
        teMatNat = teValsNatAllLags{p};
        meanTeNat = nanmean(teMatNat, 2);
        figNat = figure(4000 + p); clf;
        plot(delayRange, meanTeNat, '-o', 'LineWidth', 2);
        xlabel('Lag (bins)'); ylabel('Mean TE');
        title(sprintf('Nat: TE_{%s→%s} vs Lag', areas{a2}, areas{a1}));
        grid on;
        % Save Naturalistic plot
        saveas(figNat, fullfile(paths.dropPath, sprintf('TE_Nat_%s_to_%s.png', areas{a2}, areas{a1})));
        % Reach
        teMatRea = teValsReaAllLags{p};
        meanTeRea = nanmean(teMatRea, 2);
        figRea = figure(5000 + p); clf;
        plot(delayRange, meanTeRea, '-o', 'LineWidth', 2);
        xlabel('Lag (bins)'); ylabel('Mean TE');
        title(sprintf('Reach: TE_{%s→%s} vs Lag', areas{a2}, areas{a1}));
        grid on;
        % Save Reach plot
        saveas(figRea, fullfile(paths.dropPath, sprintf('TE_Reach_%s_to_%s.png', areas{a2}, areas{a1})));
    end
end



%% ==============================================     Analysis Parameters     ==============================================

maxLag = 12; % maximum delay to test (in bins)
delayRange = 1:maxLag;



% Note: The transfer entropy values (teValsNat and teValsRea) are already computed 
% in the same time bins and aligned with the criticality measures, so no additional
% processing is needed before correlation analysis

%% ==============================================     Correlation with Criticality Metrics     ==============================================

% Use Pearson correlation
measures = {'d2', 'mrBr'};
measureNames = {'Distance to Criticality (d2)', 'MR Branching Ratio'};

corrResultsNat = struct();
corrResultsRea = struct();

for m = 1:length(measures)
    measure = measures{m};
    critNat = results.naturalistic.(measure);
    critRea = results.reach.(measure);
    
    % For each area (as target)
    for aIdx = 1:length(areasToTest)
        a = areasToTest(aIdx);
        % Find all pairs where this area is the target (first in pair)
        pairIdx = find(areaPairs(:,1) == a);
        for p = pairIdx'
            % Naturalistic
            teVals = bestTEVecNat{p};
            critVals = critNat{a};
            critVals = critVals(~isnan(critVals));
            validIdx = ~isnan(teVals) & ~isnan(critVals);
            if sum(validIdx) > 10
                [r, pval] = corr(teVals(validIdx)', critVals(validIdx)', 'Type', 'Pearson');
            else
                r = NaN; pval = NaN;
            end
            corrResultsNat.(measure)(aIdx, p) = r;
            
            % Plot
            figure(2000 + 100*m + aIdx); clf;
            yyaxis left; plot(teVals, 'b-'); ylabel('Transfer Entropy');
            yyaxis right; plot(critVals, 'r-'); ylabel(measureNames{m});
            xlabel('Time (binned index)');
            title(sprintf('Nat: Area %s TE vs %s, r=%.2f, p=%.3f', areas{a}, measureNames{m}, r, pval));
            legend({'TE', measureNames{m}});
            grid on;
            
            % Reach
            teValsR = bestTEVecRea{p};
            critValsR = critRea{a};
            critValsR = critValsR(~isnan(critValsR));
            validIdxR = ~isnan(teValsR) & ~isnan(critValsR);
            if sum(validIdxR) > 10
                [rR, pvalR] = corr(teValsR(validIdxR)', critValsR(validIdxR)', 'Type', 'Pearson');
            else
                rR = NaN; pvalR = NaN;
            end
            corrResultsRea.(measure)(aIdx, p) = rR;
            
            % Plot
            figure(3000 + 100*m + aIdx); clf;
            yyaxis left; plot(teValsR, 'b-'); ylabel('Transfer Entropy');
            yyaxis right; plot(critValsR, 'r-'); ylabel(measureNames{m});
            xlabel('Time (binned index)');
            title(sprintf('Reach: Area %s TE vs %s, r=%.2f, p=%.3f', areas{a}, measureNames{m}, rR, pvalR));
            legend({'TE', measureNames{m}});
            grid on;
        end
    end
end

%% ==============================================     Visualization     ==============================================

% Plot TE over time for each area pair and measure
for m = 1:length(measures)
    figure(1000 + m); clf;
    set(gcf, 'Position', monitorTwo);
    for p = 1:numPairs
        subplot(1, numPairs, p);
        % Get time points for naturalistic data
        timePointsNat = results.naturalistic.startS{areaPairs(p,1)};
        validIdxNat = ~isnan(bestTEVecNat{p});
        plot(timePointsNat(validIdxNat)/60, bestTEVecNat{p}(validIdxNat), '-o', 'LineWidth', 2, 'MarkerSize', 4);
        hold on;
        
        % Get time points for reach data
        timePointsRea = results.reach.startS{areaPairs(p,1)};
        validIdxRea = ~isnan(bestTEVecRea{p});
        plot(timePointsRea(validIdxRea)/60, bestTEVecRea{p}(validIdxRea), '--s', 'LineWidth', 2, 'MarkerSize', 4);
        
        xlabel('Time (minutes)');
        ylabel('Transfer Entropy');
        title(sprintf('%s: %s vs %s', measureNames{m}, areas{areaPairs(p,1)}, areas{areaPairs(p,2)}));
        legend({'Naturalistic', 'Reach'}, 'Location', 'best');
        grid on;
    end
    sgtitle(sprintf('Transfer Entropy Over Time (%s)', measureNames{m}));
end


%% ==============================================     Summary Statistics     ==============================================

fprintf('\n=== Summary Statistics ===\n');

% Print summary statistics for Transfer Entropy
fprintf('\nNaturalistic Data:\n');
for m = 1:length(measures)
    measure = measures{m};
    teData = bestTEVecNat; % Use the vector of max TE values
    % fprintf('%s - Mean TE: %.3f, Max TE: %.3f\n', measureNames{m}, ...
    %     nanmean(teData(:)), nanmax(teData(:)));
end

fprintf('\nReach Data:\n');
for m = 1:length(measures)
    measure = measures{m};
    teData = bestTEVecRea; % Use the vector of max TE values
    % fprintf('%s - Mean TE: %.3f, Max TE: %.3f\n', measureNames{m}, ...
    %     nanmean(teData(:)), nanmax(teData(:)));
end

%% ==============================================     Save Results     ==============================================

%% ==============================================     Save Results     ==============================================

% Save all results
teResults = struct();
teResults.areas = areas;
teResults.measures = measures;
teResults.measureNames = measureNames;
teResults.areaPairs = areaPairs;

% Transfer Entropy results
teResults.naturalistic = bestTEVecNat; % Store the vectors of max TE values
teResults.reach = bestTEVecRea; % Store the vectors of max TE values

% Analysis parameters
teResults.params.maxLag = maxLag;
teResults.params.delayRange = delayRange;
teResults.params.stepSize = stepSize;
teResults.params.areasToTest = areasToTest;
teResults.params.unifiedBinSizeNat = results.naturalistic.unifiedBinSize;
teResults.params.unifiedWindowSizeNat = results.naturalistic.unifiedWindowSize;
teResults.params.unifiedBinSizeRea = results.reach.unifiedBinSize;
teResults.params.unifiedWindowSizeRea = results.reach.unifiedWindowSize;

% Save to file
save(fullfile(paths.dropPath, 'criticality_mutual_information_results.mat'), 'teResults');

fprintf('\nAnalysis complete! Results saved to criticality_mutual_information_results.mat\n'); 







% Function to calculate permutation-based p-values
function pValue = permutation_test(data1, data2, nPermutations, nBins)
    % Remove NaN values
    validIdx = ~isnan(data1) & ~isnan(data2);
    data1 = data1(validIdx);
    data2 = data2(validIdx);
    
    if length(data1) < 10
        pValue = nan;
        return;
    end
    
    % Calculate observed MI
    observedMI = calculate_mutual_information(data1, data2, nBins);
    
    % Permutation test
    nullMI = zeros(nPermutations, 1);
    for perm = 1:nPermutations
        % Shuffle one of the vectors
        shuffledData2 = data2(randperm(length(data2)));
        nullMI(perm) = calculate_mutual_information(data1, shuffledData2, nBins);
    end
    
    % Calculate p-value
    pValue = sum(nullMI >= observedMI) / nPermutations;
end

function test_te_bins(X, Y, nTeBinsList, winSamples, stepSamples, nTestWindows, plotTitlePrefix)
    % Helper to test TE stability across nTeBins for a few random windows
    nTime = size(X, 1);
    nWindows = floor((nTime - winSamples) / stepSamples) + 1;
    if nWindows < nTestWindows
        winIdx = 1:nWindows;
    else
        winIdx = sort(randperm(nWindows, nTestWindows));
    end
    teValsAll = zeros(length(nTeBinsList), length(winIdx));
    nEmptyBinsAll = zeros(length(nTeBinsList), length(winIdx));
    for w = 1:length(winIdx)
        startIdx = (winIdx(w) - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;
        xWin = sum(X(startIdx:endIdx, :), 2);
        yWin = sum(Y(startIdx:endIdx, :), 2);
        for i = 1:length(nTeBinsList)
            nBins = nTeBinsList(i);
            teValsAll(i, w) = calculate_transfer_entropy(xWin, yWin, 1, nBins);
            % Histogram for xWin
            [counts, ~] = histcounts(xWin, nBins);
            nEmptyBinsAll(i, w) = sum(counts == 0);
        end
    end
    % Plot TE vs nTeBins
    figure;
    subplot(2,1,1);
    plot(nTeBinsList, mean(teValsAll, 2), '-o', 'LineWidth', 2);
    hold on;
    plot(nTeBinsList, mean(teValsAll, 2) + std(teValsAll, 0, 2), '--', 'Color', [0.7 0.7 0.7]);
    plot(nTeBinsList, mean(teValsAll, 2) - std(teValsAll, 0, 2), '--', 'Color', [0.7 0.7 0.7]);
    xlabel('nTeBins'); ylabel('TE');
    title(sprintf('%s: TE vs nTeBins (mean ± std across %d windows)', plotTitlePrefix, length(winIdx)));
    grid on;
    % Plot empty bins vs nTeBins
    subplot(2,1,2);
    plot(nTeBinsList, mean(nEmptyBinsAll, 2), '-o', 'LineWidth', 2);
    hold on;
    plot(nTeBinsList, mean(nEmptyBinsAll, 2) + std(nEmptyBinsAll, 0, 2), '--', 'Color', [0.7 0.7 0.7]);
    plot(nTeBinsList, mean(nEmptyBinsAll, 2) - std(nEmptyBinsAll, 0, 2), '--', 'Color', [0.7 0.7 0.7]);
    xlabel('nTeBins'); ylabel('Number of Empty Bins');
    title(sprintf('%s: Empty Bins vs nTeBins (mean ± std across %d windows)', plotTitlePrefix, length(winIdx)));
    grid on;
end
