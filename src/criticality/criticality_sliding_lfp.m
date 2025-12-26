%%
% Criticality Sliding Window Analysis Script for LFP (d2 and DFA)
% Performs d2 and DFA criticality analysis on LFP binnedEnvelopes and raw LFP data
% Uses common time points (startS) for all analyses, with different window sizes
%
% Expected workspace variables:
%   - binnedEnvelopes: Cell array {area}{band} = [nFrames_b x 1] of binned LFP envelopes
%   - bands: Cell array of frequency bands (e.g., {'alpha', [8 13]; 'beta', [13 30]})
%   - areas: Cell array of area names (e.g., {'M23', 'M56', 'DS', 'VS'})
%   - bandBinSizes: Array of bin sizes for each band (seconds)
%   - lfpPerArea: Raw LFP data [nSamples x nAreas]
%   - lfpBinSize: Array of bin sizes for raw LFP analysis (seconds)
%   - opts: Options structure with fsLfp, etc.
%   - saveDir: Directory to save results
%   - slidingWindowSize: Window size for d2 analysis (seconds)

% =============================    Configuration    =============================

% Flags
loadExistingResults = false;
makePlots = true;
plotBinnedEnvelopes = true;  % Plot results from binnedEnvelopes (power bands)
plotRawLfp = true;           % Plot results from raw LFP
plotD2 = true;               % Plot d2 results
plotDFA = true;              % Plot DFA results

% Analysis flags
analyzeD2 = true;      % compute d2
analyzeDFA = true;     % compute DFA alpha

% Permutation testing flags
enablePermutations = false;  % Set to true to perform circular permutation testing
nShuffles = 3;  % Number of circular permutations to perform

% Analysis parameters
minSegmentLength = 50;
pOrder = 10;
critType = 2;

% =============================    Data Validation    =============================
% Verify that required variables are in workspace
requiredVars = {'binnedEnvelopes', 'bands', 'bandBinSizes'};
for i = 1:length(requiredVars)
    if ~exist(requiredVars{i}, 'var')
        error('Required variable %s not found in workspace.', requiredVars{i});
    end
end

% Determine number of bands and areas
numBands = size(bands, 1);
numAreas = length(binnedEnvelopes);
if numAreas ~= length(areas)
    warning('Number of areas in binnedEnvelopes (%d) does not match areas array (%d). Using binnedEnvelopes length.', numAreas, length(areas));
    if numAreas < length(areas)
        areas = areas(1:numAreas);
    end
end

% Areas to analyze
if ~exist('areasToTest', 'var')
    areasToTest = 1:numAreas;
end

% Check for raw LFP data
if exist('lfpPerArea', 'var') && ~isempty(lfpPerArea) && exist('lfpBinSize', 'var') && ~isempty(lfpBinSize)
    if ~exist('opts', 'var') || ~isfield(opts, 'fsLfp')
        error('opts.fsLfp must be defined for raw LFP analysis.');
    end
    numLfpBins = length(lfpBinSize);
    hasRawLfp = true;
else
    numLfpBins = 0;
    hasRawLfp = false;
end

% =============================    Calculate Common Parameters    =============================
% Check required parameters
if ~exist('slidingWindowSize', 'var') || isempty(slidingWindowSize)
    error('slidingWindowSize must be defined in workspace.');
end

fprintf('\n=== LFP Criticality Analysis Setup ===\n');
fprintf('Number of bands: %d\n', numBands);
fprintf('Number of areas: %d\n', numAreas);

% Calculate common step size: 2 * max(bandBinSizes)
maxBandBinSize = max(bandBinSizes);
stepSize = 20 * maxBandBinSize;
fprintf('Step size: %.3f s\n', stepSize);

% Determine total session duration
% Use the minimum duration across all signals to ensure all windows fit
durations = [];
if hasRawLfp
    durations(end+1) = size(lfpPerArea, 1) / opts.fsLfp;
end
% Check all bands for binned envelopes
for a = 1:numAreas
    if length(binnedEnvelopes) >= a && ~isempty(binnedEnvelopes{a})
        for b = 1:numBands
            if length(binnedEnvelopes{a}) >= b && ~isempty(binnedEnvelopes{a}{b})
                durations(end+1) = length(binnedEnvelopes{a}{b}) * bandBinSizes(b);
            end
        end
    end
end

if isempty(durations)
    error('Cannot determine session duration from available data.');
end

totalSessionDuration = min(durations);
fprintf('Total session duration: %.1f s (minimum across all signals)\n', totalSessionDuration);
if length(durations) > 1
    fprintf('  Duration range: %.1f - %.1f s\n', min(durations), max(durations));
end

% =============================    Window Size Calculations    =============================
% d2: Same window size for all signals
d2WindowSize = slidingWindowSize;
fprintf('d2 window size: %.2f s (common for all signals)\n', d2WindowSize);

% DFA: Different window sizes
% For binned envelopes: use max(bandBinSizes) to ensure enough samples
dfaEnvWinSamples_min = 1000;
dfaEnvWinSize = max(dfaEnvWinSamples_min * max(bandBinSizes), 30);
fprintf('DFA binned envelopes window size: %.2f s (min samples: %d)\n', dfaEnvWinSize, dfaEnvWinSamples_min);

% For raw LFP: calculated per bin size (will be done in loop)
dfaLfpWinSamples_min = 2000;
fprintf('DFA raw LFP window size: calculated per bin size (min samples: %d)\n', dfaLfpWinSamples_min);

% Calculate maximum window size across all analyses to determine valid time range
maxWindowSize = max(d2WindowSize, dfaEnvWinSize);
if hasRawLfp
    % Also consider raw LFP DFA window sizes
    for lb = 1:numLfpBins
        dfaLfpWinSize = max(dfaLfpWinSamples_min * lfpBinSize(lb), 30);
        maxWindowSize = max(maxWindowSize, dfaLfpWinSize);
    end
end
fprintf('Maximum window size: %.2f s\n', maxWindowSize);

% Calculate common time points (startS) for all analyses
% These are the center times of windows, determined by step size
% Start from stepSize/2 (half step from beginning) and step by stepSize
% End at totalSessionDuration - maxWindowSize/2 to ensure last window fits
lastCenterTime = totalSessionDuration - maxWindowSize/2;
startS = (stepSize/2) : stepSize : lastCenterTime;
numWindows = length(startS);
fprintf('Common time points (startS): %d windows\n', numWindows);
fprintf('  First center: %.2f s, Last center: %.2f s\n', startS(1), startS(end));
fprintf('  Last window would extend to: %.2f s (data ends at: %.2f s)\n', startS(end) + maxWindowSize/2, totalSessionDuration);

% =============================    Initialize Results    =============================
% Structure: metric{area}{band/bin} = [1 x numWindows]
d2 = cell(1, numAreas);
dfa = cell(1, numAreas);
d2Permuted = cell(1, numAreas);

% Initialize for binned envelopes
for a = 1:numAreas
    d2{a} = cell(1, numBands);
    dfa{a} = cell(1, numBands);
    d2Permuted{a} = cell(1, numBands);
end

% Initialize for raw LFP
if hasRawLfp
    d2Lfp = cell(1, numAreas);
    dfaLfp = cell(1, numAreas);
    for a = 1:numAreas
        d2Lfp{a} = cell(1, numLfpBins);
        dfaLfp{a} = cell(1, numLfpBins);
    end
end

% =============================    Analysis    =============================
fprintf('\n=== Processing Areas ===\n');

for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});
    
    % ========== Binned Envelopes Analysis ==========
    if length(binnedEnvelopes) < a || isempty(binnedEnvelopes{a})
        fprintf('  Skipping: No binned envelopes data for area %s\n', areas{a});
    else
        areaEnvelopes = binnedEnvelopes{a};
        
        for b = 1:numBands
            fprintf('  Processing band %d (%s)...\n', b, bands{b, 1});
            tic;
            
            bandSignal = areaEnvelopes{b};
            bandBinSize = bandBinSizes(b);
            numFrames_b = length(bandSignal);
            actualDuration = numFrames_b * bandBinSize;
            
            % Initialize arrays
            d2{a}{b} = nan(1, numWindows);
            if analyzeDFA
                dfa{a}{b} = nan(1, numWindows);
            end
            
            % Calculate window sizes in samples
            d2WinSamples = round(d2WindowSize / bandBinSize);
            if analyzeDFA
                dfaWinSamples = round(dfaEnvWinSize / bandBinSize);
            end
            
            % Skip if not enough samples
            if d2WinSamples < minSegmentLength
                fprintf('    Skipping: Not enough samples for d2 (need %d, have %d)\n', minSegmentLength, d2WinSamples);
                continue;
            end
            
            % Debug: Check actual data duration vs expected
            if abs(actualDuration - totalSessionDuration) > 0.1
                fprintf('    Warning: Actual duration (%.2f s) differs from totalSessionDuration (%.2f s) by %.2f s\n', ...
                    actualDuration, totalSessionDuration, abs(actualDuration - totalSessionDuration));
            end
            
            % Process each time point
            for w = 1:numWindows
                centerTime = startS(w);
                
                % ========== d2 Analysis ==========
                if analyzeD2
                    % Calculate window bounds centered on startS
                    d2StartTime = centerTime - d2WindowSize/2;
                    d2EndTime = centerTime + d2WindowSize/2;
                    
                    % Convert to sample indices
                    % Use floor for end index to ensure we never exceed data bounds
                    d2StartIdx = max(1, round(d2StartTime / bandBinSize) + 1);
                    d2EndIdx = min(numFrames_b, floor(d2EndTime / bandBinSize));
                    
                    % Check if window fits and is reasonable size (at least 80% of expected)
                    actualWinSamples = d2EndIdx - d2StartIdx + 1;
                    if d2StartIdx >= 1 && d2EndIdx <= numFrames_b && actualWinSamples >= 0.8 * d2WinSamples
                        try
                            wSignal = bandSignal(d2StartIdx:d2EndIdx);
                            [varphi, ~] = myYuleWalker3(wSignal, pOrder);
                            d2{a}{b}(w) = getFixedPointDistance2(pOrder, critType, varphi);
                        catch
                            d2{a}{b}(w) = nan;
                        end
                    end
                end
                
                % ========== DFA Analysis ==========
                if analyzeDFA
                    % Calculate window bounds centered on startS
                    dfaStartTime = centerTime - dfaEnvWinSize/2;
                    dfaEndTime = centerTime + dfaEnvWinSize/2;
                    
                    % Convert to sample indices
                    % Use floor for end index to ensure we never exceed data bounds
                    dfaStartIdx = max(1, round(dfaStartTime / bandBinSize) + 1);
                    dfaEndIdx = min(numFrames_b, floor(dfaEndTime / bandBinSize));
                    
                    % Check if window fits and is reasonable size (at least 80% of expected)
                    actualWinSamples = dfaEndIdx - dfaStartIdx + 1;
                    if dfaStartIdx >= 1 && dfaEndIdx <= numFrames_b && actualWinSamples >= 0.8 * dfaWinSamples
                        try
                            dfaSignal = bandSignal(dfaStartIdx:dfaEndIdx);
                            dfaAlpha = compute_DFA(dfaSignal, false);
                            dfa{a}{b}(w) = dfaAlpha;
                        catch
                            dfa{a}{b}(w) = nan;
                        end
                    end
                end
            end
            
            % Debug: Check last few windows
            if analyzeD2
                lastValidD2 = find(~isnan(d2{a}{b}), 1, 'last');
                if ~isempty(lastValidD2)
                    lastValidTime = startS(lastValidD2);
                    expectedLastTime = actualDuration - d2WindowSize/2;
                    fprintf('    d2: Last valid window at %.2f s (expected: %.2f s, diff: %.2f s)\n', ...
                        lastValidTime, expectedLastTime, abs(lastValidTime - expectedLastTime));
                end
            end
            if analyzeDFA
                lastValidDFA = find(~isnan(dfa{a}{b}), 1, 'last');
                if ~isempty(lastValidDFA)
                    lastValidTime = startS(lastValidDFA);
                    expectedLastTime = actualDuration - dfaEnvWinSize/2;
                    fprintf('    DFA: Last valid window at %.2f s (expected: %.2f s, diff: %.2f s)\n', ...
                        lastValidTime, expectedLastTime, abs(lastValidTime - expectedLastTime));
                end
            end
            
            fprintf('    Band %d completed in %.1f minutes\n', b, toc/60);
        end
    end
    
    % ========== Raw LFP Analysis ==========
    if hasRawLfp
        fprintf('  Processing raw LFP for area %s...\n', areas{a});
        rawSignal = lfpPerArea(:, a);
        fsRaw = opts.fsLfp;
        
        for lb = 1:numLfpBins
            targetBinSize = lfpBinSize(lb);
            fprintf('    Binning raw LFP with size %.3f s...\n', targetBinSize);
            
            % Bin raw LFP signal
            samplesPerBin = round(targetBinSize * fsRaw);
            numBins = floor(length(rawSignal) / samplesPerBin);
            
            if numBins < minSegmentLength
                fprintf('      Skipping: Not enough bins (need %d, have %d)\n', minSegmentLength, numBins);
                continue;
            end
            
            % Efficient binning
            binnedSignal = mean(reshape(rawSignal(1:numBins*samplesPerBin), samplesPerBin, numBins), 1)';
            actualDuration = numBins * targetBinSize;
            
            % Calculate DFA window size for this bin size
            dfaLfpWinSize = max(dfaLfpWinSamples_min * targetBinSize, 30);
            
            % Calculate window sizes in samples
            d2WinSamples = round(d2WindowSize / targetBinSize);
            dfaWinSamples = round(dfaLfpWinSize / targetBinSize);
            
            % Initialize arrays
            d2Lfp{a}{lb} = nan(1, numWindows);
            if analyzeDFA
                dfaLfp{a}{lb} = nan(1, numWindows);
            end
            
            % Debug: Check actual data duration vs expected
            if abs(actualDuration - totalSessionDuration) > 0.1
                fprintf('      Warning: Actual duration (%.2f s) differs from totalSessionDuration (%.2f s) by %.2f s\n', ...
                    actualDuration, totalSessionDuration, abs(actualDuration - totalSessionDuration));
            end
            
            % Process each time point
            for w = 1:numWindows
                centerTime = startS(w);
                
                % ========== d2 Analysis ==========
                if analyzeD2
                    % Calculate window bounds centered on startS
                    d2StartTime = centerTime - d2WindowSize/2;
                    d2EndTime = centerTime + d2WindowSize/2;
                    
                    % Convert to bin indices
                    % Use floor for end index to ensure we never exceed data bounds
                    d2StartIdx = max(1, round(d2StartTime / targetBinSize) + 1);
                    d2EndIdx = min(numBins, floor(d2EndTime / targetBinSize));
                    
                    % Check if window fits and is reasonable size (at least 80% of expected)
                    actualWinSamples = d2EndIdx - d2StartIdx + 1;
                    if d2StartIdx >= 1 && d2EndIdx <= numBins && actualWinSamples >= 0.8 * d2WinSamples
                        try
                            wSignal = binnedSignal(d2StartIdx:d2EndIdx);
                            [varphi, ~] = myYuleWalker3(wSignal, pOrder);
                            d2Lfp{a}{lb}(w) = getFixedPointDistance2(pOrder, critType, varphi);
                        catch
                            d2Lfp{a}{lb}(w) = nan;
                        end
                    end
                end
                
                % ========== DFA Analysis ==========
                if analyzeDFA
                    % Calculate window bounds centered on startS
                    dfaStartTime = centerTime - dfaLfpWinSize/2;
                    dfaEndTime = centerTime + dfaLfpWinSize/2;
                    
                    % Convert to bin indices
                    % Use floor for end index to ensure we never exceed data bounds
                    dfaStartIdx = max(1, round(dfaStartTime / targetBinSize) + 1);
                    dfaEndIdx = min(numBins, floor(dfaEndTime / targetBinSize));
                    
                    % Check if window fits and is reasonable size (at least 80% of expected)
                    actualWinSamples = dfaEndIdx - dfaStartIdx + 1;
                    if dfaStartIdx >= 1 && dfaEndIdx <= numBins && actualWinSamples >= 0.8 * dfaWinSamples
                        try
                            dfaSignal = binnedSignal(dfaStartIdx:dfaEndIdx);
                            dfaAlpha = compute_DFA(dfaSignal, false);
                            dfaLfp{a}{lb}(w) = dfaAlpha;
                        catch
                            dfaLfp{a}{lb}(w) = nan;
                        end
                    end
                end
            end
            
            % Debug: Check last few windows
            if analyzeD2
                lastValidD2 = find(~isnan(d2Lfp{a}{lb}), 1, 'last');
                if ~isempty(lastValidD2)
                    lastValidTime = startS(lastValidD2);
                    expectedLastTime = actualDuration - d2WindowSize/2;
                    fprintf('      d2: Last valid window at %.2f s (expected: %.2f s, diff: %.2f s)\n', ...
                        lastValidTime, expectedLastTime, abs(lastValidTime - expectedLastTime));
                end
            end
            if analyzeDFA
                lastValidDFA = find(~isnan(dfaLfp{a}{lb}), 1, 'last');
                if ~isempty(lastValidDFA)
                    lastValidTime = startS(lastValidDFA);
                    expectedLastTime = actualDuration - dfaLfpWinSize/2;
                    fprintf('      DFA: Last valid window at %.2f s (expected: %.2f s, diff: %.2f s)\n', ...
                        lastValidTime, expectedLastTime, abs(lastValidTime - expectedLastTime));
                end
            end
            
            fprintf('    Raw LFP bin size %.3f s completed\n', targetBinSize);
        end
    end
end

% =============================    Save Results    =============================
results = struct();
results.areas = areas;
results.bands = bands;
results.startS = startS;
results.stepSize = stepSize;
results.d2WindowSize = d2WindowSize;
results.dfaEnvWinSize = dfaEnvWinSize;
results.dfaEnvWinSamples_min = dfaEnvWinSamples_min;
results.dfaLfpWinSamples_min = dfaLfpWinSamples_min;
results.bandBinSizes = bandBinSizes;
results.d2 = d2;
results.dfa = dfa;
if hasRawLfp
    results.d2Lfp = d2Lfp;
    results.dfaLfp = dfaLfp;
    results.lfpBinSize = lfpBinSize;
end
results.params.slidingWindowSize = slidingWindowSize;
results.params.analyzeD2 = analyzeD2;
results.params.analyzeDFA = analyzeDFA;
results.params.pOrder = pOrder;
results.params.critType = critType;

% Create results path
if ~exist('saveDir', 'var') || isempty(saveDir)
    if ~exist('paths', 'var')
        paths = get_paths;
    end
    saveDir = fullfile(paths.saveDataPath, 'criticality_lfp');
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end
end

if ~exist('sessionName', 'var') || isempty(sessionName)
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_lfp_win%.1f.mat', slidingWindowSize));
else
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_lfp_win%.1f_%s.mat', slidingWindowSize, sessionName));
end

save(resultsPath, 'results');
fprintf('\nSaved results to: %s\n', resultsPath);

% =============================    Plotting    =============================
if makePlots
    fprintf('\n=== Creating Plots ===\n');
    
    % Extract filename prefix
    if exist('dataBaseName', 'var') && ~isempty(dataBaseName)
        filePrefix = dataBaseName(1:min(8, length(dataBaseName)));
    elseif exist('sessionName', 'var') && ~isempty(sessionName)
        filePrefix = sessionName(1:min(8, length(sessionName)));
    elseif exist('saveDir', 'var') && ~isempty(saveDir)
        [~, dirName, ~] = fileparts(saveDir);
        filePrefix = dirName(1:min(8, length(dirName)));
    else
        filePrefix = '';
    end
    
    % Detect monitors
    monitorPositions = get(0, 'MonitorPositions');
    if size(monitorPositions, 1) >= 2
        targetPos = monitorPositions(size(monitorPositions, 1), :);
    else
        targetPos = monitorPositions(1, :);
    end
    
    % Collect data for axis limits
    allStartS = startS(:);
    allD2 = [];
    allDFA = [];
    
    for a = areasToTest
        if plotD2 && plotBinnedEnvelopes
            for b = 1:numBands
                if ~isempty(d2{a}{b})
                    allD2 = [allD2(:); d2{a}{b}(~isnan(d2{a}{b}))'];
                end
            end
        end
        if plotD2 && plotRawLfp && hasRawLfp
            for lb = 1:numLfpBins
                if ~isempty(d2Lfp{a}{lb})
                    allD2 = [allD2(:); d2Lfp{a}{lb}(~isnan(d2Lfp{a}{lb}))'];
                end
            end
        end
        if plotDFA && plotBinnedEnvelopes
            for b = 1:numBands
                if ~isempty(dfa{a}{b})
                    allDFA = [allDFA(:); dfa{a}{b}(~isnan(dfa{a}{b}))'];
                end
            end
        end
        if plotDFA && plotRawLfp && hasRawLfp
            for lb = 1:numLfpBins
                if ~isempty(dfaLfp{a}{lb})
                    allDFA = [allDFA(:); dfaLfp{a}{lb}(~isnan(dfaLfp{a}{lb}))'];
                end
            end
        end
    end
    
    % Determine axis limits
    xMin = min(allStartS);
    xMax = max(allStartS);
    
    if ~isempty(allD2)
        yMinD2 = min(allD2);
        yMaxD2 = max(allD2);
        yRangeD2 = yMaxD2 - yMinD2;
        yMinD2 = yMinD2 - 0.05 * yRangeD2;
        yMaxD2 = yMaxD2 + 0.05 * yRangeD2;
    else
        yMinD2 = 0;
        yMaxD2 = 0.5;
    end
    
    if ~isempty(allDFA)
        yMinDFA = min(allDFA);
        yMaxDFA = max(allDFA);
        yRangeDFA = yMaxDFA - yMinDFA;
        yMinDFA = max(0.3, yMinDFA - 0.05 * yRangeDFA);
        yMaxDFA = min(1.7, yMaxDFA + 0.05 * yRangeDFA);
    else
        yMinDFA = 0.3;
        yMaxDFA = 1.7;
    end
    
    % ========== d2 Plot ==========
    if plotD2
        figure(910); clf;
        set(gcf, 'Position', targetPos);
        numRows = length(areasToTest);
        ha = tight_subplot(numRows, 1, [0.035 0.04], [0.03 0.08], [0.08 0.04]);
        bandColors = lines(numBands);
        
        for idx = 1:length(areasToTest)
            a = areasToTest(idx);
            axes(ha(idx)); hold on;
            
            % Plot raw LFP d2
            if plotRawLfp && hasRawLfp
                numLfpBins = length(lfpBinSize);
                grayColors = repmat(linspace(0.8, 0, numLfpBins)', 1, 3);
                for lb = 1:numLfpBins
                    if ~isempty(d2Lfp{a}{lb})
                        validIdx = ~isnan(d2Lfp{a}{lb});
                        if any(validIdx)
                            plot(startS(validIdx), d2Lfp{a}{lb}(validIdx), '-', 'Color', grayColors(lb, :), 'LineWidth', 2, ...
                                'DisplayName', sprintf('Raw LFP (%.3fs bin)', lfpBinSize(lb)));
                        end
                    end
                end
            end
            
            % Plot binned envelope d2
            if plotBinnedEnvelopes
                for b = 1:numBands
                    if ~isempty(d2{a}{b})
                        validIdx = ~isnan(d2{a}{b});
                        if any(validIdx)
                            plot(startS(validIdx), d2{a}{b}(validIdx), '-', 'Color', bandColors(b, :), 'LineWidth', 2, ...
                                'DisplayName', sprintf('%s (%.3fs bin)', bands{b, 1}, bandBinSizes(b)));
                        end
                    end
                end
            end
            
            title(sprintf('%s - d2 Analysis', areas{a}));
            xlabel('Time (s)');
            ylabel('d2');
            xlim([xMin, xMax]);
            ylim([yMinD2, yMaxD2]);
            set(gca, 'XTickLabelMode', 'auto');
            set(gca, 'YTickLabelMode', 'auto');
            if idx == 1
                legend('Location', 'best');
            end
            grid on;
        end
        
        if exist('dataType', 'var') && strcmp(dataType, 'reach')
            if ~isempty(filePrefix)
                sgtitle(sprintf('[%s] LFP d2 Analysis - win=%.2fs, step=%.3fs', filePrefix, d2WindowSize, stepSize));
            else
                sgtitle(sprintf('LFP d2 Analysis - win=%.2fs, step=%.3fs', d2WindowSize, stepSize));
            end
        else
            if ~isempty(filePrefix)
                sgtitle(sprintf('[%s] LFP d2 Analysis - win=%.2fs, step=%.3fs', filePrefix, d2WindowSize, stepSize));
            else
                sgtitle(sprintf('LFP d2 Analysis - win=%.2fs, step=%.3fs', d2WindowSize, stepSize));
            end
        end
        
        if ~isempty(filePrefix)
            exportgraphics(gcf, fullfile(saveDir, sprintf('%s_criticality_lfp_d2_win%.1f.png', filePrefix, slidingWindowSize)), 'Resolution', 300);
        else
            exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_lfp_d2_win%.1f.png', slidingWindowSize)), 'Resolution', 300);
        end
    end
    
    % ========== DFA Plot ==========
    if plotDFA
        figure(911); clf;
        set(gcf, 'Position', targetPos);
        numRows = length(areasToTest);
        haDFA = tight_subplot(numRows, 1, [0.035 0.04], [0.03 0.08], [0.08 0.04]);
        bandColors = lines(numBands);
        
        for idx = 1:length(areasToTest)
            a = areasToTest(idx);
            axes(haDFA(idx)); hold on;
            
            % Plot raw LFP DFA
            if plotRawLfp && hasRawLfp
                numLfpBins = length(lfpBinSize);
                grayColors = repmat(linspace(0.8, 0, numLfpBins)', 1, 3);
                for lb = 1:numLfpBins
                    if ~isempty(dfaLfp{a}{lb})
                        validIdx = ~isnan(dfaLfp{a}{lb});
                        if any(validIdx)
                            plot(startS(validIdx), dfaLfp{a}{lb}(validIdx), '-', 'Color', grayColors(lb, :), 'LineWidth', 1.5, ...
                                'DisplayName', sprintf('DFA Raw LFP (%.3fs bin)', lfpBinSize(lb)));
                        end
                    end
                end
            end
            
            % Plot binned envelope DFA
            if plotBinnedEnvelopes
                for b = 1:numBands
                    if ~isempty(dfa{a}{b})
                        validIdx = ~isnan(dfa{a}{b});
                        if any(validIdx)
                            plot(startS(validIdx), dfa{a}{b}(validIdx), '-', 'Color', bandColors(b, :), 'LineWidth', 1.5, ...
                                'DisplayName', sprintf('DFA %s', bands{b, 1}));
                        end
                    end
                end
            end
            
            title(sprintf('%s - DFA Analysis', areas{a}));
            xlabel('Time (s)');
            ylabel('DFA \\alpha');
            xlim([xMin, xMax]);
            ylim([yMinDFA, yMaxDFA]);
            set(gca, 'XTickLabelMode', 'auto');
            set(gca, 'YTickLabelMode', 'auto');
            if idx == 1
                legend('Location', 'best');
            end
            grid on;
        end
        
            if ~isempty(filePrefix)
                sgtitle(sprintf('[%s] LFP DFA Analysis - step=%.3fs', filePrefix, stepSize));
            else
                sgtitle(sprintf('LFP DFA Analysis - step=%.3fs', stepSize));
            end
        
        if ~isempty(filePrefix)
            exportgraphics(gcf, fullfile(saveDir, sprintf('%s_criticality_lfp_dfa_win%.1f.png', filePrefix, slidingWindowSize)), 'Resolution', 300);
        else
            exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_lfp_dfa_win%.1f.png', slidingWindowSize)), 'Resolution', 300);
        end
    end
end

fprintf('\n=== Analysis Complete ===\n');

