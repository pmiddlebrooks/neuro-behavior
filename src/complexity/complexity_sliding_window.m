%%
% Complexity Sliding Window Analysis Script (Lempel-Ziv Complexity)
% Computes Lempel-Ziv complexity in sliding windows for spike or LFP data
% Analyzes data using sliding window approach; saves results to data-specific folders
%
% NOTE: This script now uses the new modular function architecture.
% For new code, consider using complexity_analysis() function directly.
%
% Expected workspace variables (loaded by criticality_sliding_data_prep.m):
%   For spikes:
%     - dataType: 'reach', 'naturalistic', 'schall', or 'hong'
%     - dataMat: Neural matrix [nTimePoints x nNeurons]
%     - areas: Cell array of area names
%     - idMatIdx: Cell array of neuron indices per area
%     - opts: Options structure
%     - saveDir: Directory to save results
%   For LFP:
%     - lfpPerArea: Raw LFP data [nSamples x nAreas]
%     - opts: Options structure with fsLfp
%     - areas: Cell array of area names
%     - saveDir: Directory to save results

% =============================    Configuration    =============================

% Data source selection
if ~exist('dataSource', 'var')
    dataSource = 'spikes';  % 'spikes' or 'lfp'
end

% Flags
loadExistingResults = false;
makePlots = true;


% Normalization parameters
    nShuffles = 3;  % Number of random shuffles for normalization

% LFP-specific parameters
lfpLowpassFreq = 80;  % Low-pass filter frequency for LFP (Hz)

% Areas to analyze
if ~exist('areasToTest', 'var')
    if exist('areas', 'var')
        areasToTest = 1:length(areas);
    else
        error('areas variable not found. Please ensure data is loaded.');
    end
end

% =============================    Data Validation    =============================
fprintf('\n=== Complexity Sliding Window Analysis Setup ===\n');
fprintf('Data source: %s\n', dataSource);

if strcmp(dataSource, 'spikes')
    % Verify required variables for spike data
    requiredVars = {'dataType', 'dataMat', 'areas', 'idMatIdx', 'opts', 'saveDir'};
    for i = 1:length(requiredVars)
        if ~exist(requiredVars{i}, 'var')
            error('Required variable %s not found in workspace. Please run criticality_sliding_data_prep.m first.', requiredVars{i});
        end
    end
    
            optimalBinSize = repmat(binSize, 1, length(areas));
    
    numAreas = length(areas);
    fprintf('Number of areas: %d\n', numAreas);
    fprintf('Window size: %.2f s\n', slidingWindowSize);
    
elseif strcmp(dataSource, 'lfp')
    % Verify required variables for LFP data
    requiredVars = {'lfpPerArea', 'opts', 'areas', 'saveDir'};
    for i = 1:length(requiredVars)
        if ~exist(requiredVars{i}, 'var')
            error('Required variable %s not found in workspace. Please run criticality_sliding_data_prep.m first.', requiredVars{i});
        end
    end
    
    if ~isfield(opts, 'fsLfp')
        error('opts.fsLfp must be defined for LFP analysis.');
    end
    
    numAreas = size(lfpPerArea, 2);
    if numAreas ~= length(areas)
        warning('Number of areas in lfpPerArea (%d) does not match areas array (%d). Using lfpPerArea size.', numAreas, length(areas));
        if numAreas < length(areas)
            areas = areas(1:numAreas);
        end
    end
    
    fprintf('Number of areas: %d\n', numAreas);
    fprintf('LFP sampling rate: %.1f Hz\n', opts.fsLfp);
    fprintf('Low-pass filter: %.1f Hz\n', lfpLowpassFreq);
    fprintf('Window size: %.2f s\n', slidingWindowSize);
    
    % Calculate step size for LFP (use a fraction of window size)
    if ~exist('stepSize', 'var') || isempty(stepSize)
        stepSize = slidingWindowSize / 10;  % Default: 10 steps per window
    end
    fprintf('Step size: %.3f s\n', stepSize);
else
    error('Invalid dataSource. Must be ''spikes'' or ''lfp''.');
end

% =============================    Initialize Results    =============================
% Structure: lzComplexity{area} = [1 x numWindows]
%           lzComplexityNormalized{area} = [1 x numWindows] (normalized by shuffle)
%           lzComplexityNormalizedBernoulli{area} = [1 x numWindows] (normalized by Bernoulli)
lzComplexity = cell(1, numAreas);
lzComplexityNormalized = cell(1, numAreas);
lzComplexityNormalizedBernoulli = cell(1, numAreas);
startS = cell(1, numAreas);

% =============================    Analysis    =============================
fprintf('\n=== Processing Areas ===\n');

for a = areasToTest
    fprintf('\nProcessing area %s (%s)...\n', areas{a}, dataSource);
    tic;
    
    if strcmp(dataSource, 'spikes')
        % ========== Spike Data Analysis ==========
        aID = idMatIdx{a};
        
        % Bin data using optimal bin size for this area
        aDataMat = neural_matrix_ms_to_frames(dataMat(:, aID), optimalBinSize(a));
        numTimePoints = size(aDataMat, 1);
        
        % Calculate window and step sizes in samples
        winSamples = round(slidingWindowSize / optimalBinSize(a));
        stepSamples = round(stepSize / optimalBinSize(a));
        if stepSamples < 1
            stepSamples = 1;  % Minimum step of 1 sample
        end
        
        
        % Calculate number of windows
        numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
        fprintf('  Time points: %d, Window samples: %d, Step samples: %d, Windows: %d\n', ...
            numTimePoints, winSamples, stepSamples, numWindows);
        
        % Initialize arrays
        lzComplexity{a} = nan(1, numWindows);
        lzComplexityNormalized{a} = nan(1, numWindows);
        startS{a} = nan(1, numWindows);
        
        % Process each window
        for w = 1:numWindows
            startIdx = (w - 1) * stepSamples + 1;
            endIdx = startIdx + winSamples - 1;
            
            % Calculate center time for this window
            startS{a}(w) = (startIdx + round(winSamples/2) - 1) * optimalBinSize(a);
            
            % Extract window data [nSamples x nNeurons]
            wData = aDataMat(startIdx:endIdx, :);
            
            % Concatenate across neurons over time
            % Transpose first, then reshape to concatenate rows (across neurons at each time point)
            % This creates a sequence: neuron1_t1, neuron2_t1, ..., neuronN_t1, neuron1_t2, ...
            nNeurons = size(wData, 2);
            nSamples = size(wData, 1);
            concatenatedSeq = reshape(wData', nSamples * nNeurons, 1);
            
            % Binarize: any value > 0 becomes 1, 0 stays 0
            binarySeq = double(concatenatedSeq > 0);
            
            % Calculate Lempel-Ziv complexity
            try
                lzComplexity{a}(w) = limpel_ziv_complexity(binarySeq, 'method', 'binary');
                
                % Normalize by shuffled version: compute mean LZ complexity of shuffled sequences
                shuffledLZ = nan(1, nShuffles);
                for s = 1:nShuffles
                    % Randomly shuffle the binary sequence
                    shuffledSeq = binarySeq(randperm(length(binarySeq)));
                    shuffledLZ(s) = limpel_ziv_complexity(shuffledSeq, 'method', 'binary');
                end
                
                % Normalize: divide original by mean of shuffled
                meanShuffledLZ = nanmean(shuffledLZ);
                if meanShuffledLZ > 0
                    lzComplexityNormalized{a}(w) = lzComplexity{a}(w) / meanShuffledLZ;
                else
                    lzComplexityNormalized{a}(w) = nan;
                end
                
                % Normalize by rate-matched Bernoulli control: compute mean LZ complexity of Bernoulli sequences
                % Calculate firing rate (proportion of 1s) in the original sequence
                firingRate = mean(binarySeq);
                
                % Generate multiple Bernoulli sequences with same length and rate
                bernoulliLZ = nan(1, nShuffles);
                for s = 1:nShuffles
                    % Generate random Bernoulli sequence with same rate
                    bernoulliSeq = double(rand(length(binarySeq), 1) < firingRate);
                    bernoulliLZ(s) = limpel_ziv_complexity(bernoulliSeq, 'method', 'binary');
                end
                
                % Normalize: divide original by mean of Bernoulli
                meanBernoulliLZ = nanmean(bernoulliLZ);
                if meanBernoulliLZ > 0
                    lzComplexityNormalizedBernoulli{a}(w) = lzComplexity{a}(w) / meanBernoulliLZ;
                else
                    lzComplexityNormalizedBernoulli{a}(w) = nan;
                end
            catch ME
                fprintf('    Warning: Error computing LZ complexity for window %d: %s\n', w, ME.message);
                lzComplexity{a}(w) = nan;
                lzComplexityNormalized{a}(w) = nan;
                lzComplexityNormalizedBernoulli{a}(w) = nan;
            end
        end
        
    elseif strcmp(dataSource, 'lfp')
        % ========== LFP Data Analysis ==========
        rawSignal = lfpPerArea(:, a);
        fsRaw = opts.fsLfp;
        
        % Low-pass filter at 80 Hz
        if lfpLowpassFreq < fsRaw / 2
            filteredSignal = lowpass(rawSignal, lfpLowpassFreq, fsRaw);
        else
            warning('Low-pass frequency (%.1f Hz) >= Nyquist (%.1f Hz). Skipping filter.', ...
                lfpLowpassFreq, fsRaw / 2);
            filteredSignal = rawSignal;
        end
        
        % Calculate window and step sizes in samples
        winSamples = round(slidingWindowSize * fsRaw);
        stepSamples = round(stepSize * fsRaw);
        if stepSamples < 1
            stepSamples = 1;  % Minimum step of 1 sample
        end
        
        numTimePoints = length(filteredSignal);
        
         
        % Calculate number of windows
        numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
        fprintf('  Time points: %d, Window samples: %d, Step samples: %d, Windows: %d\n', ...
            numTimePoints, winSamples, stepSamples, numWindows);
        
        % Initialize arrays
        lzComplexity{a} = nan(1, numWindows);
        lzComplexityNormalized{a} = nan(1, numWindows);
        lzComplexityNormalizedBernoulli{a} = nan(1, numWindows);
        startS{a} = nan(1, numWindows);
        
        % Process each window
        for w = 1:numWindows
            startIdx = (w - 1) * stepSamples + 1;
            endIdx = startIdx + winSamples - 1;
            
            % Calculate center time for this window
            startS{a}(w) = (startIdx + round(winSamples/2) - 1) / fsRaw;
            
            % Extract window signal
            wSignal = filteredSignal(startIdx:endIdx);
            
            % Binarize: compare each sample to window mean
            windowMean = mean(wSignal);
            binarySeq = double(wSignal > windowMean);
            
            % Calculate Lempel-Ziv complexity
            try
                lzComplexity{a}(w) = limpel_ziv_complexity(binarySeq, 'method', 'binary');
                
                % Normalize by shuffled version: compute mean LZ complexity of shuffled sequences
                shuffledLZ = nan(1, nShuffles);
                for s = 1:nShuffles
                    % Randomly shuffle the binary sequence
                    shuffledSeq = binarySeq(randperm(length(binarySeq)));
                    shuffledLZ(s) = limpel_ziv_complexity(shuffledSeq, 'method', 'binary');
                end
                
                % Normalize: divide original by mean of shuffled
                meanShuffledLZ = nanmean(shuffledLZ);
                if meanShuffledLZ > 0
                    lzComplexityNormalized{a}(w) = lzComplexity{a}(w) / meanShuffledLZ;
                else
                    lzComplexityNormalized{a}(w) = nan;
                end
                
                % Normalize by rate-matched Bernoulli control: compute mean LZ complexity of Bernoulli sequences
                % Calculate firing rate (proportion of 1s) in the original sequence
                firingRate = mean(binarySeq);
                
                % Generate multiple Bernoulli sequences with same length and rate
                bernoulliLZ = nan(1, nShuffles);
                for s = 1:nShuffles
                    % Generate random Bernoulli sequence with same rate
                    bernoulliSeq = double(rand(length(binarySeq), 1) < firingRate);
                    bernoulliLZ(s) = limpel_ziv_complexity(bernoulliSeq, 'method', 'binary');
                end
                
                % Normalize: divide original by mean of Bernoulli
                meanBernoulliLZ = nanmean(bernoulliLZ);
                if meanBernoulliLZ > 0
                    lzComplexityNormalizedBernoulli{a}(w) = lzComplexity{a}(w) / meanBernoulliLZ;
                else
                    lzComplexityNormalizedBernoulli{a}(w) = nan;
                end
            catch ME
                fprintf('    Warning: Error computing LZ complexity for window %d: %s\n', w, ME.message);
                lzComplexity{a}(w) = nan;
                lzComplexityNormalized{a}(w) = nan;
                lzComplexityNormalizedBernoulli{a}(w) = nan;
            end
        end
    end
    
    fprintf('  Area %s completed in %.1f minutes\n', areas{a}, toc/60);
end

% =============================    Save Results    =============================
results = struct();
results.dataSource = dataSource;
results.areas = areas;
results.startS = startS;
results.lzComplexity = lzComplexity;
results.lzComplexityNormalized = lzComplexityNormalized;
results.lzComplexityNormalizedBernoulli = lzComplexityNormalizedBernoulli;
results.params.slidingWindowSize = slidingWindowSize;
results.params.stepSize = stepSize;
results.params.nShuffles = nShuffles;

if strcmp(dataSource, 'spikes')
    results.params.optimalBinSize = optimalBinSize;
    if exist('dataType', 'var')
        results.dataType = dataType;
    end
elseif strcmp(dataSource, 'lfp')
    results.params.lfpLowpassFreq = lfpLowpassFreq;
    results.params.fsLfp = opts.fsLfp;
end

% Create results path
if ~exist('saveDir', 'var') || isempty(saveDir)
    if ~exist('paths', 'var')
        paths = get_paths;
    end
    if strcmp(dataSource, 'spikes')
        saveDir = fullfile(paths.saveDataPath, 'complexity');
    else
        saveDir = fullfile(paths.saveDataPath, 'complexity_lfp');
    end
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end
end

% Create filename
if strcmp(dataSource, 'spikes')
    if exist('dataType', 'var') && strcmp(dataType, 'reach') && exist('sessionName', 'var') && ~isempty(sessionName)
        resultsPath = fullfile(saveDir, sprintf('complexity_sliding_window_%s_win%.1f_%s.mat', dataSource, slidingWindowSize, sessionName));
    elseif exist('dataType', 'var')
        resultsPath = fullfile(saveDir, sprintf('complexity_sliding_window_%s_win%.1f_%s.mat', dataSource, slidingWindowSize, dataType));
    else
        resultsPath = fullfile(saveDir, sprintf('complexity_sliding_window_%s_win%.1f.mat', dataSource, slidingWindowSize));
    end
else
    if exist('sessionName', 'var') && ~isempty(sessionName)
        resultsPath = fullfile(saveDir, sprintf('complexity_sliding_window_%s_win%.1f_%s.mat', dataSource, slidingWindowSize, sessionName));
    else
        resultsPath = fullfile(saveDir, sprintf('complexity_sliding_window_%s_win%.1f.mat', dataSource, slidingWindowSize));
    end
end

save(resultsPath, 'results');
fprintf('\nSaved results to: %s\n', resultsPath);

% =============================    Plotting    =============================
if makePlots
    fprintf('\n=== Creating Plots ===\n');
    
    % Extract filename prefix
    if exist('sessionName', 'var') && ~isempty(sessionName)
        filePrefix = sessionName(1:min(8, length(sessionName)));
    elseif exist('dataBaseName', 'var') && ~isempty(dataBaseName)
        filePrefix = dataBaseName(1:min(8, length(dataBaseName)));
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
    allStartS = [];
    allLZ = [];
    allLZNorm = [];
    
    for a = areasToTest
        if ~isempty(startS{a})
            allStartS = [allStartS, startS{a}];
        end
        if ~isempty(lzComplexity{a})
            allLZ = [allLZ, lzComplexity{a}(~isnan(lzComplexity{a}))];
        end
        if ~isempty(lzComplexityNormalized{a})
            allLZNorm = [allLZNorm, lzComplexityNormalized{a}(~isnan(lzComplexityNormalized{a}))];
        end
    end
    
    % Determine axis limits
    if ~isempty(allStartS)
        xMin = min(allStartS);
        xMax = max(allStartS);
    else
        xMin = 0;
        xMax = 100;
    end
    
    if ~isempty(allLZ)
        yMinLZ = min(allLZ);
        yMaxLZ = max(allLZ);
        yRangeLZ = yMaxLZ - yMinLZ;
        yMinLZ = max(0, yMinLZ - 0.05 * yRangeLZ);
        yMaxLZ = yMaxLZ + 0.05 * yRangeLZ;
    else
        yMinLZ = 0;
        yMaxLZ = 100;
    end
    
    if ~isempty(allLZNorm)
        yMinLZNorm = min(allLZNorm);
        yMaxLZNorm = max(allLZNorm);
        yRangeLZNorm = yMaxLZNorm - yMinLZNorm;
        yMinLZNorm = max(0, yMinLZNorm - 0.05 * yRangeLZNorm);
        yMaxLZNorm = yMaxLZNorm + 0.05 * yRangeLZNorm;
    else
        yMinLZNorm = 0;
        yMaxLZNorm = 2;
    end
    
    % ========== LZ Complexity Plot (Normalized) ==========
    figure(914); clf;
    set(gcf, 'Position', targetPos);
    numRows = length(areasToTest);
    ha = tight_subplot(numRows, 1, [0.035 0.04], [0.03 0.08], [0.08 0.04]);
    
    % Define colors for each area
    areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1]};  % Red, Green, Blue, Magenta for M23, M56, DS, VS
    
    for idx = 1:length(areasToTest)
        a = areasToTest(idx);
        axes(ha(idx)); hold on;
        
        % Plot normalized LZ complexity (primary)
        if ~isempty(lzComplexityNormalized{a}) && ~isempty(startS{a})
            validIdx = ~isnan(lzComplexityNormalized{a});
            if any(validIdx)
                plot(startS{a}(validIdx), lzComplexityNormalized{a}(validIdx), '-', 'Color', areaColors{min(a, length(areaColors))}, 'LineWidth', 2, ...
                    'DisplayName', areas{a});
            end
        end
        
        % Add reference line at 1.0 (shuffled mean)
        yline(1.0, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'DisplayName', 'Shuffled mean');
        
        % Add vertical lines at reach onsets (only for reach data)
        if strcmp(dataSource, 'spikes') && exist('dataType', 'var') && strcmp(dataType, 'reach')
            if ~isempty(startS{a}) && exist('reachStart', 'var')
                plotTimeRange = [startS{a}(1), startS{a}(end)];
                reachOnsetsInRange = reachStart(reachStart >= plotTimeRange(1) & reachStart <= plotTimeRange(2));
                
                if ~isempty(reachOnsetsInRange)
                    for i = 1:length(reachOnsetsInRange)
                        xline(reachOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                    end
                    if exist('startBlock2', 'var')
                        xline(startBlock2, 'Color', [1 0 0], 'LineWidth', 3);
                    end
                end
            end
        end
        
        title(sprintf('%s - Normalized Lempel-Ziv Complexity', areas{a}));
        xlabel('Time (s)');
        ylabel('Normalized LZ Complexity');
        if ~isempty(startS{a})
            xlim([startS{a}(1), startS{a}(end)]);
        else
            xlim([xMin, xMax]);
        end
        ylim([yMinLZNorm, yMaxLZNorm]);
        set(gca, 'XTickLabelMode', 'auto');
        set(gca, 'YTickLabelMode', 'auto');
        grid on;
    end
    
    % Create title
    if strcmp(dataSource, 'spikes') && exist('dataType', 'var')
        if ~isempty(filePrefix)
            sgtitle(sprintf('[%s] %s Normalized Lempel-Ziv Complexity - %s, win=%.2fs, step=%.3fs, nShuffles=%d', ...
                filePrefix, dataType, dataSource, slidingWindowSize, stepSize, nShuffles));
        else
            sgtitle(sprintf('%s Normalized Lempel-Ziv Complexity - %s, win=%.2fs, step=%.3fs, nShuffles=%d', ...
                dataType, dataSource, slidingWindowSize, stepSize, nShuffles));
        end
    else
        if ~isempty(filePrefix)
            sgtitle(sprintf('[%s] Normalized Lempel-Ziv Complexity - %s, win=%.2fs, step=%.3fs, nShuffles=%d', ...
                filePrefix, dataSource, slidingWindowSize, stepSize, nShuffles));
        else
            sgtitle(sprintf('Normalized Lempel-Ziv Complexity - %s, win=%.2fs, step=%.3fs, nShuffles=%d', ...
                dataSource, slidingWindowSize, stepSize, nShuffles));
        end
    end
    
    % Save figure
    if ~isempty(filePrefix)
        exportgraphics(gcf, fullfile(saveDir, sprintf('%s_complexity_%s_win%.1f.png', filePrefix, dataSource, slidingWindowSize)), 'Resolution', 300);
    else
        exportgraphics(gcf, fullfile(saveDir, sprintf('complexity_%s_win%.1f.png', dataSource, slidingWindowSize)), 'Resolution', 300);
    end
end

fprintf('\n=== Analysis Complete ===\n');

