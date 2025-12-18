%%
% Criticality Sliding Window Analysis Script for LFP Power Bands (d2)
% Performs d2 criticality analysis on LFP binnedEnvelopes data
% Analyzes each power band separately using sliding window approach
%
% Expected workspace variables:
%   - binnedEnvelopes: [nFrames x nBands*nAreas] matrix of binned LFP envelopes
%   - bands: Cell array of frequency bands (e.g., {'alpha', [8 13]; 'beta', [13 30]})
%   - areas: Cell array of area names (e.g., {'M23', 'M56', 'DS', 'VS'})
%   - opts: Options structure with frameSize, fsLfp, etc.
%   - saveDir: Directory to save results

% =============================    Configuration    =============================

% Flags
loadExistingResults = false;
makePlots = true;
plotBinnedEnvelopes = true;  % Plot results from binnedEnvelopes (power bands)
plotRawLfp = false;           % Plot results from raw LFP
plotDFA = true;              % Plot DFA results on right y-axis

% Analysis flags
analyzeD2 = true;      % compute d2
analyzeDFA = true;     % compute DFA alpha

% Permutation testing flags
enablePermutations = false;  % Set to true to perform circular permutation testing
nShuffles = 3;  % Number of circular permutations to perform

% Analysis parameters
minSegmentLength = 50;

% Get binSizes from workspace (calculated in data prep script)
% Use max bin size for d2StepSize to ensure consistent stepping across bands
if exist('binSizes', 'var') && ~isempty(binSizes)
    maxBinSize = max([binSizes(:); lfpBinSize(:)]);
    d2StepSize = maxBinSize;  % Step size in seconds (how much to slide the window)
    fprintf('Using frequency-dependent bin sizes. Max bin size: %.3f s, d2StepSize: %.3f s\n', maxBinSize, d2StepSize);
else
    % Fallback if binSizes not provided (backward compatibility)
    if exist('binSize', 'var')
        d2StepSize = binSize * 4;
        maxBinSize = binSize;
    else
        error('Neither binSizes nor binSize found in workspace. Please run criticality_sliding_data_prep.m first.');
    end
end

% Optimal bin/window size selection mode
useOptimalBinWindowFunction = false;  % For LFP, we use frequency-dependent bin sizes
optimalWindowSize = slidingWindowSize;  % Window size in seconds

% Areas to analyze
if ~exist('areas', 'var')
    % Default areas if not provided
    areas = {'M23', 'M56', 'DS', 'VS'};
end
if ~exist('areasToTest', 'var')
    areasToTest = 1:length(areas);  
end

% Optimal bin/window size search parameters
pOrder = 10;
critType = 2;


% =============================    Data Validation    =============================
% Verify that required variables are in workspace
requiredVars = {'binnedEnvelopes', 'bands'};
for i = 1:length(requiredVars)
    if ~exist(requiredVars{i}, 'var')
        error('Required variable %s not found in workspace.', requiredVars{i});
    end
end
% opts is optional (for backward compatibility)
if ~exist('opts', 'var')
    opts = struct();
end

% Determine number of bands and areas
numBands = size(bands, 1);

% Convert binnedEnvelopes to cell structure if it's a matrix
% Expected input: either cell array {area} = [nFrames x numBands] or matrix [nFrames x numBands*nAreas]
if ~iscell(binnedEnvelopes)
    % Convert matrix to cell structure
    numCols = size(binnedEnvelopes, 2);
    numAreas = floor(numCols / numBands);
    
    if numCols ~= numBands * numAreas
        warning('binnedEnvelopes has %d columns, expected multiple of %d (numBands). Adjusting...', numCols, numBands);
        numAreas = floor(numCols / numBands);
        if numAreas < 1
            error('Not enough columns in binnedEnvelopes for the number of bands');
        end
    end
    
    % Convert to cell structure: binnedEnvelopes{area} = [nFrames x numBands]
    binnedEnvelopesCell = cell(1, numAreas);
    for a = 1:numAreas
        areaBandIdx = (1:numBands) + numBands * (a - 1);
        binnedEnvelopesCell{a} = binnedEnvelopes(:, areaBandIdx);
    end
    binnedEnvelopes = binnedEnvelopesCell;
    clear binnedEnvelopesCell;
end

% Now binnedEnvelopes is a cell array: {area} = [nFrames x numBands]
numAreas = length(binnedEnvelopes);
if numAreas ~= length(areas)
    warning('Number of areas in binnedEnvelopes (%d) does not match areas array (%d). Using binnedEnvelopes length.', numAreas, length(areas));
    if numAreas < length(areas)
        areas = areas(1:numAreas);
    end
end

% Get number of frames from first area, first band (will vary by band due to different bin sizes)
% Note: Each band may have different number of frames due to different bin sizes
if iscell(binnedEnvelopes{1}) && iscell(binnedEnvelopes{1}{1})
    % New structure: binnedEnvelopes{area}{band} = [nFrames_b x 1]
    numFrames = size(binnedEnvelopes{1}{1}, 1);  % Just for reference, will vary by band
else
    % Old structure: binnedEnvelopes{area} = [nFrames x numBands]
    numFrames = size(binnedEnvelopes{1}, 1);
end

% Create results path
if ~exist('saveDir', 'var') || isempty(saveDir)
    saveDir = fullfile(paths.saveDataPath, 'criticality_lfp');
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end
end

if ~exist('sessionName', 'var') || isempty(sessionName)
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_ar_lfp_win%d.mat', slidingWindowSize));
else
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_ar_lfp_win%d_%s.mat', slidingWindowSize, sessionName));
end

% =============================    Analysis    =============================
fprintf('\n=== LFP d2 Analysis ===\n');
fprintf('Number of bands: %d\n', numBands);
fprintf('Number of areas: %d\n', numAreas);

% Determine total session duration in seconds to ensure time alignment
if exist('lfpPerArea', 'var') && ~isempty(lfpPerArea)
    totalSessionDuration = size(lfpPerArea, 1) / opts.fsLfp;
else
    % Fallback to binned envelopes if raw LFP not available
    % (assuming first area/band represents session duration)
    if iscell(binnedEnvelopes{1})
        if exist('binSizes', 'var') && ~isempty(binSizes)
            totalSessionDuration = length(binnedEnvelopes{1}{1}) * binSizes(1);
        elseif exist('binSize', 'var')
            totalSessionDuration = length(binnedEnvelopes{1}{1}) * binSize;
        else
            % Absolute fallback
            totalSessionDuration = length(binnedEnvelopes{1}{1}) * 0.05; 
        end
    else
        if exist('binSize', 'var')
            totalSessionDuration = size(binnedEnvelopes{1}, 1) * binSize;
        else
            totalSessionDuration = size(binnedEnvelopes{1}, 1) * 0.05;
        end
    end
end

% Define fixed window start times in seconds to ensure all traces have same length and alignment
windowStartTimesSeconds = 0 : d2StepSize : (totalSessionDuration - slidingWindowSize);
numWindowsGlobal = length(windowStartTimesSeconds);
windowCenterTimesSeconds = windowStartTimesSeconds + slidingWindowSize/2;

if exist('binSizes', 'var')
    fprintf('Frequency-dependent bin sizes: ');
    for b = 1:numBands
        fprintf('%s: %.3f s (%.1f ms)', bands{b,1}, binSizes(b), binSizes(b)*1000);
        if b < numBands
            fprintf(', ');
        end
    end
    fprintf('\n');
    fprintf('d2StepSize (max bin size): %.3f s (%.1f ms)\n', d2StepSize, d2StepSize*1000);
else
    if exist('numFrames', 'var')
        fprintf('Number of frames: %d\n', numFrames);
    end
    if exist('binSize', 'var')
        fprintf('Frame size: %.3f s\n', binSize);
    end
end
fprintf('Total windows for analysis: %d (aligned across all traces)\n', numWindowsGlobal);

% Initialize results
% Structure: d2{area}{band} = [1 x numWindows]
d2 = cell(1, numAreas);
dfa = cell(1, numAreas); % DFA alpha for power bands
d2Permuted = cell(1, numAreas);
startS = cell(1, numAreas);

% Initialize storage for raw LFP d2 results if lfpBinSize is provided
if exist('lfpBinSize', 'var') && ~isempty(lfpBinSize)
    numLfpBins = length(lfpBinSize);
    d2Lfp = cell(1, numAreas);
    dfaLfp = cell(1, numAreas); % DFA alpha for raw LFP
    startSLfp = cell(1, numAreas);
else
    numLfpBins = 0;
end

for a = 1:numAreas
    d2{a} = cell(1, numBands);
    dfa{a} = cell(1, numBands);
    d2Permuted{a} = cell(1, numBands);
    startS{a} = cell(1, numBands);
    if numLfpBins > 0
        d2Lfp{a} = cell(1, numLfpBins);
        dfaLfp{a} = cell(1, numLfpBins);
        startSLfp{a} = cell(1, numLfpBins);
    end
end

% DFA analysis parameters
dfaEnvBinSize = 0.02; % 50 Hz for power envelopes
dfaEnvWinSamples_min = 1000;
dfaEnvWinSize = max(dfaEnvWinSamples_min * dfaEnvBinSize, 30); % seconds

dfaLfpWinSamples_min = 2000;
% For LFP, we will use lfpBinSize from configuration (e.g., 0.005s or 200Hz)

% Process each area
for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});
    
    % Extract this area's binned envelopes (now stored in cell structure)
    if length(binnedEnvelopes) < a || isempty(binnedEnvelopes{a})
        fprintf('Skipping: No binned envelopes data for area %s...\n', areas{a});
        continue
    end
    areaEnvelopes = binnedEnvelopes{a};  % Either [nFrames x numBands] or {band} = [nFrames_b x 1]
        
    % Process each band separately
    for b = 1:numBands
        fprintf('  Processing band %d (%s)...\n', b, bands{b, 1});
        tic;
        
        % Extract this band's time series, bin size, and time points
            % New structure: areaEnvelopes{band} = [nFrames_b x 1]
            bandSignal = areaEnvelopes{b};  % [nFrames_b x 1]
            bandBinSize = binSizes(b);  % Use band-specific bin size
            
            % Get time points for this band if available
            if exist('timePoints', 'var') && ~isempty(timePoints) && ...
                    length(timePoints) >= a && ~isempty(timePoints{a}) && ...
                    length(timePoints{a}) >= b && ~isempty(timePoints{a}{b})
                bandTimePoints = timePoints{a}{b};  % [nFrames_b x 1]
                useTimePoints = true;
            else
                useTimePoints = false;
            end
        
        numFrames_b = length(bandSignal);
        
        % Calculate window parameters using band-specific bin size
        winSamples = round(slidingWindowSize / bandBinSize);
        
        % Skip this band if there aren't enough samples per window
        if winSamples < minSegmentLength
            fprintf('    Skipping: Not enough data in band %d (%s)...\n', b, bands{b, 1});
            continue
        end
        
        % Initialize arrays for this band
        d2{a}{b} = nan(1, numWindowsGlobal);
        startS{a}{b} = windowCenterTimesSeconds;
        
        % Sliding window analysis
        for w = 1:numWindowsGlobal
            % Calculate start index based on global window start time
            startIdx = round(windowStartTimesSeconds(w) / bandBinSize) + 1;
            endIdx = startIdx + winSamples - 1;
            
            % Ensure we don't exceed data bounds
            if endIdx > numFrames_b
                % If we can't fit a full window, it will remain NaN
                continue;
            end
            
            % Extract window data
            wSignal = bandSignal(startIdx:endIdx);
            
            % Compute d2
            if analyzeD2
                try
                    [varphi, ~] = myYuleWalker3(wSignal, pOrder);
                    d2{a}{b}(w) = getFixedPointDistance2(pOrder, critType, varphi);
                catch ME
                    fprintf('    Warning: d2 calculation failed for window %d: %s\n', w, ME.message);
                    d2{a}{b}(w) = nan;
                end
            end

            % Compute DFA alpha for power bands
            if analyzeDFA
                try
                    % Define DFA window for power envelopes
                    dfaWinSamples = round(dfaEnvWinSize / bandBinSize);
                    % Center DFA window on current sliding window center
                    dfaStartIdx = round(windowCenterTimesSeconds(w) / bandBinSize) - round(dfaWinSamples/2) + 1;
                    dfaEndIdx = dfaStartIdx + dfaWinSamples - 1;
                    
                    if dfaStartIdx >= 1 && dfaEndIdx <= numFrames_b
                        dfaSignal = bandSignal(dfaStartIdx:dfaEndIdx);
                        
                        % Downsample to 50 Hz (0.02s bins) if current binSize is smaller
                        if bandBinSize < dfaEnvBinSize
                            % Simple bin averaging for downsampling
                            dsFactor = round(dfaEnvBinSize / bandBinSize);
                            dfaSignal_ds = arrayfun(@(i) mean(dfaSignal(i:min(i+dsFactor-1, end))), 1:dsFactor:length(dfaSignal))';
                        else
                            dfaSignal_ds = dfaSignal;
                        end
                        
                        dfa{a}{b}(w) = compute_DFA(dfaSignal_ds, false);
                    end
                catch
                    dfa{a}{b}(w) = nan;
                end
            end
        end
        
        fprintf('    Band %d completed in %.1f minutes\n', b, toc/60);
        
        % Permutations for this band
        if enablePermutations
            fprintf('    Running %d phase-randomized permutations per window for band %d...\n', nShuffles, b);
            ticPerm = tic;
            
            % Initialize storage for permutation results [numWindowsGlobal x nShuffles]
            d2Permuted{a}{b} = nan(numWindowsGlobal, nShuffles);
            
            % Permute each window independently
            for w = 1:numWindowsGlobal
                % Skip if main d2 calculation failed or was skipped
                if isnan(d2{a}{b}(w))
                    continue;
                end
                
                startIdx = round(windowStartTimesSeconds(w) / bandBinSize) + 1;
                endIdx = startIdx + winSamples - 1;
                
                % Extract this window's data
                windowData = bandSignal(startIdx:endIdx);
                
                % For each shuffle, phase-randomize this window's data
                for shuffle = 1:nShuffles
                    % Phase randomization: preserves power spectrum (autocorrelation) 
                    % but randomizes phase relationships (breaks critical dynamics)
                    permutedWindowData = phase_randomize_signal(windowData);
                    
                    % Compute d2 on permuted data
                    try
                        [varphiPerm, ~] = myYuleWalker3(permutedWindowData, pOrder);
                        d2Permuted{a}{b}(w, shuffle) = getFixedPointDistance2(pOrder, critType, varphiPerm);
                    catch ME
                        % Silently skip failed permutations
                        d2Permuted{a}{b}(w, shuffle) = nan;
                    end
                end
            end
            
            fprintf('    Permutations completed in %.1f minutes\n', toc(ticPerm)/60);
        end
    end

    % Process raw LFP for this area for each specified bin size
    if numLfpBins > 0 && exist('lfpPerArea', 'var') && ~isempty(lfpPerArea)
        fprintf('  Processing raw LFP for area %s...\n', areas{a});
        rawSignal = lfpPerArea(:, a);
        fsRaw = opts.fsLfp;

        for lb = 1:numLfpBins
            targetBinSize = lfpBinSize(lb);
            fprintf('    Binning raw LFP with size %.3f s...\n', targetBinSize);
            
            % Simple binning by averaging
            samplesPerBin = round(targetBinSize * fsRaw);
            numBins = floor(length(rawSignal) / samplesPerBin);
            
            if numBins < minSegmentLength
                fprintf('      Skipping: Not enough bins for bin size %.3f s\n', targetBinSize);
                continue;
            end
            
            % Efficient binning using reshape and mean
            binnedSignal = mean(reshape(rawSignal(1:numBins*samplesPerBin), samplesPerBin, numBins), 1)';
            
            % Sliding window analysis on binned raw LFP
            winSamples = round(slidingWindowSize / targetBinSize);
            
            if winSamples < minSegmentLength
                fprintf('      Skipping: Not enough samples per window for bin size %.3f s\n', targetBinSize);
                continue;
            end
            
            d2Lfp{a}{lb} = nan(1, numWindowsGlobal);
            startSLfp{a}{lb} = windowCenterTimesSeconds;
            
            for w = 1:numWindowsGlobal
                % Calculate start index based on global window start time
                startIdx = round(windowStartTimesSeconds(w) / targetBinSize) + 1;
                endIdx = startIdx + winSamples - 1;
                
                % Ensure we don't exceed data bounds
                if endIdx > numBins
                    continue;
                end
                
                % Extract window data and compute d2
                wSignal = binnedSignal(startIdx:endIdx);
                try
                    if analyzeD2
                        [varphi, ~] = myYuleWalker3(wSignal, pOrder);
                        d2Lfp{a}{lb}(w) = getFixedPointDistance2(pOrder, critType, varphi);
                    end
                catch
                    d2Lfp{a}{lb}(w) = nan;
                end

                % Compute DFA alpha for raw LFP
                if analyzeDFA
                    try
                        % Use lfpBinSize (targetBinSize) for DFA
                        % make the window size the maximum size that either accomodates 2000 bins or spans 30 sec.
                        dfaWinSamples = max(dfaLfpWinSamples_min, round(30 / targetBinSize));
                        
                        % Center DFA window on current sliding window center
                        dfaStartIdx = round(windowCenterTimesSeconds(w) / targetBinSize) - round(dfaWinSamples/2) + 1;
                        dfaEndIdx = dfaStartIdx + dfaWinSamples - 1;
                        
                        if dfaStartIdx >= 1 && dfaEndIdx <= numBins
                            dfaSignal = binnedSignal(dfaStartIdx:dfaEndIdx);
                            dfaLfp{a}{lb}(w) = compute_DFA(dfaSignal, false);
                        end
                    catch
                        dfaLfp{a}{lb}(w) = nan;
                    end
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
results.d2 = d2;
results.dfa = dfa;
results.startS = startS;
results.optimalWindowSize = optimalWindowSize;
if numLfpBins > 0
    results.d2Lfp = d2Lfp;
    results.dfaLfp = dfaLfp;
    results.startSLfp = startSLfp;
    results.lfpBinSize = lfpBinSize;
end
results.d2StepSize = d2StepSize;
results.d2WindowSize = slidingWindowSize;
results.params.slidingWindowSize = slidingWindowSize;
results.params.analyzeD2 = analyzeD2;
results.params.pOrder = pOrder;
results.params.critType = critType;
if exist('binSizes', 'var')
    results.binSizes = binSizes;  % Store frequency-dependent bin sizes
    results.params.binSizes = binSizes;
elseif exist('binSize', 'var')
    results.params.binSize = binSize;
end

% Save permutation results
if enablePermutations
    results.enablePermutations = true;
    results.nShuffles = nShuffles;
    results.d2Permuted = d2Permuted;
    
    % Calculate mean and SEM for permutations
    d2PermutedMean = cell(1, numAreas);
    d2PermutedSEM = cell(1, numAreas);
    
    for a = 1:numAreas
        d2PermutedMean{a} = cell(1, numBands);
        d2PermutedSEM{a} = cell(1, numBands);
        for b = 1:numBands
            if ~isempty(d2Permuted{a}{b})
                d2PermutedMean{a}{b} = nanmean(d2Permuted{a}{b}, 2);
                d2PermutedSEM{a}{b} = nanstd(d2Permuted{a}{b}, 0, 2) / sqrt(nShuffles);
            else
                d2PermutedMean{a}{b} = [];
                d2PermutedSEM{a}{b} = [];
            end
        end
    end
    
    results.d2PermutedMean = d2PermutedMean;
    results.d2PermutedSEM = d2PermutedSEM;
else
    results.enablePermutations = false;
    results.nShuffles = 0;
end

save(resultsPath, 'results');
fprintf('Saved LFP d2 results to %s\n', resultsPath);

%% =============================    Plotting    =============================
if makePlots
    % Extract filename prefix for titles and filenames
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
    
    % Detect monitors and size figure to full screen (prefer second monitor if present)
    monitorPositions = get(0, 'MonitorPositions');
    monitorOne = monitorPositions(1, :);
    monitorTwo = monitorPositions(size(monitorPositions, 1), :);
    if size(monitorPositions, 1) >= 2
        targetPos = monitorTwo;
    else
        targetPos = monitorOne;
    end
    
    % First, collect all data to determine axis limits (including permuted data)
    allStartS = [];
    allD2 = [];
    
    for a = areasToTest
        % Include binned envelope results if requested
        if plotBinnedEnvelopes
            for b = 1:numBands
                % Check if startS{a}{b} exists, is numeric, and is not empty
                if length(startS) >= a && length(startS{a}) >= b && ...
                        isnumeric(startS{a}{b}) && ~isempty(startS{a}{b})
                    allStartS = [allStartS(:); startS{a}{b}(~isnan(startS{a}{b}))'];
                end
                % Check if d2{a}{b} exists, is numeric, and is not empty
                if length(d2) >= a && length(d2{a}) >= b && ...
                        isnumeric(d2{a}{b}) && ~isempty(d2{a}{b})
                    allD2 = [allD2(:); d2{a}{b}(~isnan(d2{a}{b}(:)))'];
                end
                
                % Include permuted data in axis limits
                if enablePermutations && exist('d2Permuted', 'var') && ...
                        length(d2Permuted) >= a && length(d2Permuted{a}) >= b && ...
                        isnumeric(d2Permuted{a}{b}) && ~isempty(d2Permuted{a}{b})
                    permutedMean = nanmean(d2Permuted{a}{b}, 2);
                    permutedStd = nanstd(d2Permuted{a}{b}, 0, 2);
                    validIdx = ~isnan(permutedMean) & ~isnan(permutedStd);
                    if any(validIdx)
                        permutedVals = [permutedMean(validIdx) + permutedStd(validIdx); permutedMean(validIdx) - permutedStd(validIdx)];
                        allD2 = [allD2(:); permutedVals(:)];
                    end
                end
            end
        end
        
        % Include raw LFP results if requested
        if plotRawLfp && exist('d2Lfp', 'var') && length(d2Lfp) >= a && ~isempty(d2Lfp{a})
            for lb = 1:length(d2Lfp{a})
                if ~isempty(d2Lfp{a}{lb}) && ~isempty(startSLfp{a}{lb})
                    allStartS = [allStartS(:); startSLfp{a}{lb}(~isnan(startSLfp{a}{lb}))'];
                    allD2 = [allD2(:); d2Lfp{a}{lb}(~isnan(d2Lfp{a}{lb}))'];
                end
            end
        end
    end
    
    % Determine axis limits
    if ~isempty(allStartS)
        xMin = min(allStartS);
        xMax = max(allStartS);
    else
        xMin = 0;
        xMax = 1;
    end
    
    if ~isempty(allD2)
        yMinD2 = min(allD2(:));
        yMaxD2 = max(allD2(:));
        % Add small padding
        yRangeD2 = yMaxD2 - yMinD2;
        yMinD2 = yMinD2 - 0.05 * yRangeD2;
        yMaxD2 = yMaxD2 + 0.05 * yRangeD2;
    else
        yMinD2 = 0;
        yMaxD2 = 0.5;
    end
    
    % Create figure with one row per area
    figure(910); clf;
    set(gcf, 'Position', targetPos);
    numRows = length(areasToTest);
    ha = tight_subplot(numRows, 1, [0.035 0.04], [0.03 0.08], [0.08 0.04]);
    
    % Define colors for each band
    bandColors = lines(numBands);  % Use MATLAB's lines colormap
    
    % Plot each area
    for idx = 1:length(areasToTest)
        a = areasToTest(idx);
        axes(ha(idx)); hold on;
        
        % Plot raw LFP d2 for each bin size in grayscale (light to dark)
        if plotRawLfp && exist('d2Lfp', 'var') && length(d2Lfp) >= a && ~isempty(d2Lfp{a})
            yyaxis left;
            numLfpBins = length(d2Lfp{a});
            grayColors = repmat(linspace(0.8, 0, numLfpBins)', 1, 3);
            for lb = 1:numLfpBins
                if ~isempty(d2Lfp{a}{lb}) && ~isempty(startSLfp{a}{lb})
                    plot(startSLfp{a}{lb}, d2Lfp{a}{lb}, '-', 'Color', grayColors(lb, :), 'LineWidth', 2, ...
                        'DisplayName', sprintf('Raw LFP (%.3fs bin)', lfpBinSize(lb)));
                end
            end
        end

        % Plot d2 for each band on the same axis
        if plotBinnedEnvelopes
            yyaxis left;
            for b = 1:numBands
                if ~isempty(d2{a}{b}) && ~isempty(startS{a}{b})
                    if exist('binSizes', 'var')
                        bandBinSize = binSizes(b);
                    else
                        bandBinSize = binSize; % fallback
                    end
                    plot(startS{a}{b}, d2{a}{b}, '-', 'Color', bandColors(b, :), 'LineWidth', 2, ...
                        'DisplayName', sprintf('%s (%.3fs bin)', bands{b, 1}, bandBinSize));
                end
                
                % Plot permutation mean ± std per window if available
                if enablePermutations && exist('d2Permuted', 'var') && ~isempty(d2Permuted{a}{b})
                    if exist('binSizes', 'var')
                        bandBinSize = binSizes(b);
                    else
                        bandBinSize = binSize; % fallback
                    end
                    % Calculate mean and std for each window across shuffles
                    permutedMean = nanmean(d2Permuted{a}{b}, 2);
                    permutedStd = nanstd(d2Permuted{a}{b}, 0, 2);
                    
                    % Find valid indices (where we have data)
                    validIdx = ~isnan(permutedMean) & ~isnan(permutedStd) & ~isnan(startS{a}{b}(:));
                    if any(validIdx)
                        xFill = startS{a}{b}(validIdx);
                        yMean = permutedMean(validIdx);
                        yStd = permutedStd(validIdx);
                        
                        % Ensure row vectors for fill
                        if iscolumn(xFill); xFill = xFill'; end
                        if iscolumn(yMean); yMean = yMean'; end
                        if iscolumn(yStd); yStd = yStd'; end
                        
                        % Plot shaded region (mean ± std) with lighter color
                        bandColorLight = bandColors(b, :) * 0.7 + 0.3;  % Lighten the color
                        fill([xFill, fliplr(xFill)], ...
                             [yMean + yStd, fliplr(yMean - yStd)], ...
                             bandColorLight, 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
                             'DisplayName', sprintf('%s Permuted mean ± std (%.3fs bin)', bands{b, 1}, bandBinSize));
                        
                        % Plot mean line with dashed style
                        plot(startS{a}{b}(validIdx), permutedMean(validIdx), '--', ...
                            'Color', bandColors(b, :) * 0.8, 'LineWidth', 1, ...
                            'DisplayName', sprintf('%s Permuted mean (%.3fs bin)', bands{b, 1}, bandBinSize));
                    end
                end
            end
        end
        
        % Plot DFA results on right y-axis if requested
        if plotDFA
            yyaxis right;
            if plotRawLfp && exist('dfaLfp', 'var') && length(dfaLfp) >= a && ~isempty(dfaLfp{a})
                for lb = 1:length(dfaLfp{a})
                    if ~isempty(dfaLfp{a}{lb})
                        plot(windowCenterTimesSeconds, dfaLfp{a}{lb}, ':', 'Color', grayColors(lb, :), 'LineWidth', 1.5, ...
                            'DisplayName', sprintf('DFA Raw LFP (%.3fs bin)', lfpBinSize(lb)));
                    end
                end
            end
            
            if plotBinnedEnvelopes && exist('dfa', 'var') && length(dfa) >= a && ~isempty(dfa{a})
                for b = 1:numBands
                    if ~isempty(dfa{a}{b})
                        plot(windowCenterTimesSeconds, dfa{a}{b}, '--', 'Color', bandColors(b, :), 'LineWidth', 1.5, ...
                            'DisplayName', sprintf('DFA %s', bands{b, 1}));
                    end
                end
            end
            ylabel('DFA \alpha');
            ylim([0.3, 1.7]); % Standard range for DFA alpha
            grid off; % Keep left axis grid
        end
        
        % Add vertical lines at reach onsets (only for reach data)
        % Note: HandleVisibility is set to 'off' to exclude from legend
        if exist('dataType', 'var') && strcmp(dataType, 'reach')
            % Filter reach onsets to only show those within the current plot's time range
            if exist('reachStart', 'var') && ~isempty(reachStart)
                reachOnsetsInRange = reachStart(reachStart >= xMin & reachStart <= xMax);
                
                if ~isempty(reachOnsetsInRange)
                    for i = 1:length(reachOnsetsInRange)
                        h = xline(reachOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                        h.HandleVisibility = 'off';
                    end
                end
            end
            % Add block 2 start line if available
            if exist('startBlock2', 'var') && ~isempty(startBlock2) && startBlock2 >= xMin && startBlock2 <= xMax
                h = xline(startBlock2, 'Color', [1 0 0], 'LineWidth', 3);
                h.HandleVisibility = 'off';
            end
        end
        
        % Add vertical lines at saccade/response onsets (only for schall data)
        % Note: HandleVisibility is set to 'off' to exclude from legend
        if exist('dataType', 'var') && strcmp(dataType, 'schall')
            % Filter response onsets to only show those within the current plot's time range
            if exist('responseOnset', 'var') && ~isempty(responseOnset)
                responseOnsetsInRange = responseOnset(responseOnset >= xMin & responseOnset <= xMax);
                
                if ~isempty(responseOnsetsInRange)
                    for i = 1:length(responseOnsetsInRange)
                        h = xline(responseOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                        h.HandleVisibility = 'off';
                    end
                end
            end
        end
        
        yyaxis left;
        title(sprintf('%s - d2 Analysis', areas{a}));
        xlabel('Time (s)');
        ylabel('d2');
        xlim([xMin, xMax]);
        ylim([yMinD2, yMaxD2]);
        % Ensure tick labels are visible
        set(gca, 'XTickLabelMode', 'auto');
        set(gca, 'YTickLabelMode', 'auto');
        if idx == 1
            legend('Location', 'best');
        end
        grid on;
    end
    
    % Add super title
    if exist('dataType', 'var') && strcmp(dataType, 'reach')
        if ~isempty(filePrefix)
            sgtitle(sprintf('[%s] LFP d2 Analysis with reach onsets (gray dashed) - win=%gs, step=%gs', filePrefix, slidingWindowSize, d2StepSize));
        else
            sgtitle(sprintf('LFP d2 Analysis with reach onsets (gray dashed) - win=%gs, step=%gs', slidingWindowSize, d2StepSize));
        end
    else
        if ~isempty(filePrefix)
            sgtitle(sprintf('[%s] LFP d2 Analysis - win=%gs, step=%gs', filePrefix, slidingWindowSize, d2StepSize));
        else
            sgtitle(sprintf('LFP d2 Analysis - win=%gs, step=%gs', slidingWindowSize, d2StepSize));
        end
    end
    
    % Save figure
    if ~isempty(filePrefix)
        exportgraphics(gcf, fullfile(saveDir, sprintf('%s_criticality_lfp_ar_win%d.png', filePrefix, slidingWindowSize)), 'Resolution', 300);
        fprintf('Saved LFP d2 plots to: %s\n', fullfile(saveDir, sprintf('%s_criticality_lfp_ar_win%d.png', filePrefix, slidingWindowSize)));
    else
        exportgraphics(gcf, fullfile(saveDir, sprintf('criticality_lfp_ar_win%d.png', slidingWindowSize)), 'Resolution', 300);
        fprintf('Saved LFP d2 plots to: %s\n', fullfile(saveDir, sprintf('criticality_lfp_ar_win%d.png', slidingWindowSize)));
    end
end

fprintf('\n=== LFP Power Band d2 Analysis Complete ===\n');

%% ==============================================     Helper Functions     ==============================================

function permutedSignal = phase_randomize_signal(signal)
% PHASE_RANDOMIZE_SIGNAL - Phase randomization of a time series signal
% Preserves the power spectrum (autocorrelation) while randomizing phase
% This breaks critical dynamics while maintaining temporal structure
%
% Method: Takes FFT, randomizes phase while keeping magnitude, then inverse FFT
% For real-valued signals, maintains conjugate symmetry of FFT
%
% Input:
%   signal - 1D time series vector
% Output:
%   permutedSignal - Phase-randomized signal with same power spectrum

    % Ensure signal is a column vector
    originalOrientation = isrow(signal);
    if originalOrientation
        signal = signal(:);
    end
    
    nSamples = length(signal);
    
    % Compute FFT
    fftSignal = fft(signal);
    
    % Get magnitude (power spectrum) - this will be preserved
    magnitude = abs(fftSignal);
    
    % Generate random phases with proper conjugate symmetry for real signals
    % For real signals: X[k] = conj(X[N-k]) for k = 1, ..., N/2-1
    randomPhase = zeros(nSamples, 1);
    
    % DC component (k=0) - can be any phase, but for real signal it's real
    % We'll set it to 0 (or random, but it doesn't matter since magnitude is preserved)
    randomPhase(1) = 0;
    
    % Positive frequencies (k = 1 to floor((N-1)/2))
    nPosFreq = floor((nSamples - 1) / 2);
    if nPosFreq > 0
        randomPhase(2:nPosFreq+1) = 2 * pi * rand(nPosFreq, 1);
    end
    
    % Nyquist frequency (k = N/2) if N is even - must be real (phase = 0 or pi)
    if mod(nSamples, 2) == 0
        randomPhase(nSamples/2 + 1) = pi * randi([0, 1]);  % 0 or pi
    end
    
    % Negative frequencies - conjugate of positive frequencies
    % X[N-k] = conj(X[k]) for k = 1, ..., floor((N-1)/2)
    for k = 1:nPosFreq
        randomPhase(nSamples - k + 1) = -randomPhase(k + 1);
    end
    
    % Reconstruct FFT with original magnitude but random phase
    permutedFFT = magnitude .* exp(1i * randomPhase);
    
    % Inverse FFT to get permuted signal (should be real)
    permutedSignal = real(ifft(permutedFFT));
    
    % Ensure output is same orientation as input
    if originalOrientation
        permutedSignal = permutedSignal';
    end
end

