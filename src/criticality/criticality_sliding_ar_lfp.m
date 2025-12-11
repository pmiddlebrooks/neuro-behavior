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

% Analysis flags
analyzeD2 = true;      % compute d2

% Permutation testing flags
enablePermutations = false;  % Set to true to perform circular permutation testing
nShuffles = 3;  % Number of circular permutations to perform

% Analysis parameters
minSegmentLength = 50;
d2StepSize = binSize * 4;  % Step size in seconds (how much to slide the window)

% Optimal bin/window size selection mode
useOptimalBinWindowFunction = false;  % For LFP, we use the frameSize from opts
optimalBinSize = binSize;  % Use the frame size from binnedEnvelopes
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
requiredVars = {'binnedEnvelopes', 'bands', 'opts'};
for i = 1:length(requiredVars)
    if ~exist(requiredVars{i}, 'var')
        error('Required variable %s not found in workspace.', requiredVars{i});
    end
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

% Get number of frames from first area (should be same for all)
numFrames = size(binnedEnvelopes{1}, 1);

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
fprintf('\n=== LFP Power Band d2 Analysis ===\n');
fprintf('Number of bands: %d\n', numBands);
fprintf('Number of areas: %d\n', numAreas);
fprintf('Number of frames: %d\n', numFrames);
fprintf('Frame size: %.3f s\n', binSize);

% Initialize results
% Structure: d2{area}{band} = [1 x numWindows]
d2 = cell(1, numAreas);
d2Permuted = cell(1, numAreas);
startS = cell(1, numAreas);

for a = 1:numAreas
    d2{a} = cell(1, numBands);
    d2Permuted{a} = cell(1, numBands);
    startS{a} = cell(1, numBands);
end

% Process each area
for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});
    
    % Extract this area's binned envelopes (now stored in cell structure)
    if length(binnedEnvelopes) < a || isempty(binnedEnvelopes{a})
        fprintf('Skipping: No binned envelopes data for area %s...\n', areas{a});
        continue
    end
    areaEnvelopes = binnedEnvelopes{a};  % [nFrames x numBands]
    
    % Calculate window parameters
    stepSamples = round(d2StepSize / optimalBinSize);
    winSamples = round(slidingWindowSize / optimalBinSize);
    numWindows = floor((numFrames - winSamples) / stepSamples) + 1;
    
    % Skip this area if there aren't enough samples
    if winSamples < minSegmentLength
        fprintf('Skipping: Not enough data in %s...\n', areas{a});
        continue
    end
    
    % Process each band separately
    for b = 1:numBands
        fprintf('  Processing band %d (%s)...\n', b, bands{b, 1});
        tic;
        
        % Extract this band's time series
        bandSignal = areaEnvelopes(:, b);  % [nFrames x 1]
        
        % Initialize arrays for this band
        d2{a}{b} = nan(1, numWindows);
        startS{a}{b} = nan(1, numWindows);
        
        % Sliding window analysis
        for w = 1:numWindows
            startIdx = (w - 1) * stepSamples + 1;
            endIdx = startIdx + winSamples - 1;
            
            % Ensure we don't exceed data bounds
            if endIdx > numFrames
                endIdx = numFrames;
            end
            
            startS{a}{b}(w) = (startIdx + round(winSamples/2) - 1) * optimalBinSize;
            
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
        end
        
        fprintf('    Band %d completed in %.1f minutes\n', b, toc/60);
        
        % Permutations for this band
        if enablePermutations
            fprintf('    Running %d phase-randomized permutations per window for band %d...\n', nShuffles, b);
            ticPerm = tic;
            
            % Initialize storage for permutation results [nWindows x nShuffles]
            d2Permuted{a}{b} = nan(numWindows, nShuffles);
            
            % Permute each window independently
            for w = 1:numWindows
                startIdx = (w - 1) * stepSamples + 1;
                endIdx = startIdx + winSamples - 1;
                
                % Ensure we don't exceed data bounds
                if endIdx > numFrames
                    endIdx = numFrames;
                end
                
                % Extract this window's data
                windowData = bandSignal(startIdx:endIdx);
                winSamples_window = length(windowData);
                
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
end

% =============================    Save Results    =============================
results = struct();
results.areas = areas;
results.bands = bands;
results.d2 = d2;
results.startS = startS;
results.optimalBinSize = optimalBinSize;
results.optimalWindowSize = optimalWindowSize;
results.d2StepSize = d2StepSize;
results.d2WindowSize = slidingWindowSize;
results.params.slidingWindowSize = slidingWindowSize;
results.params.analyzeD2 = analyzeD2;
results.params.pOrder = pOrder;
results.params.critType = critType;
results.params.binSize = binSize;

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
        
        % Plot d2 for each band on the same axis
        for b = 1:numBands
            if ~isempty(d2{a}{b}) && ~isempty(startS{a}{b})
                plot(startS{a}{b}, d2{a}{b}, '-', 'Color', bandColors(b, :), 'LineWidth', 2, ...
                    'DisplayName', sprintf('%s', bands{b, 1}));
            end
            
            % Plot permutation mean ± std per window if available
            if enablePermutations && exist('d2Permuted', 'var') && ~isempty(d2Permuted{a}{b})
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
                         'DisplayName', sprintf('%s Permuted mean ± std', bands{b, 1}));
                    
                    % Plot mean line with dashed style
                    plot(startS{a}{b}(validIdx), permutedMean(validIdx), '--', ...
                        'Color', bandColors(b, :) * 0.8, 'LineWidth', 1, ...
                        'DisplayName', sprintf('%s Permuted mean', bands{b, 1}));
                end
            end
        end
        
        title(sprintf('%s - d2 for all power bands', areas{a}));
        xlabel('Time (s)');
        ylabel('d2');
        xlim([xMin, xMax]);
        ylim([yMinD2, yMaxD2]);
        % Ensure tick labels are visible
        set(gca, 'XTickLabelMode', 'auto');
        set(gca, 'YTickLabelMode', 'auto');
        legend('Location', 'best');
        grid on;
    end
    
    % Add super title
    if ~isempty(filePrefix)
        sgtitle(sprintf('[%s] LFP Power Band d2 Analysis - win=%gs, step=%gs', filePrefix, slidingWindowSize, d2StepSize));
    else
        sgtitle(sprintf('LFP Power Band d2 Analysis - win=%gs, step=%gs', slidingWindowSize, d2StepSize));
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

