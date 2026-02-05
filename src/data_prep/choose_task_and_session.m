%%
% =============================    Configuration    =============================
% Data source selection
dataSource = 'spikes';  % 'spikes' or 'lfp'

% Initialize paths
paths = get_paths;

% Initialize options structure
opts = neuro_behavior_options;
opts.frameSize = .001;
opts.firingRateCheckTime = 3 * 60;
opts.collectStart = 0;
opts.minFiringRate = .25;
opts.maxFiringRate = 200;

% LFP-specific options (used when dataSource == 'lfp')
if strcmp(dataSource, 'lfp')
    % Frequency bands for LFP analysis
    bands = {'alpha', [8 13]; ...
        'beta', [13 30]; ...
        'lowGamma', [30 50]; ...
        'highGamma', [50 80]};


    % LFP cleaning parameters
    lfpCleanParams = struct();
    lfpCleanParams.spikeThresh = 4;
    lfpCleanParams.spikeWinSize = 50;
    lfpCleanParams.notchFreqs = [60 120 180];
    lfpCleanParams.lowpassFreq = 300;
    lfpCleanParams.useHampel = true;
    lfpCleanParams.hampelK = 5;
    lfpCleanParams.hampelNsigma = 3;
    lfpCleanParams.detrendOrder = 'linear';
end






%% =============================    Data Loading    =============================
fprintf('\n=== Loading %s %s data ===\n', sessionType, dataSource);

% Create filename suffix based on PCA flag (will be set in analysis script)
% This is a placeholder - actual value will be set in criticality_sliding_window_ar.m

%% =============================    Spontaneous Data Loading    =============================
% Data type selection
sessionType = 'spontaneous';  % 'reach' , 'spontaneous' , 'schall' , 'hong'

sessionName =  'ag112321_1';
% sessionName =  'ag112321_2';
sessionName =  'ey042822';  % Good M56, Bad DS
% sessionName =  'kw092821';  % Bad M56, Good DS
% sessionName =  'kw092121';  % Ok... few M56


%% =============================    Reach Data Loading    =============================
sessionType = 'reach';  % 'reach' , 'spontaneous' , 'schall' , 'hong'

% For reach data: specify session name (uncomment and set one)
sessionName =  'AB2_28-Apr-2023 17_50_02_NeuroBeh';  % GOOD (only 4 m23)
% sessionName =  'AB2_01-May-2023 15_34_59_NeuroBeh';  % GOOD (only 6 m23)
% sessionName =  'AB2_11-May-2023 17_31_00_NeuroBeh';  % OK (21 M56, 11 M23)
% sessionName =  'AB2_30-May-2023 12_49_52_NeuroBeh';  %  GOOD
% sessionName =  'AB6_27-Mar-2025 14_04_12_NeuroBeh';  % GOOD enough (only 4 m23 though)
% sessionName =  'AB6_29-Mar-2025 15_21_05_NeuroBeh';    % BAD (few neurons)
% sessionName =  'AB6_02-Apr-2025 14_18_54_NeuroBeh';   % Few M56 neurons (good DS)
% sessionName =  'AB6_03-Apr-2025 13_34_09_NeuroBeh';  % BAD (few neurons)
sessionName =  'Y4_06-Oct-2023 14_14_53_NeuroBeh';  % GOOD
% sessionName =  'Y15_26-Aug-2025 12_24_22_NeuroBeh';   % BAD (max is m56 with 19 neurons)
% sessionName =  'Y15_27-Aug-2025 14_02_21_NeuroBeh';  % GOOD
sessionName =  'Y15_28-Aug-2025 19_47_07_NeuroBeh';  % GOOD
sessionName =  'Y17_20-Aug-2025 17_34_48_NeuroBeh';  % GOOD (but only 23 M56)
% sessionName =  'test/reach_test';




%% =============================    Schall Choice countermanding Data Loading    =============================
sessionType = 'schall';  % 'reach' , 'spontaneous' , 'schall' , 'hong'

% For schall data: specify session name (uncomment and set one)
sessionName =  'bp229n02-mm';
sessionName =  'bp200n02';
% sessionName =  'bp240n02';
% sessionName =  'jp121n02';
sessionName =  'jp124n04';
sessionName =  'jp125n04';
% sessionName = goodSessionsCCM{end};


%% =============================    Kate Hong whisker/lick Data Loading    =============================
sessionType = 'hong';  % 'reach' , 'spontaneous' , 'schall' , 'hong'
paths = get_paths;

sessionName = 'spikeData';








%% =============================    Call Analysis Script    =============================
% All data is now loaded into the workspace. Call the appropriate analysis script.





%% ===========================     SPIKING DATA SCRIPTS
%%  D2 sliding window
%  Sliding window size (seconds)
binSize = .01;
slidingWindowSize = 2;

criticality_sliding_window_ar
%% Avalanche analyses
% Sliding window and step size (seconds)
slidingWindowSize = 180;  % seconds - user specified
avStepSize = 20;          % seconds - user specified

criticality_sliding_window_av






%% ===========================     LFP SCRIPTS
% Minimum bin size (for highest frequency band)
minBinSize = .005;  % 5 ms

% Compute frequency-dependent bin sizes for each band
% Higher frequencies get smaller bins, lower frequencies get larger bins
% This accounts for the fact that higher frequencies change faster
numBands = size(bands, 1);
bandAvgFreqs = zeros(1, numBands);
for b = 1:numBands
    bandAvgFreqs(b) = mean(bands{b, 2});  % Average of frequency range
end

% Find highest frequency band
[~, highestFreqIdx] = max(bandAvgFreqs);
highestFreqAvg = bandAvgFreqs(highestFreqIdx);

% Calculate bin size for each band: proportional to frequency
% Lower frequency bands get larger bin sizes
bandBinSizes = zeros(1, numBands);
for b = 1:numBands
    % Bin size is proportional to highest freq / current freq
    % Rounded up (ceil) to nearest multiple of minBinSize
    calculatedBinSize = minBinSize * (highestFreqAvg / bandAvgFreqs(b));
    bandBinSizes(b) = ceil(calculatedBinSize / minBinSize) * minBinSize;
end
% bandBinSizes = [.01 .01 .01 .01]
fprintf('Frequency-dependent bin sizes:\n');
for b = 1:numBands
    fprintf('  %s (%.1f Hz avg): %.3f s (%.1f ms)\n', bands{b,1}, bandAvgFreqs(b), bandBinSizes(b), bandBinSizes(b)*1000);
end
fprintf('  Max bin size: %.3f s (%.1f ms)\n', max(bandBinSizes), max(bandBinSizes)*1000);

% Compute binned envelopes for each area (store as cell structure)
% Each band is binned separately with its own bin size
numAreas = size(lfpPerArea, 2);

if strcmp(sessionType, 'schall')
    % Schall: FEF is single area, but may have multiple channels
    if numAreas == 1
        % Single channel - use it directly
        % Bin each band separately with its own bin size
        binnedEnvelopes = {cell(1, numBands)};
        binnedPower = {cell(1, numBands)};
        timePoints = {cell(1, numBands)};
        for b = 1:numBands
            % Call lfp_bin_bandpower for just this band
            % Pass bands(b, :) directly - it's already a 1x2 cell array {name, freqRange}
            singleBand = bands(b, :);  % 1x2 cell array: {bandName, [freqRange]}
            [iBinnedPower, iBinnedEnvelopes, iTimePoints] = lfp_bin_bandpower(lfpPerArea, opts.fsLfp, singleBand, bandBinSizes(b), 'cwt');
            binnedEnvelopes{1}{b} = iBinnedEnvelopes';  % [nFrames_b x 1]
            binnedPower{1}{b} = iBinnedPower';  % [nFrames_b x 1]
            timePoints{1}{b} = iTimePoints(:);  % [nFrames_b x 1]
        end
    else
        % Multiple channels - average them
        lfpMean = mean(lfpPerArea, 2);
        % Bin each band separately with its own bin size
        binnedEnvelopes = {cell(1, numBands)};
        binnedPower = {cell(1, numBands)};
        timePoints = {cell(1, numBands)};
        for b = 1:numBands
            % Pass bands(b, :) directly - it's already a 1x2 cell array {name, freqRange}
            singleBand = bands(b, :);  % 1x2 cell array: {bandName, [freqRange]}
            [iBinnedPower, iBinnedEnvelopes, iTimePoints] = lfp_bin_bandpower(lfpMean, opts.fsLfp, singleBand, bandBinSizes(b), 'cwt');
            binnedEnvelopes{1}{b} = iBinnedEnvelopes';  % [nFrames_b x 1]
            binnedPower{1}{b} = iBinnedPower';  % [nFrames_b x 1]
            timePoints{1}{b} = iTimePoints(:);  % [nFrames_b x 1]
        end
    end
    fprintf('LFP data loaded: %d bands (each with different frame counts)\n', numBands);
else
    % Spontaneous and Reach: multiple areas
    binnedEnvelopes = cell(1, numAreas);
    binnedPower = cell(1, numAreas);
    timePoints = cell(1, numAreas);
    for iArea = 1:numAreas
        binnedEnvelopes{iArea} = cell(1, numBands);
        binnedPower{iArea} = cell(1, numBands);
        timePoints{iArea} = cell(1, numBands);
        % Bin each band separately with its own bin size
        for b = 1:numBands
            % Pass bands(b, :) directly - it's already a 1x2 cell array {name, freqRange}
            singleBand = bands(b, :);  % 1x2 cell array: {bandName, [freqRange]}
            [iBinnedPower, iBinnedEnvelopes, iTimePoints] = lfp_bin_bandpower(lfpPerArea(:, iArea), opts.fsLfp, singleBand, bandBinSizes(b), 'cwt');
            binnedEnvelopes{iArea}{b} = iBinnedEnvelopes';  % [nFrames_b x 1]
            binnedPower{iArea}{b} = iBinnedPower';  % [nFrames_b x 1]
            timePoints{iArea}{b} = iTimePoints(:);  % [nFrames_b x 1]
        end
    end
    fprintf('LFP data loaded: %d areas, %d bands/area (each band with different frame counts)\n', numAreas, numBands);
end

lfpBinSize = 0.005; % 200 Hz
% Store bandBinSizes in workspace for use in analysis script
% This will be used to determine d2StepSize
%%
% D2 sliding window
% Sliding window size (seconds)
% binnedEnvelopes = binnedPower;
slidingWindowSize = 10;     % Window size in seconds
% criticality_sliding_ar_lfp
criticality_sliding_lfp

%% Avalanche analyses
% Sliding window and step size (seconds)











%%   =======================    LZC    ==================
% Check which binSize you want to use
[proportions, spikeCounts] = neural_pct_spike_count(dataMat, [.001 .002 .005 .01 .02], 8);
%%
% Analysis parameters
slidingWindowSize = 60;  % Window size in seconds
stepSize = 45;  % Step size in seconds (for spikes, will be calculated from optimalBinSize)

binSize = .02;
binSize = .002;
complexity_sliding_window