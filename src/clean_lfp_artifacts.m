function lfpClean = clean_lfp_artifacts(lfpSignal, fs, varargin)
% clean_lfp_artifacts: Comprehensive artifact and spike removal for LFP signals
%
% This function implements a multi-stage pipeline to remove spikes, artifacts,
% and noise from LFP signals while preserving the underlying neural oscillations.
%
% Inputs:
%   - lfpSignal: LFP signal (column vector or matrix with signals in columns)
%   - fs: Sampling rate (Hz)
%   - varargin: Optional parameter-value pairs:
%     * 'spikeThresh': Z-score threshold for spike detection (default: 4)
%     * 'spikeWinSize': Window size around spikes for removal in samples (default: 50)
%     * 'notchFreqs': Frequencies to notch filter (default: [60 120 180])
%     * 'lowpassFreq': Lowpass cutoff frequency (default: 300)
%     * 'useHampel': Use Hampel filter for outlier detection (default: true)
%     * 'hampelK': Half-window size for Hampel filter (default: 5)
%     * 'hampelNsigma': Number of standard deviations for Hampel (default: 3)
%     * 'detrendOrder': Order of detrending ('linear', 'constant', or polynomial order) (default: 'linear')
%     * 'visualize': Show visualization of processing steps (default: false)
%     * 'visualizeChannel': Which channel to visualize if multiple (default: 1)
%     * 'visualizeSamples': Sample range to visualize [start end], empty for all (default: [])
%
% Outputs:
%   - lfpClean: Cleaned LFP signal(s) with same dimensions as input
%
% Processing pipeline:
%   1. Detrend to remove slow drifts
%   2. Hampel filter for robust outlier detection and removal
%   3. Z-score based spike detection and removal with spline interpolation
%   4. Notch filtering to remove line noise and harmonics
%   5. Lowpass filtering to remove high-frequency noise
%   6. Final median filter to smooth any remaining artifacts

% Parse optional inputs
p = inputParser;
addParameter(p, 'spikeThresh', 4, @isnumeric);
addParameter(p, 'spikeWinSize', 50, @isnumeric);
addParameter(p, 'notchFreqs', [60 120 180], @isnumeric);
addParameter(p, 'lowpassFreq', 300, @isnumeric);
addParameter(p, 'useHampel', true, @islogical);
addParameter(p, 'hampelK', 5, @isnumeric);
addParameter(p, 'hampelNsigma', 3, @isnumeric);
addParameter(p, 'detrendOrder', 'linear', @(x) ischar(x) || isnumeric(x));
addParameter(p, 'visualize', false, @islogical);
addParameter(p, 'visualizeChannel', 1, @isnumeric);
addParameter(p, 'visualizeSamples', [], @isnumeric);
parse(p, varargin{:});

spikeThresh = p.Results.spikeThresh;
spikeWinSize = p.Results.spikeWinSize;
notchFreqs = p.Results.notchFreqs;
lowpassFreq = p.Results.lowpassFreq;
useHampel = p.Results.useHampel;
hampelK = p.Results.hampelK;
hampelNsigma = p.Results.hampelNsigma;
detrendOrder = p.Results.detrendOrder;
visualize = p.Results.visualize;
visualizeChannel = p.Results.visualizeChannel;
visualizeSamples = p.Results.visualizeSamples;

% Ensure signal is column vector or matrix with signals in columns
if size(lfpSignal, 1) == 1 && size(lfpSignal, 2) > 1
    lfpSignal = lfpSignal';
end
[nSamples, nChannels] = size(lfpSignal);
lfpClean = zeros(size(lfpSignal));

% Initialize visualization storage
if visualize && visualizeChannel <= nChannels
    visChIdx = visualizeChannel;
    visSignal = lfpSignal(:, visChIdx);
    visSteps = struct();
    visSteps.original = visSignal;
    visSteps.spikeIdx = [];
    visSteps.outlierIdx = [];
    visSteps.spikeMask = false(nSamples, 1);
    visSteps.robustZ = [];
    visSteps.afterDetrend = [];
    visSteps.afterHampel = [];
    visSteps.afterSpikeRemoval = [];
    visSteps.afterNotch = [];
    visSteps.afterLowpass = [];
    visSteps.final = [];
end

% Process each channel independently
for chIdx = 1:nChannels
    signal = lfpSignal(:, chIdx);
    signalOriginal = signal; % Store original for visualization
    
    % Step 1: Detrend to remove slow drifts and DC offset
    % This should be done first to prevent artifacts from affecting detrending
    signal = detrend(signal, detrendOrder);
    
    if visualize && chIdx == visualizeChannel
        visSteps.afterDetrend = signal;
    end
    
    % Step 2: Hampel filter for robust outlier detection
    % This is more robust than z-score for detecting outliers in the presence of artifacts
    outlierIdx = [];
    if useHampel
        try
            signalHampel = signal;
            [~, outlierIdx] = hampel(signal, hampelK, hampelNsigma);
            if any(outlierIdx)
                % Replace outliers with NaN for interpolation
                signalHampel(outlierIdx) = NaN;
                % Interpolate using spline for smooth reconstruction
                validIdx = ~isnan(signalHampel);
                if sum(validIdx) > 2
                    signalHampel = interp1(find(validIdx), signalHampel(validIdx), ...
                        1:nSamples, 'spline', 'extrap');
                else
                    signalHampel = signal; % Fallback if too many outliers
                end
                signal = signalHampel;
            end
        catch
            % If hampel is not available, skip this step
            warning('Hampel filter not available, skipping robust outlier detection');
        end
    end
    
    if visualize && chIdx == visualizeChannel
        visSteps.afterHampel = signal;
        visSteps.outlierIdx = outlierIdx;
    end
    
    % Step 3: Z-score based spike detection and removal
    % Use robust statistics (median and MAD) for better spike detection
    signalMed = median(signal);
    signalMAD = mad(signal, 1); % Median absolute deviation
    robustZ = (signal - signalMed) / (1.4826 * signalMAD); % Convert MAD to SD estimate
    
    % Find spikes
    spikeIdx = find(abs(robustZ) > spikeThresh);
    
    % Remove spikes with larger window and better interpolation
    spikeMask = false(nSamples, 1);
    if ~isempty(spikeIdx)
        % Expand spike indices to include surrounding samples
        for i = 1:length(spikeIdx)
            winStart = max(1, spikeIdx(i) - spikeWinSize);
            winEnd = min(nSamples, spikeIdx(i) + spikeWinSize);
            spikeMask(winStart:winEnd) = true;
        end
        
        % Interpolate using spline for smooth reconstruction
        validIdx = ~spikeMask;
        if sum(validIdx) > 2
            signalClean = interp1(find(validIdx), signal(validIdx), ...
                1:nSamples, 'spline', 'extrap');
        else
            signalClean = signal; % Fallback if too many spikes
        end
        signal = signalClean;
    end
    
    if visualize && chIdx == visualizeChannel
        visSteps.afterSpikeRemoval = signal;
        visSteps.spikeIdx = spikeIdx;
        visSteps.spikeMask = spikeMask;
        visSteps.robustZ = robustZ;
    end
    
    % Step 4: Notch filtering to remove line noise and harmonics
    % Apply notch filters in sequence (order matters for stability)
    signalNotched = signal;
    for f0 = notchFreqs
        % Normalize frequency to Nyquist
        wo = f0 / (fs / 2);
        % Bandwidth: narrower for better selectivity
        bw = wo / 50; % Narrower bandwidth than before
        % Design notch filter
        [b, a] = iirnotch(wo, bw);
        % Apply with zero-phase filtering
        signalNotched = filtfilt(b, a, signalNotched);
    end
    signal = signalNotched;
    
    if visualize && chIdx == visualizeChannel
        visSteps.afterNotch = signal;
    end
    
    % Step 5: Lowpass filtering to remove high-frequency noise
    % This should come after notch filtering to avoid aliasing issues
    if lowpassFreq < fs / 2
        signal = lowpass(signal, lowpassFreq, fs);
    end
    
    if visualize && chIdx == visualizeChannel
        visSteps.afterLowpass = signal;
    end
    
    % Step 6: Final median filter to smooth any remaining small artifacts
    % Use a small window to avoid smoothing out real oscillations
    medWinSize = max(3, round(0.001 * fs)); % ~1ms window, minimum 3 samples
    if mod(medWinSize, 2) == 0
        medWinSize = medWinSize + 1; % Ensure odd window size
    end
    signal = medfilt1(signal, medWinSize);
    
    if visualize && chIdx == visualizeChannel
        visSteps.final = signal;
    end
    
    % Store cleaned signal
    lfpClean(:, chIdx) = signal;
end

% Create visualization if requested
if visualize && visualizeChannel <= nChannels
    create_visualization(visSteps, fs, spikeThresh, spikeWinSize, hampelK, hampelNsigma, visualizeChannel, visualizeSamples);
end

end

function create_visualization(visSteps, fs, spikeThresh, spikeWinSize, hampelK, hampelNsigma, channelNum, visualizeSamples)
% create_visualization: Creates comprehensive visualization of artifact removal steps

% Determine sample range to plot
if isempty(visualizeSamples) || length(visualizeSamples) ~= 2
    sampleRange = 1:length(visSteps.original);
else
    sampleRange = max(1, visualizeSamples(1)):min(length(visSteps.original), visualizeSamples(2));
end
timeVec = (sampleRange - 1) / fs; % Time vector in seconds

% Create figure with multiple subplots
fig = figure('Position', [100, 100, 1400, 1000], 'Name', 'LFP Artifact Removal Pipeline');

% Subplot 1: Original signal with detected spikes and thresholds
subplot(4, 2, 1);
plot(timeVec, visSteps.original(sampleRange), 'k-', 'LineWidth', 1.5);
hold on;
if ~isempty(visSteps.spikeIdx)
    spikeInRange = visSteps.spikeIdx(visSteps.spikeIdx >= sampleRange(1) & visSteps.spikeIdx <= sampleRange(end));
    if ~isempty(spikeInRange)
        plot((spikeInRange - 1) / fs, visSteps.original(spikeInRange), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    end
end
if ~isempty(visSteps.outlierIdx)
    outlierInRange = visSteps.outlierIdx(visSteps.outlierIdx >= sampleRange(1) & visSteps.outlierIdx <= sampleRange(end));
    if ~isempty(outlierInRange)
        plot((outlierInRange - 1) / fs, visSteps.original(outlierInRange), 'gx', 'MarkerSize', 10, 'LineWidth', 2);
    end
end
title(sprintf('Original Signal (Red=Spikes, Green=Hampel Outliers)\nChannel %d', channelNum));
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
legend('Signal', 'Spikes', 'Hampel Outliers', 'Location', 'best');

% Subplot 2: Robust Z-score with threshold lines
subplot(4, 2, 2);
if ~isempty(visSteps.robustZ)
    plot(timeVec, visSteps.robustZ(sampleRange), 'b-', 'LineWidth', 1.5);
    hold on;
    yline(spikeThresh, 'r--', sprintf('Threshold = %.1f', spikeThresh), 'LineWidth', 2);
    yline(-spikeThresh, 'r--', 'LineWidth', 2);
    if ~isempty(visSteps.spikeIdx)
        spikeInRange = visSteps.spikeIdx(visSteps.spikeIdx >= sampleRange(1) & visSteps.spikeIdx <= sampleRange(end));
        if ~isempty(spikeInRange)
            plot((spikeInRange - 1) / fs, visSteps.robustZ(spikeInRange), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
        end
    end
    title(sprintf('Robust Z-Score (MAD-based)\nSpike Threshold = %.1f SD', spikeThresh));
    xlabel('Time (s)');
    ylabel('Z-Score');
    grid on;
    legend('Z-Score', 'Threshold', '', 'Detected Spikes', 'Location', 'best');
end

% Subplot 3: Spike windows visualization
subplot(4, 2, 3);
plot(timeVec, visSteps.afterDetrend(sampleRange), 'k-', 'LineWidth', 1.5);
hold on;
if any(visSteps.spikeMask(sampleRange))
    spikeWinInRange = visSteps.spikeMask(sampleRange);
    plot(timeVec(spikeWinInRange), visSteps.afterDetrend(sampleRange(spikeWinInRange)), ...
        'r.', 'MarkerSize', 15, 'LineWidth', 1);
end
if ~isempty(visSteps.spikeIdx)
    spikeInRange = visSteps.spikeIdx(visSteps.spikeIdx >= sampleRange(1) & visSteps.spikeIdx <= sampleRange(end));
    if ~isempty(spikeInRange)
        for i = 1:length(spikeInRange)
            winStart = max(sampleRange(1), spikeInRange(i) - spikeWinSize);
            winEnd = min(sampleRange(end), spikeInRange(i) + spikeWinSize);
            winTime = (winStart:winEnd) / fs;
            plot(winTime, visSteps.afterDetrend(winStart:winEnd), 'r:', 'LineWidth', 1);
        end
    end
end
title(sprintf('Spike Windows (Window Size = %d samples = %.2f ms)', ...
    spikeWinSize, spikeWinSize / fs * 1000));
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
legend('Signal', 'Spike Windows', 'Location', 'best');

% Subplot 4: After detrending
subplot(4, 2, 4);
plot(timeVec, visSteps.afterDetrend(sampleRange), 'b-', 'LineWidth', 1.5);
title('After Detrending');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Subplot 5: After Hampel filter
subplot(4, 2, 5);
plot(timeVec, visSteps.afterHampel(sampleRange), 'b-', 'LineWidth', 1.5);
title(sprintf('After Hampel Filter (k=%d, nσ=%d)', hampelK, hampelNsigma));
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Subplot 6: After spike removal
subplot(4, 2, 6);
plot(timeVec, visSteps.afterSpikeRemoval(sampleRange), 'b-', 'LineWidth', 1.5);
title('After Spike Removal (Spline Interpolation)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Subplot 7: After notch filtering
subplot(4, 2, 7);
plot(timeVec, visSteps.afterNotch(sampleRange), 'b-', 'LineWidth', 1.5);
title('After Notch Filtering');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Subplot 8: Final cleaned signal
subplot(4, 2, 8);
plot(timeVec, visSteps.original(sampleRange), 'k--', 'LineWidth', 1, 'DisplayName', 'Original');
hold on;
plot(timeVec, visSteps.final(sampleRange), 'b-', 'LineWidth', 2, 'DisplayName', 'Cleaned');
title('Final Comparison');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
legend('Location', 'best');

% Add summary text
sgtitle(sprintf('LFP Artifact Removal Pipeline - Channel %d\nSpike Threshold: %.1f SD | Window Size: %d samples (%.2f ms) | Hampel: k=%d, nσ=%d', ...
    channelNum, spikeThresh, spikeWinSize, spikeWinSize/fs*1000, hampelK, hampelNsigma), ...
    'FontSize', 12, 'FontWeight', 'bold');

end

