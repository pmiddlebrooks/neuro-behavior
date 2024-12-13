
%%
reachPath = 'E:\Projects\neuro-behavior\data\reach_data';

rData = load(fullfile(reachPath, 'AB2_042823_ReachTimes_info'));

lfpFile = fullfile(reachPath, 'AB2_042823_g0_t0.imec0.lf.bin');
metaFile = fullfile(reachPath, 'AB2_042823_g0_t0.imec0.lf.meta');
m = ReadMeta(metaFile);

fs = str2double(m.imSampRate);
nChan = str2double(m.nSavedChans);
nSamps = str2double(m.fileSizeBytes)/2/nChan;

mmf = memmapfile(lfpFile, 'Format', {'int16', [nChan nSamps], 'x'});
% mmf = memmapfile(lfpFile, 'Format', 'int16');
% mmf = memmapfile(lfpFile);
lfpReach = double(flipud([...
    mmf.Data.x(360,:);...
    mmf.Data.x(280,:);...
    mmf.Data.x(170,:);...
    mmf.Data.x(40,:)...
    ])');

%%
samples = 10000:20000;
lfpSample = double(mmf.Data.x(170,samples));
figure(9); plot(lfpSample)

% lowpass the LFP at 300hz
lfpSample = lowpass(lfpSample, 1, fs);
figure(19); plot(lfpSample)


%     % Remove spikes
% lfpReach = medfilt1(lfpReach, 3, [], 2);

% See Mark's pic for depth...
    % m23 = [3300 3800];
    % m56 = [2300 3200];
    % cc = [2100 2200];
    % ds = [1100 2000];
    % vs = [0 1000];







    %% Script to remove spikes from raw LFP signal
% Parameters for spike removal
spikeThreshold = 5; % Threshold in standard deviations to identify spikes
windowSize = 1000; % Window size around the spike to interpolate (in samples)

% Step 1: Z-score the signal for spike detection
zScoredSignal = zscore(lfpReach(:,2));

% Step 2: Detect spikes
spikeIndices = find(abs(zScoredSignal) > spikeThreshold);

% Step 3: Remove spikes by interpolation
cleanedSignal = lfpReach;
for idx = 1:length(spikeIndices)
    spikeIdx = spikeIndices(idx);

    % Define interpolation window
    startIdx = max(1, spikeIdx - windowSize);
    endIdx = min(length(lfpReach), spikeIdx + windowSize);

    % Interpolate the signal
    interpRange = [startIdx:endIdx];
    if length(interpRange) > 2
        cleanedSignal(interpRange) = interp1([startIdx, endIdx], lfpReach([startIdx, endIdx]), interpRange);
    end
end

%% Plot the original and cleaned signals
figure(8);
subplot(2, 1, 1);
plot(lfpReach(:,2), 'k');
hold on;
plot(spikeIndices, lfpReach(spikeIndices,2), 'ro');
title('Original LFP Signal with Detected Spikes');
xlabel('Time (samples)');
ylabel('Amplitude');
legend('LFP Signal', 'Spikes');

subplot(2, 1, 2);
plot(cleanedSignal(:,2), 'b');
title('Cleaned LFP Signal');
xlabel('Time (samples)');
ylabel('Amplitude');
legend('Cleaned LFP');




%%
% lowpass the LFP at 300hz
lfpReach = lowpass(lfpReach, 300, fs);


    %%
    samples = floor(size(lfpReach, 1)/10);

    %%


    % Detrend the signals (remove linear trend)
detrendedLfp = detrend(lfpReach(1:samples, lfpIdx), 'linear');

% Normalize the signals (z-score normalization)
normalizedLfp = zscore(detrendedLfp);

% Stationarity check using Augmented Dickey-Fuller (ADF) test
% Ensure you have the Econometrics Toolbox for the adftest function
[numSamples, numAreas] = size(normalizedLfp);
stationarityResults = zeros(1, numAreas); % Store p-values

for area = 1:numAreas
    [h, pValue] = adftest(normalizedLfp(:, area)); % ADF test
    stationarityResults(area) = pValue; % Save p-value

    % Display result
    if h == 1
        fprintf('Signal %d is stationary (p = %.4f).\n', area, pValue);
    else
        fprintf('Signal %d is not stationary (p = %.4f).', area, pValue);
    end
end

%%
% Plot the signals for visualization
figure(91); clf;
for area = 1:numAreas
    subplot(numAreas, 1, area);
    plot(normalizedLfp(1:samples, area));
    title(sprintf('Normalized LFP Signal (Area %d)', area));
    xlabel('Time');
    ylabel('Amplitude');
end
sgtitle('Detrended and Normalized LFP Signals');
