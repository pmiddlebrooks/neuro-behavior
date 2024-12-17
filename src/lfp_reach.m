
%% Load the LFP and behavior data
reachPath = 'E:\Projects\neuro-behavior\data\reach_data';

rData = load(fullfile(reachPath, 'AB2_042823_ReachTimes_info'));

lfpFile = fullfile(reachPath, 'AB2_042823_g0_t0.imec0.lf.bin');
metaFile = fullfile(reachPath, 'AB2_042823_g0_t0.imec0.lf.meta');
addpath('E:\Projects\toolboxes')
m = ReadMeta(metaFile);

fs = round(str2double(m.imSampRate));
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
    

%% Detrend, normalize, and check stationarity of LFP signals

% Detrend the signals (remove linear trend)
detrendedLfp = detrend(lfpReach, 'linear');

% Normalize the signals (z-score normalization)
normalizedLfp = zscore(detrendedLfp);
clear detrendedLfp

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

%% MVGC1 toolbox: peri-reach from Mark's data: Get his data from lfp_reach.m
fullTime = -.2 : 1/fs : .2; % seconds around onset
fullWindow = round(fullTime(1:end-1) * fs); % frames around onset w.r.t. zWindow (remove last frame)

nTrial = size(rData.R, 1);
lfpTest = [];
for i = 1 : nTrial
% check for noise (check if it's a usable trial)
iLfp = normalizedLfp(rData.R(i,1) + fullWindow, :)';
if ~any(iLfp(2,:) > 5)
    lfpTest = cat(3, lfpTest, iLfp);
end
end

X = lfpTest;

cd E:\Projects\toolboxes\MVGC1
startup

% Go to mvgc_demo_script.m
disp('Go to mvgc_demo_script.m')










%%
figure(55);
fullTime = -.2 : 1/fs : .2; % seconds around onset
fullWindow = round(fullTime(1:end-1) * fs); % frames around onset w.r.t. zWindow (remove last frame)

nTrial = size(rData.R, 1);
lfpTest = zeros(size(lfpReach, 3), length(fullWindow), nTrial);
for i = 1 : nTrial
    plot(fullTime(1:end-1), lfpReach(rData.R(i,1) + fullWindow, 2));
    disp(i)
    disp(rData.Block(i,:))
end

%%
samples = 10000:100000;
lfpSample = double(mmf.Data.x(170,samples));
figure(9); plot(zscore(lfpSample))

%%
writematrix(lfpReach(:,2), ['E:/Projects/neuro-behavior/data/', 'lfpSample.csv'])
% builtin('writematrix', 'lfpSample', [opts.dataPath, 'lfpSample.csv'])

%% What are those artifacts?
% [cfs, frequencies] = cwt(lfpSample, 'amor', fs, 'FrequencyLimits', freqLimits);
[cfs, frequencies] = cwt(denoisedSignal, 'amor', fs);
pws  = abs(cfs).^2;
figure(5);
imagesc(1:size(pws, 2), frequencies, pws);
% axis xy; % Flip the y-axis so lower frequencies are at the bottom

% imagesc(flipud(pws))

%% from matlab: https://www.mathworks.com/help/signal/ug/remove-the-60-hz-hum-from-a-signal.html

d = designfilt('bandstopiir','FilterOrder',2, ...
               'HalfPowerFrequency1',59,'HalfPowerFrequency2',61, ...
               'DesignMethod','butter','SampleRate',fs);
t = (1:length(lfpSample)) / fs;
buttLoop = filtfilt(d,lfpSample);

figure(8)
plot(t,lfpSample,t,buttLoop)
ylabel('Voltage (V)')
xlabel('Time (s)')
title('Open-Loop Voltage')
legend('Unfiltered','Filtered')
%%
% Notch filter example for 60 Hz
wo = 650 / (fs / 2);  % Normalized frequency
bw = wo / 35;                 % Bandwidth
[b, a] = iirnotch(wo, bw);
denoisedSignal = filtfilt(b, a, lfpSample);




%%
% lowpass the LFP at 300hz
lfpSample = lowpass(lfpSample, 300, fs);
figure(19); plot(zscore(lfpSample))


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
zScoredSignal = zscore(lfpSample);

% Step 2: Detect spikes
spikeIndices = find(abs(zScoredSignal) > spikeThreshold);

% Step 3: Remove spikes by interpolation
cleanedSignal = lfpSample;
for idx = 1:length(spikeIndices)
    spikeIdx = spikeIndices(idx);

    % Define interpolation window
    startIdx = max(1, spikeIdx - windowSize);
    endIdx = min(length(lfpSample), spikeIdx + windowSize);

    % Interpolate the signal
    interpRange = [startIdx:endIdx];
    if length(interpRange) > 2
        cleanedSignal(interpRange) = interp1([startIdx, endIdx], lfpSample([startIdx, endIdx]), interpRange);
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



function [Pxx, F] = myTimePowerSpectrumMat(x, Fs)
L = size(x,1);
NFFT = 2^nextpow2(L);
[Pxx,F] = pwelch(x,[],[],NFFT,Fs);
end