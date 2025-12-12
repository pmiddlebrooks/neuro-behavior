

%% Linear encoding model (Musall et al 2019; Runyan et al 2017




%% Load behavioral data




%% Load data

% From Eric (see gmail)
%
% - clu is 2 columns.  the first is spike time in seconds, the second is neuron ID
% - bStart100 is a 3 column matrix of start time in seconds, behavioral ID, and duration (in frames, each of which is 1/60 of a second)
% - useDeep is an index I made that has neuron ID  - same as clu(:,2) - with a reasonable baseline firing rate (I'm guessing 0.1hz to 50hz?) in the first column and depth relative to cortical surface in the second column

% Behavior labels from Eric
%
names = {'wall-rear1', 'unsure', 'unsure', 'general-groom', 'rear', ...
    'body-groom1', 'body-groom2', 'body-groom3', 'investigate1','wall-rear2', ...
    'face-groom', 'head-groom1', 'head-groom2', 'body-groom4', 'body-groom5',...
    'investigate2', 'inactive', 'investigate3', 'orient-ipsi','left-hindpaw-itch-lick',...
    'right-hindpaw-itch', 'right-hindpaw-lick', 'orient-contra', 'locomotion'};

dataPath = 'E:/Projects/ridgeRegress/data/neuralCorrHsu';
data = load(dataPath);

bhvStart = data.bStart100(:,1);
bhvID = data.bStart100(:,2);
bhvDur = data.bStart100(:,3);

% Classified behaviors a la Hsu & Yttri 20
% actionData =

% spikeData =

% lfpData =

% Spatial behavioral data (location in box, orientation, etc)

%%
neuronID = 100;
jSpikeTimes = data.clu(data.clu(:,2) == neuronID, 1);
baselineRate = length(jSpikeTimes) / jSpikeTimes(end);

%%
opts = neuro_behavior_options;
actID = 1;
acts = data.bStart100(data.bStart100(:,2) == actID & data.bStart100(:,3) >= opts.minActTime, 1);

j = 1;

iStartTime = acts(j) - opts.mPreTime/opts.frameSize;

% spikes in first bin:
for mFrame = 1 : opts.nFramePerTrial
    mFrameStartTime = iStartTime + (mFrame - 1) * opts.frameSize;
    mFrameEndTime = mFrameStartTime + opts.frameSize;
    kSpikes = sum(jSpikeTimes(jSpikeTimes >= mFrameStartTime & jSpikeTimes < mFrameEndTime));

end
% iSpikes = sum(spikes(spikes >= acts(i) - opts.mPreTime/opts.frameSize & spikes < opts.mPreTime/opts.frameSize + opts.frameSize))
% iFrameTime =
% Make a matrix with ones where events occurred
%% Design matrix

% Model Features
%   Actions
%   Last Action
%   LFP powers as a proxy of animal state (alpha, beta, gamma)
%   Physical location in box
%   Distance from nest
%   Orientation w.r.t. nest

%   Actions: 16 acts
actList = {'investigate 1', 'investigate 2', 'investigate 3', 'rear', ...
    'dive', 'paw groom', 'face groom 1', 'face groom 2', ...
    'head groom', 'contra-body groom', 'ipsi-body groom', 'contra-itch', ...
    'ipsi-itch', 'contra-orient', 'ipsi-orient', 'locomotion'};
actType = ones(length(actList), 1);


%   Last Action


%   LFP powers as a proxy of animal state (alpha, beta, gamma)


%   Physcial location in box

%   Distance from nest

%   Orientation w.r.t. nest

%% Make some synthetic data to test the valid behavior code below:
bhvID = repmat([1;2], 15, 1);
bhvDur = [repmat([1;6], 3, 1);  repmat([6;1], 5, 1); 61*ones(14,1)];
bhvStart = (cumsum(bhvDur)-1) /60;

[bhvStart, bhvID, bhvDur]







%% Convert Alex's smoothed behavioral data to format suitable for code below
fs = 1/60; % frame rate
dataPath = 'E:/Projects/ridgeRegress/data/txtfmt_data/';

dataBhv = load([dataPath,'ag25290_day2_iter13.txt']);


% ag = [6 6 6 2 2 4 4 4 4 6 6 2]';
changeBehv = [0; diff(dataBhv)]; % nonzeros at all the indices when a new behavior begins
changeBehvIdx = find(changeBehv ~= 0);

bhvDur = [changeBehvIdx(1) - 1; diff(changeBehvIdx); length(dataBhv) - changeBehvIdx(end) + 1];
bhvID = [dataBhv(1); dataBhv(changeBehvIdx)];
bhvStart = [0; cumsum(bhvDur(1:end-1)) * fs];

[bhvStart, bhvID, bhvDur]











%% Distribution of 1-back 1-forward behavioral sequences
close all

behaviors = {'unlabeled', 'torso groom', 'investigate type 0', 'unsure', 'wall rear type 1', 'rear', 'unsure', 'wall rear type 2', 'investigate type 1', 'investigate type 2', 'contra-itch', 'investigate type 3', 'sleep/scrunch type 1', 'sleep/scrunch type 2', 'wall rear type 3', 'contra-body groom', 'face groom type 1', 'dive/scrunch', 'head groom', 'ipsi-orient', 'ipsi-investigate', 'face groom type 2', 'ipsi-body groom', 'ipsi-itch type 1', 'ipsi-itch type 2', 'face groom type 3', 'paw groom', 'locomotion', 'contra-forepaw orient', 'contra-orient'};

actList = unique(bhvID)'; % [19:24];
actList(1) = []; % Get rid of -1 "unclassified" behaviors


% Input vector of integers (modify as needed)
input_vector = bhvID;

% Initialize a cell array to store results
unique_integers = unique(input_vector);
preceding_distributions = cell(length(unique_integers), 1);
following_distributions = cell(length(unique_integers), 1);

% Iterate through the input vector
for j = 2:length(input_vector) - 1
    current_integer = input_vector(j);
    preceding_integer = input_vector(j - 1);
    following_integer = input_vector(j + 1);

    % Check if the current integer is unique
    if ismember(current_integer, unique_integers)
        % Find the index of the current unique integer
        index = find(unique_integers == current_integer);

        % Append the preceding integer to the corresponding cell in the cell array
        if isempty(preceding_distributions{index})
            preceding_distributions{index} = preceding_integer;
        else
            preceding_distributions{index} = [preceding_distributions{index}, preceding_integer];
        end

        % Append the following integer to the corresponding cell in the cell array
        if isempty(following_distributions{index})
            following_distributions{index} = following_integer;
        else
            following_distributions{index} = [following_distributions{index}, following_integer];
        end
    end
end

% Calculate the number of rows and columns for the subplot layout
num_rows = ceil(length(unique_integers) / 3);
num_columns = min(3, length(unique_integers));

% Get the screen size
screen_size = get(0, 'ScreenSize');

% Create a figure with subplots for each unique integer's distribution
figure('Position', screen_size); % Make the figure as large as the current display
subplot_index = 1;

% Keep track of the counts for each behavior
hBefore = cell(length(unique_integers), 1);
hAfter = cell(length(unique_integers), 1);
hCountsBefore = cell(length(unique_integers), 1);
hCountsAfter = cell(length(unique_integers), 1);


for j = 1:length(unique_integers)
    bhvLabel = behaviors{j};
    % prevLabel = actListmax(preceding);

    unique_integer = unique_integers(j);
    preceding = preceding_distributions{j};

    subplot(num_rows, num_columns, j);
    hBefore{j} = histogram(preceding, 'BinMethod', 'integers', 'Normalization', 'count');
    hCountsBefore{j} = histcounts(preceding, min(unique_integers) - .5 : 1 : max(unique_integers) + .5);
    ylabel('Count');
    title(['Preceding ', bhvLabel, '... # ', num2str(unique_integer)]);
    xlim([-2 30])
    xticks(-1 : 29)


    % Set the same limits for all subplots
    % Add x-axis label only to the subplots in the bottom row
    if j < num_columns * num_rows - num_columns
        set(gca,'XTickLabel',[])
    end
    % Add x-axis label only to the subplots in the bottom row
    if mod(j, num_rows) == 1
        xlabel('Preceding Behaviors');
    end

    % Set the same limits for all subplots
    % axis tight;

end
% Adjust subplot layout
sgtitle('Counts of Behaviors Preceding Each Given Behavior');



% Create a second figure with subplots for each unique integer's following distribution
figure('Position', screen_size); % Make the figure as large as the current display

for j = 1:length(unique_integers)

    bhvLabel = behaviors{j};

    unique_integer = unique_integers(j);
    following = following_distributions{j};

    subplot(num_rows, num_columns, j);
    hAfter{j} = histogram(following, 'BinMethod', 'integers', 'Normalization', 'count');
    hCountsAfter{j} = histcounts(following, min(unique_integers) - .5 : 1 : max(unique_integers) + .5);
    ylabel('Count');
    title(['Following ', bhvLabel, '... # ', num2str(unique_integer)]);
    xlim([-2 30])
    xticks(-1 : 29)


    % Set the same limits for all subplots
    % Add x-axis label only to the subplots in the bottom row
    if j < num_columns * num_rows - num_columns
        set(gca,'XTickLabel',[])
    end
    % Add x-axis label only to the subplots in the bottom row
    if mod(j, num_rows) == 1
        xlabel('Following Behaviors');
    end

end



% Adjust subplot layout
sgtitle('Counts of Behaviors Following Each Behavior');







%% Get the index of a particular behavior
behavior = 'contra-itch';
actlist = find(ismember(behaviors, behavior)) - 1;



%%
% [behavStart(actAndLong), behavID(actAndLong), behavDur(actAndLong)]

startTimes = bhvStart(actAndLong);



















%% Neurons
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Find the neuron clusters (ids) in each brain region

% 0 - 500  motor l2/3
% 500 - 1240 l5/6
% 1240 - 1540 corpus callosum, where little neural activity expected
% 1540 - 2700 dorsal striatum
% 2700 - 3840 ventral striatum

m23 = [0 499];
m56 = [500 1239];
cc = [1240 1539];
ds = [1540 2699];
vs = [2700 3839];

allGood = strcmp(ci.group, 'good') & strcmp(ci.KSLabel, 'good');
allM23 = ci.depth >= m23(1) & ci.depth <= m23(2);
allM56 = ci.depth >= m56(1) & ci.depth <= m56(2);
allCC = ci.depth >= cc(1) & ci.depth <= cc(2);
allDS = ci.depth >= ds(1) & ci.depth <= ds(2);
allVS = ci.depth >= vs(1) & ci.depth <= vs(2);


sum(allGood & (allM23 | allM56 | allDS | allVS | allCC))
sum(allGood & (allM23 | allM56 | allDS | allVS))


goodM23 = allGood & allM23;


%% Unit test
% 
% spikeTimes = [100 120 300 .2*spikeFs .81*spikeFs]' ./ spikeFs;
% spikeClusters = repmat(2, length(spikeTimes), 1);
% 
% durFrames = ceil(max(spikeTimes) / .1);
% dataMat = zeros(durFrames, 1);
% useNeurons = 2;


%% Remove time frames that aren't in the design matrix
dataMat = dataMat(frameIdx,:);

imagesc(dataMat')












%% Build and save aligned neural data for Y
actList = unique(data.bStart100(:,2))'; % [19:24];
neuronList = [data.useDeep(:,1)];
% neuronList = [120 200];
minNoRepeat = 1; % A behavior can't accepted for analysis if it is being repeated within one second

% Figure out how big the data matrix will be
actionIndex = [];
for iAct = actList % length(actList)

    % How many instances of this behavior are valid throughout the
    % recording? (number of times animal is performing a behavior, and it
    % lasts more than opts.minActTime, and

    actIdx = data.bStart100(:,2) == iAct; % All instances labeled as this behavior
    lastsLongEnough = data.bStart100(:,3) >= opts.minActTime; % Only use if it lasted long enough to count

    possibleActIdx = find(actIdx & lastsLongEnough);
    % Remove instances for which the behavior has occured within
    % minNoRepeat seconds before.
    for iPossible = 1 : length(possibleActIdx)
        iTimes = data.bStart100(:,1) < data.bStart100(possibleActIdx(iPossible),1) & data.bStart100(:,1) >= data.bStart100(possibleActIdx(iPossible),1) - minNoRepeat;
        if sum(data.bStart100(iTimes, 2))
            % flag it rid of that one
        end
    end

    % indices of actIDs within the last second
    notInLastSec
    nAct = sum(actIdx & lastsLongEnough & notInLastSec);

    actionIndex = [actionIndex; iAct * ones(nAct * opts.nFramePerTrial, 1 )];
end

neuralData = zeros(length(actionIndex), length(neuronList));

for jNeuron = 1 : length(neuronList)

    % All spike times for this neuron
    jSpikeTimes = data.clu(data.clu(:,2) == neuronList(jNeuron), 1);

    jAct = [];
    for iAct = actList

        % Get the start times of this action throughout the recording
        iActStartTimes = data.bStart100(data.bStart100(:,2) == iAct & data.bStart100(:,3) >= opts.minActTime, 1);
        iActSpikes = zeros(length(iActStartTimes) * opts.nFramePerTrial, 1);
        % get spike times within a 2 sec window around the action start
        % time
        % jSpikeTimes =

        % For each instance (trial) of the action, get spike counts in frame time
        % surrounding the action start.
        for kAct = 1 : length(iActStartTimes)

            kStartTime = iActStartTimes(kAct) - opts.mPreTime/opts.frameSize * opts.frameSize;  % Animal begins action at this time
            kSpikes = zeros(opts.nFramePerTrial, 1);

            % spikes in each frame of this trial:
            for mFrame = 1 : opts.nFramePerTrial
                mFrameStartTime = kStartTime + (mFrame - 1) * opts.frameSize;
                mFrameEndTime = mFrameStartTime + opts.frameSize;
                mSpikes = sum(jSpikeTimes >= mFrameStartTime & jSpikeTimes < mFrameEndTime);
                if mSpikes > 0
                    % mSpikes
                    kSpikes(mFrame) = mSpikes;
                end

            end
            % frame start index for this instance of this action
            frameStartIndex = (kAct - 1) * opts.nFramePerTrial + 1;
            iActSpikes(frameStartIndex : frameStartIndex + opts.nFramePerTrial - 1) = kSpikes;

            % if sum(kSpikes) > 6
            %     fprintf("neuron: %d\n", jNeuron)
            %     fprintf("action: %d\n", actList(iAct))
            % end

        end % kAct

        neuralData(iAct == actionIndex, jNeuron) = iActSpikes;

    end % iAct
    % jAct = [jAct; iActSpikes];
    % neuralData(:, jNeuron) =
end % jNeuron
%%
save('E:/Projects/ridgeRegress/data/neuralDataFrames', "neuralData", "-mat")

%%
for jNeuron = 1 : length(neuronList)
    for iAct = actList
        % neuronIdx = 10;
        % actIdx = actList(10);
        neuronIdx = jNeuron;
        actIdx = iAct;

        neuralTrials = neuralData(actionIndex == actIdx, neuronIdx); % all the trials of a particular behavior
        neuralTrials = reshape(neuralTrials, opts.nFramePerTrial, length(neuralTrials)/opts.nFramePerTrial)';
        neuralRate = mean(neuralTrials * 10, 1);

        plot(neuralRate)
    end
end

%% Options for the model
optsModel = neuro_behavior_options;
% opts.actDuration = 1000; % how many ms per act
% opts.timeStepsPerAct = actDuration / timeStepWindow; % how many time steps in an act (i.e. time steps per "trial")


actKernel = eye(ceil(optsModel.windowSize / optsModel.binSize));

% Neural history


%% Options for the data


% Use 30 min of data
dataAct = dataAct(firstIndex + opts.dataDuration);

%% Build design matrix for regression
[fullMat, eventIdx] = design_matrix(events, eventType, optsModel);
















%% quick and dirty fft
channel = 28;

fs = 1250;
x = lfp(:,channel);

y = fft(x);

% plot a segment of the frequency spectrum
xa = fs/length(x)*(0:length(x)-1);
xa = xa(1:125000);
ya = y(1:125000);
plot(xa, abs(ya))



%% LFP analysis load data
dataPath = 'E:/Projects/ridgeRegress/data/txtfmt_data/';
load([dataPath, 'lfp.txt'])



%% LFP stft

fs = 1250;

channel = 20;
x = lfp(:,channel);

% x1 = x(1:fs); % one second of data
% [s,f,t] = stft(x1, fs);

% s = abs(s);

% imagesc(t, f, s)



%% Collchannel = 20;
channel = 2;
channels = 1:39;
fs = 1250;
% x = lfp(:,channel);

collectStart = 35 * 60; % start collecting data this much time into the recording (seconds)
% offsetTime = startTimes(300); % start collecting data this much time into the recording (seconds)
collectFor = 5 * 60; % collect this much data to analyze (seconds)
x30 = lfp(1 + fs * collectStart : fs * (collectStart + collectFor), channel);

addpath('E:\Projects\schalllab\figure_setup\')


%% bandpower test
bandName = {'alpha', 'beta', 'gamma'};
frequency_bands = [
    8 13;  % Alpha band
    13 30; % Beta band
    30 80  % Gamma band
    ];

for window = [.1 1 60 60*4]
    for j = 1 : length(bandName)
        p = bandpower(x30(1 : fs * window), fs, frequency_bands(j,:));
        pwrDb = 10*log10(p);
        fprintf('\n\nWindow %s\n', num2str(window))
        fprintf('%s:\t%.2f\n', bandName{j}, pwrDb)
    end
end
fprintf('\n')

%%
window = 0 : .1 : 1;
for j = 1 : length(window)
    alphaP(j) = 10*log10(bandpower(x30(fs * window(j) + 1 : fs * (window(j) + .1)), fs, [8 13]));
    betaP(j) = 10*log10(bandpower(x30(fs * window(j) + 1 : fs * (window(j) + .1)), fs, [13 30]));
    gammaP(j) = 10*log10(bandpower(x30(fs * window(j) + 1 : fs * (window(j) + .1)), fs, [30 80]));
    % pwrDb = 10*log10(p);
    % fprintf('\n\nWindow %s\n', num2str(window))
    % fprintf('%s:\t%.2f\n', bandName{i}, pwrDb)
end
figure
hold on
plot(window, alphaP)
plot(window, betaP)
plot(window, gammaP)

alphaP
mean(alphaP)
betaP
mean(betaP)
gammaP
mean(gammaP)
%%
window = 1;
for j = 1 : length(window)
    alphaP = 10*log10(bandpower(x30(1 : fs * window), fs, [8 13]))
    betaP = 10*log10(bandpower(x30(1 : fs * window), fs, [13 30]))
    gammaP = 10*log10(bandpower(x30(1 : fs * window), fs, [30 80]))
    % pwrDb = 10*log10(p);
    % fprintf('\n\nWindow %s\n', num2str(window))
    % fprintf('%s:\t%.2f\n', bandName{i}, pwrDb)
end
%% LFP test
close all

signal = x30;


window_length = .2 * fs;
noverlap = floor(window_length/2);
fft_length = 256;
fft_length = 500;
fft_length = [];

num_frames = floor((length(signal) - noverlap) / (window_length - noverlap));


[S, f_stft] = spectrogram(signal, window_length, noverlap, fft_length, fs);
% [S, f_stft] = spectrogram(current_window, rectwin(window_length), 0, 125, fs);

power_spectrogram = abs(S).^2;


% Define the user-specified frequency band ranges as [f_min, f_max] pairs in Hz
user_frequency_bands = [
    8 13;  % Alpha band
    13 30; % Beta band
    30 80  % Gamma band
    ];



% Plot only power in 0-100 Hz range
plotIdx = f_stft > 0 & f_stft <= 1000;
imagesc(1 : length(signal) / fs, f_stft(plotIdx), 10*log10(power_spectrogram(plotIdx,:))); % Convert to dB
axis xy
colorbar;

figure;
hold on
for j = 1 : 3
    freqIdx = find(f_stft >= user_frequency_bands(j,1) & f_stft <= user_frequency_bands(j,2));
    meanFreq = 10*log10(mean(power_spectrogram(freqIdx, :), 1));
    plot((1:num_frames) * (window_length - noverlap) / fs, meanFreq, 'DisplayName', num2str(user_frequency_bands(j,:)))
end
legend('Location', 'northwest');

%%
frex = logspace(log10(10),log10(fs/12),20);


%% LFP stft
figureHandle = 42;

% Define your continuous signal and sampling frequency
fs = 1250; % Sampling frequency (Hz)
signal = x30;

% Define STFT parameters
window_length = 0.3 * fs; % Length of the analysis window (100 milliseconds in samples)
window_length = 0.1 * fs; % Length of the analysis window (100 milliseconds in samples)
noverlap = floor(window_length / 4); % 50% overlap between consecutive windows
fft_length = 128;

% Calculate the number of time frames
num_frames = floor((length(signal) - noverlap) / (window_length - noverlap));


% Define the user-specified frequency band ranges as [f_min, f_max] pairs in Hz
user_frequency_bands = [
    8 13;  % Alpha band
    13 30; % Beta band
    30 80  % Gamma band
    ];

for channel = channels
    close all
    x30 = lfp(1 + fs * collectStart : fs * (collectStart + collectFor), channel);

    % Initialize a cell array to store average powers for each frequency band
    avg_powers_per_band = cell(size(user_frequency_bands, 1), 1);

    % Initialize variables to store power spectrogram
    power_spectrogram = [];

    % Perform STFT for each time frame
    for j = 1:num_frames
        start_idx = (j - 1) * (window_length - noverlap) + 1;
        end_idx = start_idx + window_length - 1;
        if end_idx > length(signal)
            break; % Stop if the last window extends beyond the signal length
        end

        % Extract the current window of the signal
        current_window = signal(start_idx:end_idx);



        % Calculate the STFT for the current window

        % Original
        [S, f_stft] = spectrogram(current_window, hamming(window_length), noverlap, fft_length, fs);

        % [s, f, t] = stft(signal,fs,Window=kaiser(125,5),OverlapLength=220,FFTLength=512);
        % stft(signal,fs,Window=kaiser(125,5),OverlapLength=0,FFTLength=512);

        % Conan version
        [S, f_stft] = spectrogram(current_window, hamming(window_length), 0, 125, fs);
        % [S, f_stft] = spectrogram(current_window, rectwin(window_length), 0, 125, fs);





        % Calculate power for this frame and store it
        power_frame = abs(S).^2;
        power_spectrogram = [power_spectrogram, power_frame];

        % Calculate average power for user-specified frequency bands
        for j = 1:size(user_frequency_bands, 1)
            freq_band_indices = find(f_stft >= user_frequency_bands(j, 1) & f_stft <= user_frequency_bands(j, 2));
            avg_power_band = mean(power_frame(freq_band_indices, :), 1);
            avg_powers_per_band{j} = [avg_powers_per_band{j}, avg_power_band];
        end
    end

    % Plot only power in 0-100 Hz range
    plotIdx = f_stft >= 1 & f_stft <= 100;

    % Plot the power spectrogram
    nRow = 1;
    nColumn = 2;
    screenOrSave = 'screen';
    squareAxes = 0;
    % [axisWidth, axisHeight, xAxesPosition, yAxesPosition] = big_figure(nRow, nColumn, figureHandle, screenOrSave);
    [axisWidth, axisHeight, xAxesPosition, yAxesPosition] = screen_figure(nRow, nColumn, figureHandle, squareAxes);
    ax(1) = axes('units', 'centimeters', 'position', [xAxesPosition(1, 1) yAxesPosition(1, 1) axisWidth axisHeight]);
    hold(ax(1))
    ax(2) = axes('units', 'centimeters', 'position', [xAxesPosition(1, 2) yAxesPosition(1, 2) axisWidth axisHeight]);
    hold(ax(2))


    % figure;
    % imagesc((1:num_frames) * (window_length - overlap) / fs, 0:fs/2, 10*log10(power_spectrogram(plotIdx,:))); % Convert to dB
    imagesc(ax(2), (1:num_frames) * (window_length - noverlap) / fs, f_stft(plotIdx), 10*log10(power_spectrogram(plotIdx,:))); % Convert to dB
    axis xy; % Flip the y-axis to have lower frequencies at the bottom
    xlabel(ax(2), 'Time (s)');
    ylabel(ax(2), 'Frequency (Hz)');
    title(ax(2), 'Power Spectrogram');
    colorbar;


    % Plot average power for each user-specified frequency band in the same axes
    % figure;
    % hold on;
    for j = 1:size(user_frequency_bands, 1)
        plot(ax(1), (1:num_frames) * (window_length - noverlap) / fs, 10*log10(avg_powers_per_band{j, :}), 'DisplayName', ...
            ['[', num2str(user_frequency_bands(j, 1)), ' - ', num2str(user_frequency_bands(j, 2)), ' Hz]']);
    end
    % hold off;
    xlabel(ax(1), 'Time (s)');
    ylabel(ax(1), 'Average Power (dB)');
    title(ax(1), 'Average Power in User-Specified Frequency Bands');
    legend(ax(1),'Location', 'northwest');

    pause
    clf
end




%% Fieldtrip testing

% make a pretend cfg from ft_definetrial, getting a single long stretch of
% lfp data to analyze
onset = 1;
offest = length(x30);
lag = 0;
cfg = [];
cfg.trl = [onset offset lag];


addpath('E:/Projects/toolboxes/fieldtrip-20230118/')

cfg = [];
cfg.method = 'mtmfft';
cfg.output = 'pow';
cfg.channel = 'channel';
cfg.taper = 'hanning';
cfg.foi          = 0:5:80;                         % analysis 2 to 30 Hz in steps of 2 Hz
cfg.t_ftimwin    = ones(length(cfg.foi),1).*0.5;   % length of time window = 0.5 sec
cfg.toi          = -0.5:0.1:1.5;                  % time window "slides" from -0.5 to 1.5 sec in steps of 0.05 sec (50 ms)



