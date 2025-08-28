function bhvID = get_reach_bhv_labels(fileName, frameSize)

% All data times are in ms


% Data from a joystick reach task. The mouse, self-paced, learns to
% move the joystick into a target area. The mouse has to learn the target
% area without perceiving it, through trial and error. Each self-initiated trial can be
% correct, resulting in reward, or an error, resulting in an error light.
% Errors can be above or below the target area. A session is divided into
% blocks, in which the target area changes without notice.
% I want to generate a vector of categorical behavior labels for all time
% points throughout the session. 
%%
paths = get_paths;

traces = load(fullfile(paths.dropPath, 'reach_data/Y4_06-Oct-2023 14_14_53_NIBEH.mat'));
dataR = load(fullfile(paths.dropPath, 'reach_data/Copy_of_Y4_100623_Spiketimes_idchan_BEH.mat'));

binSizeData = .001; % Data is in ms
binSizeHmm = .01; % For hmm
%%
% NIDATA rows
% 1 - cortical opto
% 2 - gpe opto
% 3 - snr opto
% 4 - error light
% 5 - Solenoid open
% 6 - Lick sensor - (this signal needs processing - see below )
% 7 - X
% 8 - Y
% 9 - square wave for timing


reachStart = dataR.R(:,1);
reachStop = dataR.R(:,2);
reachAmp = dataR.R(:,3); % Amplitude of each reach (distance from 0)

 % block or BlockWithErrors - columns are BlockNum Correct? ReachClassification1-4 ReachClassification-20-20
 %     -rows are reaches
 %     classification1-4:
 %     - 1 - block 1 error
 %     - 2 - block 1 Rew
 %     - 3 - block 2 error
 %     - 4 - block 2 rew
 %     - 5 - block 3 error
 %     - 6 - block 3 rew
 % 
 % 
 %     classification-20-20:
 %     - -10 - block 1 error below
 %     -  1   - block 1 Rew
 %     - 10 - block 1 error above
 %     - -20 - block 2 error below
 %     -  2   - block 2 Rew
 %     - 20 - block 2 error above

reachClass = dataR.block(:,4);

% Continuous traces normalized to max
jsTrace = zscore(traces.JSNIpos); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
jsX = zscore(traces.NIDATA(7,2:end)); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
jsY = zscore(traces.NIDATA(8,2:end)); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
jsAmp = sqrt(traces.NIDATA(7,2:end).^2 + traces.NIDATA(8,2:end).^2);
jsAmp = movmean(jsAmp, 61);
jsAmp = zscore(jsAmp); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)

% jsTrace = traces.JSNIpos ./ max(traces.JSNIpos); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
errTrace = traces.ERtrace ./ max(traces.ERtrace); % Continuous voltage trace of error light (for error trials)
solTrace = traces.SOLtrace ./ max(traces.SOLtrace); % Continous voltage trace of solenoid (for correct rewarded trials)

% Lick times
filtLick = bandpass(traces.NIDATA(6,2:end), [1 10], 1000); 
filtLick = abs(filtLick); 
filtLick(filtLick>300)=300;
filtLick = movmean(filtLick, 101);
filtLick = zscore(filtLick);
% Event times
errTimes = traces.ERtimes; % Error light turns on (errors)
solTimes = traces.SOLtimes; % Reward solenoid opens (correct)

%%
figure(88); clf; hold on;

% plot(jsTrace, 'k');
plot(jsAmp, 'k');
plot(errTrace, 'red');
plot(solTrace, 'color', [0 .6 0]);
plot(filtLick, 'b');
scatter(reachStart, -.025, 'k')
scatter(errTimes, -.05, 'MarkerFaceColor', 'r');
scatter(solTimes, -.075*ones(length(solTimes)), 'MarkerFaceColor',[0 .6 0]);

%%
testingFlag = 0;

hmmMatrix = [jsAmp', errTrace', solTrace', filtLick'];
    
% Downsample hmmMatrix from binSizeData (1 ms) to binSizeHmm (20 ms)
% Calculate downsampling factor
downsampleFactor = round(binSizeHmm / binSizeData); 
if downsampleFactor > 1
    % Downsample by taking mean of each bin
    numBins = floor(size(hmmMatrix, 1) / downsampleFactor);
    hmmMatrixDownsampled = zeros(numBins, size(hmmMatrix, 2));
    
    for i = 1:numBins
        startIdx = (i-1) * downsampleFactor + 1;
        endIdx = min(i * downsampleFactor, size(hmmMatrix, 1));
        hmmMatrixDownsampled(i, :) = mean(hmmMatrix(startIdx:endIdx, :), 1);
    end
    hmmMatrix = hmmMatrixDownsampled;
end

if testingFlag
% test the code with 1/4 of the data for speed
hmmMatrix = hmmMatrix(1:round(size(hmmMatrix, 1)/4),:);
disp('Testing with much less data line 116')
end

optsHmm.minState = .02; % Minimum state duration in seconds
optsHmm.maxNumStates = 16; % Maximum number of states
optsHmm.numFolds = 3; % Number of folds for cross-validation
optsHmm.lambda = 1; % Regularization parameter
optsHmm.plotFlag = true; % Plot summary statistics across states

% Fit HMM to find best model
[bestNumStates, stateEstimates, hmmModels, penalizedLikelihoods, stateProbabilities] = fit_hmm_crossval_cov_penalty(hmmMatrix, optsHmm);

% Display results
fprintf('Best number of states: %d\n', bestNumStates);
fprintf('State estimates shape: %s\n', mat2str(size(stateEstimates)));
fprintf('State probabilities shape: %s\n', mat2str(size(stateProbabilities)));

% Plot state estimates
figure(89); clf; hold on;
plot(stateEstimates, 'k-', 'LineWidth', 1.5);
xlabel('Time (bins)');
ylabel('State');
title(sprintf('HMM State Estimates (Best: %d states)', bestNumStates));
grid on;

% Plot state probabilities
figure(90); clf; hold on;
for state = 1:bestNumStates
    plot(stateProbabilities(:, state), 'LineWidth', 1.5);
end
xlabel('Time (bins)');
ylabel('Probability');
title(sprintf('HMM State Probabilities (Best: %d states)', bestNumStates));
legend(arrayfun(@(x) sprintf('State %d', x), 1:bestNumStates, 'UniformOutput', false));
grid on;




