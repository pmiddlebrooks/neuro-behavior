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

traceData = load(fullfile(paths.dropPath, 'reach_data/Y4_06-Oct-2023 14_14_53_NIBEH.mat'));
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
jsTrace = zscore(traceData.JSNIpos); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
jsX = zscore(traceData.NIDATA(7,2:end)); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
jsY = zscore(traceData.NIDATA(8,2:end)); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
jsAmp = sqrt(traceData.NIDATA(7,2:end).^2 + traceData.NIDATA(8,2:end).^2);
jsAmp = movmean(jsAmp, 61);
jsAmp = zscore(jsAmp); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)

% jsTrace = traces.JSNIpos ./ max(traces.JSNIpos); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
errTrace = traceData.ERtrace ./ max(traceData.ERtrace); % Continuous voltage trace of error light (for error trials)
solTrace = traceData.SOLtrace ./ max(traceData.SOLtrace); % Continous voltage trace of solenoid (for correct rewarded trials)

% Lick times
filtLick = bandpass(traceData.NIDATA(6,2:end), [1 10], 1000); 
filtLick = abs(filtLick); 
filtLick(filtLick>300)=300;
filtLick = movmean(filtLick, 101);
filtLick = zscore(filtLick);
% Event times
errTimes = traceData.ERtimes; % Error light turns on (errors)
solTimes = traceData.SOLtimes; % Reward solenoid opens (correct)

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
    jsAmpBins = hmmMatrix(:,1);
    errTraceBins = hmmMatrix(:,2);
    solTraceBins = hmmMatrix(:,3);
    filtLickBins = hmmMatrix(:,4);
end

if testingFlag
% test the code with 1/4 of the data for speed
hmmMatrix = hmmMatrix(1:round(size(hmmMatrix, 1)/4),:);
disp('Testing with much less data line 116')
end

optsHmm.minState = .02; % Minimum state duration in seconds
optsHmm.stateRange = 3:15; % Maximum number of states
optsHmm.numFolds = 3; % Number of folds for cross-validation
optsHmm.plotFlag = true; % Plot summary statistics across states
optsHmm.penaltyType = 'normalized'; % normalized  bic  aic

% Fit HMM to find best model
[bestNumStates, stateEstimates, hmmModels, penalizedLikelihoods, stateProbabilities] = fit_hmm_crossval_cov_penalty(hmmMatrix, optsHmm);

% Display results
fprintf('Best number of states: %d\n', bestNumStates);
fprintf('State estimates shape: %s\n', mat2str(size(stateEstimates)));
fprintf('State probabilities shape: %s\n', mat2str(size(stateProbabilities)));

% Plot state estimates with rectangles and state probabilities
figure(89); clf; hold on;

% Get colors for states using distinguishable_colors
numStates = bestNumStates;
colors = distinguishable_colors(numStates);

% Plot rectangles for each state
for state = 1:numStates
    stateIdx = stateEstimates == state;
    if any(stateIdx)
        % Find contiguous segments
        d = diff([0; stateIdx; 0]);
        startIdx = find(d == 1);
        endIdx = find(d == -1) - 1;
        
        % Plot each segment as a rectangle
        for i = 1:length(startIdx)
            xStart = startIdx(i);
            xEnd = endIdx(i);
            patch([xStart xEnd xEnd xStart], [0 0 1 1], colors(state,:), ...
                'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
    end
end

% Overlay state probability traces with same color scheme
for state = 1:numStates
    plot(stateProbabilities(:, state), 'Color', colors(state,:), 'LineWidth', 2);
end

xlabel('Time (bins)');
ylabel('State Probability');
title(sprintf('HMM State Estimates and Probabilities (Best: %d states)', bestNumStates));
ylim([0 1]);
grid on;
hold off;

%% Plot comparison of all models 

plotAllModelsComparison(hmmMatrix, hmmModels, optsHmm.stateRange, optsHmm.minState);

% Overlay behavioral variables on the model comparison plot
figure(90); hold on;

% Get the current y-axis limits to position the overlays
ylimits = ylim;
yRange = ylimits(2) - ylimits(1);

% Scale and position the behavioral variables to fit within the plot
% Position them at the bottom with some spacing
overlayY = ylimits(1) - yRange * 0.1; % 10% below the bottom of the plot

% Overlay jsAmp (joystick amplitude)
jsAmpScaled = zscore(jsAmpBins) * (yRange * 0.15) + overlayY;
plot(jsAmpScaled, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Joystick Amplitude');

% Overlay errTrace (error light)
errTraceScaled = zscore(errTraceBins) * (yRange * 0.15) + overlayY;
plot(errTraceScaled, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Error Light');

% Overlay solTrace (solenoid/reward)
solTraceScaled = zscore(solTraceBins) * (yRange * 0.15) + overlayY;
plot(solTraceScaled, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Solenoid/Reward');

% Overlay filtLick (licking)
filtLickScaled = zscore(filtLickBins) * (yRange * 0.15) + overlayY;
plot(filtLickScaled, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Licking');

% Add legend for the behavioral variables
% legend('Location', 'southeast', 'NumColumns', 2);

% Update y-axis limits to include the overlays
ylim([overlayY - yRange * 0.1, ylimits(2)]);


hold off;

%% Save HMM results and models
% Setup directories
hmmdir = fullfile(paths.dropPath, 'hmm');
if ~exist(hmmdir, 'dir')
    mkdir(hmmdir);
end

% Create comprehensive results structure
hmm_results = struct();

% Analysis metadata
hmm_results.metadata = struct();
hmm_results.metadata.analysis_date = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
hmm_results.metadata.data_type = 'Reach';
hmm_results.metadata.analysis_status = 'SUCCESS';

% Data parameters
hmm_results.data_params = struct();
hmm_results.data_params.bin_size_data = binSizeData; % Original data bin size (1ms)
hmm_results.data_params.bin_size_model = binSizeHmm; % HMM model bin size (10ms)
hmm_results.data_params.downsample_factor = downsampleFactor;
hmm_results.data_params.num_time_bins = size(hmmMatrix, 1);
hmm_results.data_params.num_features = size(hmmMatrix, 2);

% HMM options and parameters
hmm_results.hmm_options = optsHmm;

% Model selection results
hmm_results.model_selection = struct();
hmm_results.model_selection.candidate_states = optsHmm.stateRange;
hmm_results.model_selection.best_num_states = bestNumStates;
hmm_results.model_selection.penalized_likelihoods = penalizedLikelihoods;

% All HMM models
hmm_results.all_models = hmmModels;

% State estimates for all models
hmm_results.state_estimates = struct();
for i = 1:length(optsHmm.stateRange)
    numStates = optsHmm.stateRange(i);
    if ~isempty(hmmModels{i})
        % Get state estimates for this model
        initialStateEstimates = cluster(hmmModels{i}, hmmMatrix);
        stateEstimates = filterStatesByDuration(initialStateEstimates, optsHmm.minState, 0.005);
        
        % Store in structure
        fieldName = sprintf('states_%d', numStates);
        hmm_results.state_estimates.(fieldName) = stateEstimates;
    end
end

% Best model results
hmm_results.best_model = struct();
hmm_results.best_model.num_states = bestNumStates;
hmm_results.best_model.state_estimates = stateEstimates;
hmm_results.best_model.state_probabilities = stateProbabilities;
hmm_results.best_model.model = hmmModels{find(optsHmm.stateRange == bestNumStates)};

% Behavioral data (downsampled to match HMM)
hmm_results.behavioral_data = struct();
hmm_results.behavioral_data.jsAmp = jsAmpBins;
hmm_results.behavioral_data.errTrace = errTraceBins;
hmm_results.behavioral_data.solTrace = solTraceBins;
hmm_results.behavioral_data.filtLick = filtLickBins;

% Create filename with timestamp and analysis info
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
filename = sprintf('HMM_results_Reach_%s.mat', timestamp);
filepath = fullfile(hmmdir, filename);

% Save the results
fprintf('Saving HMM results to: %s\n', filepath);
save(filepath, 'hmm_results', '-v7.3');

% Also save a summary text file
summary_filename = sprintf('HMM_summary_Reach_%s.txt', timestamp);
summary_filepath = fullfile(hmmdir, summary_filename);

% Create summary file
fid = fopen(summary_filepath, 'w');
if fid ~= -1
    fprintf(fid, 'HMM Analysis Summary\n');
    fprintf(fid, '===================\n\n');
    fprintf(fid, 'Analysis Date: %s\n', hmm_results.metadata.analysis_date);
    fprintf(fid, 'Data Type: %s\n', hmm_results.metadata.data_type);
    fprintf(fid, '\n');
    fprintf(fid, 'Data Parameters:\n');
    fprintf(fid, '  Original bin size: %.6f seconds (%.1f ms)\n', hmm_results.data_params.bin_size_data, hmm_results.data_params.bin_size_data*1000);
    fprintf(fid, '  Model bin size: %.6f seconds (%.1f ms)\n', hmm_results.data_params.bin_size_model, hmm_results.data_params.bin_size_model*1000);
    fprintf(fid, '  Downsample factor: %d\n', hmm_results.data_params.downsample_factor);
    fprintf(fid, '  Number of time bins: %d\n', hmm_results.data_params.num_time_bins);
    fprintf(fid, '  Number of features: %d\n', hmm_results.data_params.num_features);
    fprintf(fid, '\n');
    fprintf(fid, 'HMM Parameters:\n');
    fprintf(fid, '  State range tested: %s\n', mat2str(optsHmm.stateRange));
    fprintf(fid, '  Best number of states: %d\n', bestNumStates);
    fprintf(fid, '  Minimum state duration: %.3f seconds\n', optsHmm.minState);
    fprintf(fid, '  Number of CV folds: %d\n', optsHmm.numFolds);
    fprintf(fid, '  Penalty type: %s\n', optsHmm.penaltyType);
    fprintf(fid, '\n');
    fprintf(fid, 'Files saved:\n');
    fprintf(fid, '  Results: %s\n', filename);
    fprintf(fid, '  Summary: %s\n', summary_filename);
    fclose(fid);
    
    fprintf('Summary saved to: %s\n', summary_filepath);
end

fprintf('HMM analysis and results saved successfully!\n');

end

function plotAllModelsComparison(hmmMatrix, hmmModels, stateRange, minStateDuration)
% PLOTALLMODELSCOMPARISON Plots state estimates for all HMM models in a single stacked view
%
% Inputs:
%   hmmMatrix - Input data matrix
%   hmmModels - Cell array of fitted HMM models
%   stateRange - Vector of state numbers tested
%   minStateDuration - Minimum state duration for filtering

figure(90); clf; hold on;

% Find maximum number of states across all models
maxStates = max(stateRange);
numModels = length(stateRange);

% Plot each model in a stacked format
for i = 1:numModels
    numStates = stateRange(i);
    
    % Skip if model is empty
    if isempty(hmmModels{i})
        continue;
    end
    
    % Get state estimates for this model
    initialStateEstimates = cluster(hmmModels{i}, hmmMatrix);
    stateEstimates = filterStatesByDuration(initialStateEstimates, minStateDuration, 0.005);
    
    % Calculate vertical position for this model
    % Center each model on its state number with ±0.5 span
    yCenter = numStates;
    yStart = yCenter - 0.5;
    yEnd = yCenter + 0.5;
    
    % Get colors for states
    colors = distinguishable_colors(numStates);
    
    % Plot rectangles for each state
    for state = 1:numStates
        stateIdx = stateEstimates == state;
        if any(stateIdx)
            % Find contiguous segments
            d = diff([0; stateIdx; 0]);
            startIdx = find(d == 1);
            endIdx = find(d == -1) - 1;
            
            % Plot each segment as a rectangle
            for j = 1:length(startIdx)
                xStart = startIdx(j);
                xEnd = endIdx(j);
                
                % Position each state within its model's range
                % Each model is exactly one row high, centered on its state number
                stateYStart = yCenter - 0.5;  % Bottom of the single row
                stateYEnd = yCenter + 0.5;    % Top of the single row
                
                patch([xStart xEnd xEnd xStart], [stateYStart stateYStart stateYEnd stateYEnd], colors(state,:), ...
                    'FaceAlpha', 0.4, 'EdgeColor', 'none');
            end
        end
    end
    
    % Add model label on the right side
    text(size(hmmMatrix, 1) + 50, yCenter, sprintf('%d States', numStates), ...
        'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
end

% Set plot properties
xlabel('Time (bins)');
ylabel('State Number');
title('HMM State Estimates Comparison - All Models Stacked', 'FontSize', 14, 'FontWeight', 'bold');

% Set y-axis limits and ticks
% Y-axis should show the state numbers with some padding
ylim([0, maxStates + 1]);
yticks(0:maxStates);
yticklabels(0:maxStates);

% Add grid for better readability
grid on;
grid minor;

% Add legend for state colors (if not too many states)
if maxStates <= 10
    legendHandles = [];
    legendLabels = {};
    for state = 1:maxStates
        colors = distinguishable_colors(maxStates);
        h = patch([0 1 1 0], [0 0 1 1], colors(state,:), 'FaceAlpha', 0.6);
        % legendHandles = [legendHandles, h];
        % legendLabels{state} = sprintf('State %d', state);
    end
    % legend(legendHandles, legendLabels, 'Location', 'northeast', 'NumColumns', 2);
end

hold off;
end

function colors = distinguishable_colors(n)
% DISTINGUISHABLE_COLORS Generates distinguishable colors for plotting
%
% Inputs:
%   n - Number of colors needed
%
% Outputs:
%   colors - n x 3 matrix of RGB values

if n <= 7
    % Use MATLAB's built-in lines for small numbers
    colors = lines(n);
else
    % Use HSV color space for larger numbers
    hues = linspace(0, 1, n);
    saturations = 0.7 * ones(1, n);
    values = 0.9 * ones(1, n);
    
    % Convert HSV to RGB
    colors = hsv2rgb([hues', saturations', values']);
    
    % Ensure first few colors are distinct
    if n > 7
        colors(1:7, :) = lines(7);
    end
end
end

function filteredStates = filterStatesByDuration(stateEstimates, minDuration, binSize)
% FILTERSTATESBYDURATION Filters out state segments that are shorter than minDuration
%
% Inputs:
%   stateEstimates - Vector of state assignments
%   minDuration - Minimum duration in seconds
%   binSize - Size of each bin in seconds
%
% Outputs:
%   filteredStates - Vector with short segments set to 0

minBins = round(minDuration / binSize);
filteredStates = stateEstimates;

% Find continuous segments of each state
uniqueStates = unique(stateEstimates);
for state = uniqueStates'
    if state == 0
        continue; % Skip already filtered states
    end

    stateIndices = find(stateEstimates == state);

    % Find continuous segments
    segmentStarts = [stateIndices(1); stateIndices(find(diff(stateIndices) > 1) + 1)];
    segmentEnds = [stateIndices(find(diff(stateIndices) > 1)); stateIndices(end)];

    % Filter short segments
    for seg = 1:length(segmentStarts)
        segmentLength = segmentEnds(seg) - segmentStarts(seg) + 1;
        if segmentLength < minBins
            filteredStates(segmentStarts(seg):segmentEnds(seg)) = 0;
        end
    end
end
end




