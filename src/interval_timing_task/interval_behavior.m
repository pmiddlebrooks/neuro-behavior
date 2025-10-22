function interval_behavior()
% INTERVAL_BEHAVIOR - Analyze interval timing task performance
% 
% Variables:
%   dataPath - path to log file directory
%   resultsPath - path to save analysis results
%   logFile - name of log file
%   interTrialInterval - duration between trials (ms)
%   rewardWindow - duration reward is available (ms)
%   mouseName - identifier for the mouse
%
% Goal: Analyze mouse performance on interval timing task, calculating
% accuracy, error types, trial durations, and generating visualizations
% to assess learning progression.

%%
    % Task parameters from Arduino code
    interTrialInterval = 8000;  % 8 seconds in ms
    rewardWindow = 3000;        % 3 seconds in ms
    
    % File paths
    dataPath = 'E:\Dropbox\Data\interval_timing_task\data\mouse99';
    resultsPath = 'E:\Dropbox\Data\interval_timing_task\results\mouse99';
    logFile = 'log.txt';
    
    % Extract mouse name from path
    mouseName = 'mouse99';
    
    % Create results directory if it doesn't exist
    if ~exist(resultsPath, 'dir')
        mkdir(resultsPath);
    end
    
    % Parse log data
    fprintf('Parsing log data...\n');
    [events, states] = parse_log_data(fullfile(dataPath, logFile));
    
    % Analyze trials
    fprintf('Analyzing trials...\n');
    trials = analyze_trials(events, states, interTrialInterval, rewardWindow);
    
    % Calculate performance metrics
    fprintf('Calculating performance metrics...\n');
    metrics = calculate_metrics(trials);
    
    % Generate plots
    fprintf('Generating plots...\n');
    create_session_plot(events, trials, mouseName, interTrialInterval, rewardWindow);
    create_performance_plot(trials, mouseName, interTrialInterval, rewardWindow);
    
    % Save results
    fprintf('Saving results...\n');
    save_results(resultsPath, trials, metrics, mouseName, interTrialInterval, rewardWindow);
    
    % Display summary
    display_summary(metrics, mouseName);
    
    fprintf('Analysis complete!\n');
end

function [events, states] = parse_log_data(logFilePath)
% PARSE_LOG_DATA - Parse Arduino log file into structured data
%
% Variables:
%   logFilePath - path to log file
%   fid - file identifier
%   rawData - raw text data from file
%   timestamps - all timestamps
%   eventTypes - all event types
%   values - all values/states
%   stateIdx - logical index for state events
%   eventIdx - logical index for regular events
%
% Goal: Extract all events and states from log file for analysis

    fprintf('Reading log file...\n');
    
    % Read entire file at once - much faster than line by line
    fid = fopen(logFilePath, 'r');
    if fid == -1
        error('Could not open log file: %s', logFilePath);
    end
    
    % Read all lines at once
    rawData = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    rawData = rawData{1};
    
    fprintf('Parsing %d lines...\n', length(rawData));
    
    % Pre-allocate arrays for maximum efficiency
    nLines = length(rawData);
    timestamps = zeros(nLines, 1);
    eventTypes = cell(nLines, 1);
    values = cell(nLines, 1);
    
    % Parse all lines efficiently
    for i = 1:nLines
        line = rawData{i};
        if isempty(line)
            continue;
        end
        
        % Parse line: timestamp eventType value/state
        parts = strsplit(line, ' ');
        if length(parts) < 3
            continue;
        end
        
        timestamps(i) = str2double(parts{1});
        eventTypes{i} = parts{2};
        
        if strcmp(parts{2}, 'S')
            % State event - join remaining parts
            values{i} = strjoin(parts(3:end), ' ');
        else
            % Regular event - convert value to number
            values{i} = str2double(parts{3});
        end
    end
    
    % Remove empty entries
    validIdx = ~cellfun(@isempty, eventTypes);
    timestamps = timestamps(validIdx);
    eventTypes = eventTypes(validIdx);
    values = values(validIdx);
    
    % Separate state events from regular events
    stateIdx = strcmp(eventTypes, 'S');
    eventIdx = ~stateIdx;
    
    % Create events structure
    events = struct('timestamp', num2cell(timestamps(eventIdx)), ...
                   'type', eventTypes(eventIdx), ...
                   'value', values(eventIdx));
    
    % Create states structure
    states = struct('timestamp', num2cell(timestamps(stateIdx)), ...
                  'state', values(stateIdx));
    
    fprintf('Parsed %d events and %d states\n', length(events), length(states));
end

function trials = analyze_trials(events, states, interTrialInterval, rewardWindow)
% ANALYZE_TRIALS - Identify and analyze individual trials
%
% Variables:
%   events - parsed event data
%   states - parsed state data
%   interTrialInterval - duration between trials (ms)
%   rewardWindow - duration reward is available (ms)
%   trialStartTimes - timestamps when trials begin
%   trialEndTimes - timestamps when trials end
%   eventTimestamps - all event timestamps
%   eventTypes - all event types
%   eventValues - all event values
%
% Goal: Identify trial boundaries and classify trial outcomes efficiently

    fprintf('Analyzing trials...\n');
    
    % Extract arrays for faster processing
    eventTimestamps = [events.timestamp];
    eventTypes = {events.type};
    eventValues = [events.value];
    
    % Find trial start times (TRIAL_ARMED states)
    trialStartIdx = strcmp({states.state}, 'TRIAL_ARMED');
    trialStartTimes = [states(trialStartIdx).timestamp];
    
    % Find trial end times (ITI_START states)
    trialEndIdx = strcmp({states.state}, 'ITI_START');
    trialEndTimes = [states(trialEndIdx).timestamp];
    
    fprintf('Found %d trial starts and %d trial ends\n', length(trialStartTimes), length(trialEndTimes));
    
    % Pre-allocate trial structure
    nTrials = length(trialStartTimes);
    trials = struct('startTime', cell(nTrials, 1), 'endTime', cell(nTrials, 1), ...
                   'type', cell(nTrials, 1), 'duration', cell(nTrials, 1), ...
                   'rewardWindowStart', cell(nTrials, 1), 'beamBreakTime', cell(nTrials, 1));
    
    % Process trials efficiently
    for i = 1:nTrials
        startTime = trialStartTimes(i);
        
        % Find the next ITI_START after this trial start
        endIdx = find(trialEndTimes > startTime, 1);
        if isempty(endIdx)
            continue; % Skip incomplete trial
        end
        endTime = trialEndTimes(endIdx);
        
        % Find events within this trial period
        trialEventIdx = eventTimestamps >= startTime & eventTimestamps < endTime;
        trialTimestamps = eventTimestamps(trialEventIdx);
        trialTypes = eventTypes(trialEventIdx);
        trialValues = eventValues(trialEventIdx);
        
        % Find reward window start (LED on)
        ledOnIdx = find(strcmp(trialTypes, 'L') & trialValues == 1, 1);
        if isempty(ledOnIdx)
            continue;
        end
        rewardWindowStart = trialTimestamps(ledOnIdx);
        
        % Find beam break during reward window
        rewardWindowIdx = trialTimestamps >= rewardWindowStart;
        beamBreakIdx = find(rewardWindowIdx & strcmp(trialTypes, 'B') & trialValues == 1, 1);
        
        % Classify trial type
        if ~isempty(beamBreakIdx)
            beamBreakTime = trialTimestamps(beamBreakIdx);
            trialType = 'correct';
        else
            beamBreakTime = NaN;
            % Check if reward window expired
            ledOffIdx = find(rewardWindowIdx & strcmp(trialTypes, 'L') & trialValues == 0, 1);
            if ~isempty(ledOffIdx)
                trialType = 'abort';
            else
                trialType = 'unknown';
            end
        end
        
        % Check for early errors (beam break before reward window)
        earlyIdx = trialTimestamps < rewardWindowStart;
        earlyBeamIdx = find(earlyIdx & strcmp(trialTypes, 'B') & trialValues == 1, 1);
        if ~isempty(earlyBeamIdx)
            trialType = 'early_error';
        end
        
        % Store trial data
        trials(i).startTime = startTime;
        trials(i).endTime = endTime;
        trials(i).type = trialType;
        trials(i).duration = endTime - startTime;
        trials(i).rewardWindowStart = rewardWindowStart;
        trials(i).beamBreakTime = beamBreakTime;
    end
    
    % Remove empty trials
    validTrials = ~cellfun(@isempty, {trials.type});
    trials = trials(validTrials);
    
    fprintf('Analyzed %d trials\n', length(trials));
end

function metrics = calculate_metrics(trials)
% CALCULATE_METRICS - Calculate performance metrics from trial data
%
% Variables:
%   trials - analyzed trial data
%   totalTrials - total number of trials
%   correctTrials - number of correct trials
%   earlyErrorTrials - number of early error trials
%   abortTrials - number of abort trials
%   accuracy - percentage of correct trials
%   correctDurations - durations of correct trials
%   errorDurations - durations of error trials
%
% Goal: Calculate accuracy, error rates, and trial duration statistics

    if isempty(trials)
        metrics = struct();
        return;
    end
    
    totalTrials = length(trials);
    correctTrials = sum(strcmp({trials.type}, 'correct'));
    earlyErrorTrials = sum(strcmp({trials.type}, 'early_error'));
    abortTrials = sum(strcmp({trials.type}, 'abort'));
    
    accuracy = (correctTrials / totalTrials) * 100;
    
    % Trial durations
    correctIdx = strcmp({trials.type}, 'correct');
    errorIdx = strcmp({trials.type}, 'early_error') | strcmp({trials.type}, 'abort');
    
    correctDurations = [trials(correctIdx).duration];
    errorDurations = [trials(errorIdx).duration];
    
    metrics.totalTrials = totalTrials;
    metrics.correctTrials = correctTrials;
    metrics.earlyErrorTrials = earlyErrorTrials;
    metrics.abortTrials = abortTrials;
    metrics.accuracy = accuracy;
    metrics.correctDurationMean = mean(correctDurations);
    metrics.correctDurationStd = std(correctDurations);
    metrics.errorDurationMean = mean(errorDurations);
    metrics.errorDurationStd = std(errorDurations);
end

function create_session_plot(events, trials, mouseName, interTrialInterval, rewardWindow)
% CREATE_SESSION_PLOT - Plot task events throughout the session
%
% Variables:
%   events - parsed event data
%   trials - analyzed trial data
%   mouseName - mouse identifier
%   interTrialInterval - trial interval duration
%   rewardWindow - reward window duration
%   sessionDuration - total session duration
%   ledEvents - LED on/off events
%   rewardEvents - solenoid events
%   beamEvents - beam break events
%
% Goal: Visualize task events over time to assess learning progression

    if isempty(events) || isempty(trials)
        return;
    end
    
    sessionDuration = max([events.timestamp]) - min([events.timestamp]);
    
    % Extract different event types
    ledEvents = events(strcmp({events.type}, 'L'));
    rewardEvents = events(strcmp({events.type}, 'R'));
    beamEvents = events(strcmp({events.type}, 'B'));
    
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot LED events
    subplot(4,1,1);
    plot([ledEvents.timestamp]/1000, [ledEvents.value], 'g-', 'LineWidth', 2);
    ylabel('LED State');
    ylim([-0.1, 1.1]);
    title(sprintf('%s - Session Events (Interval: %ds, Reward Window: %ds)', ...
          mouseName, interTrialInterval/1000, rewardWindow/1000));
    
    % Plot reward events
    subplot(4,1,2);
    plot([rewardEvents.timestamp]/1000, [rewardEvents.value], 'r-', 'LineWidth', 2);
    ylabel('Solenoid');
    ylim([-0.1, 1.1]);
    
    % Plot beam events
    subplot(4,1,3);
    plot([beamEvents.timestamp]/1000, [beamEvents.value], 'b-', 'LineWidth', 2);
    ylabel('Beam State');
    ylim([-0.1, 1.1]);
    
    % Plot trial outcomes
    subplot(4,1,4);
    hold on;
    colors = containers.Map({'correct', 'early_error', 'abort'}, {'g', 'r', 'm'});
    
    for i = 1:length(trials)
        trial = trials(i);
        color = colors(trial.type);
        plot([trial.startTime, trial.endTime]/1000, [i, i], color, 'LineWidth', 3);
    end
    
    ylabel('Trial Number');
    xlabel('Time (seconds)');
    legend({'Correct', 'Early Error', 'Abort'}, 'Location', 'best');
    
    % Add trial outcome markers
    for i = 1:length(trials)
        trial = trials(i);
        if strcmp(trial.type, 'correct') && ~isnan(trial.beamBreakTime)
            plot(trial.beamBreakTime/1000, i, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
        end
    end
end

function create_performance_plot(trials, mouseName, interTrialInterval, rewardWindow)
% CREATE_PERFORMANCE_PLOT - Plot performance metrics over time
%
% Variables:
%   trials - analyzed trial data
%   mouseName - mouse identifier
%   interTrialInterval - trial interval duration
%   rewardWindow - reward window duration
%   trialNumbers - trial sequence numbers
%   runningAccuracy - cumulative accuracy
%   trialDurations - duration of each trial
%
% Goal: Show learning progression through accuracy and timing

    if isempty(trials)
        return;
    end
    
    trialNumbers = 1:length(trials);
    runningAccuracy = zeros(size(trialNumbers));
    trialDurations = [trials.duration];
    
    % Calculate running accuracy
    for i = 1:length(trials)
        correctSoFar = sum(strcmp({trials(1:i).type}, 'correct'));
        runningAccuracy(i) = (correctSoFar / i) * 100;
    end
    
    figure('Position', [200, 200, 1000, 600]);
    
    % Plot accuracy over time
    subplot(2,1,1);
    plot(trialNumbers, runningAccuracy, 'b-', 'LineWidth', 2);
    ylabel('Cumulative Accuracy (%)');
    xlabel('Trial Number');
    title(sprintf('%s - Performance Over Time (Interval: %ds, Reward Window: %ds)', ...
          mouseName, interTrialInterval/1000, rewardWindow/1000));
    grid on;
    
    % Plot trial durations
    subplot(2,1,2);
    colors = containers.Map({'correct', 'early_error', 'abort'}, {'g', 'r', 'm'});
    
    for i = 1:length(trials)
        color = colors(trials(i).type);
        plot(i, trialDurations(i)/1000, 'o', 'Color', color, 'MarkerFaceColor', color, 'MarkerSize', 6);
        hold on;
    end
    
    ylabel('Trial Duration (seconds)');
    xlabel('Trial Number');
    legend({'Correct', 'Early Error', 'Abort'}, 'Location', 'best');
    grid on;
end

function save_results(resultsPath, trials, metrics, mouseName, interTrialInterval, rewardWindow)
% SAVE_RESULTS - Save analysis results to files
%
% Variables:
%   resultsPath - directory to save results
%   trials - analyzed trial data
%   metrics - performance metrics
%   mouseName - mouse identifier
%   interTrialInterval - trial interval duration
%   rewardWindow - reward window duration
%   resultsFile - path to results file
%
% Goal: Save trial data and metrics for further analysis

    % Save trial data
    trialData = struct2table(trials);
    writetable(trialData, fullfile(resultsPath, 'trial_data.csv'));
    
    % Save metrics
    metricsFile = fullfile(resultsPath, 'metrics.mat');
    save(metricsFile, 'metrics', 'mouseName', 'interTrialInterval', 'rewardWindow');
    
    % Save summary text file
    summaryFile = fullfile(resultsPath, 'summary.txt');
    fid = fopen(summaryFile, 'w');
    fprintf(fid, 'Interval Timing Task Analysis Summary\n');
    fprintf(fid, '=====================================\n\n');
    fprintf(fid, 'Mouse: %s\n', mouseName);
    fprintf(fid, 'Inter-trial Interval: %d ms (%.1f s)\n', interTrialInterval, interTrialInterval/1000);
    fprintf(fid, 'Reward Window: %d ms (%.1f s)\n\n', rewardWindow, rewardWindow/1000);
    
    fprintf(fid, 'Performance Metrics:\n');
    fprintf(fid, '-------------------\n');
    fprintf(fid, 'Total Trials: %d\n', metrics.totalTrials);
    fprintf(fid, 'Correct Trials: %d\n', metrics.correctTrials);
    fprintf(fid, 'Early Error Trials: %d\n', metrics.earlyErrorTrials);
    fprintf(fid, 'Abort Trials: %d\n', metrics.abortTrials);
    fprintf(fid, 'Accuracy: %.1f%%\n\n', metrics.accuracy);
    
    fprintf(fid, 'Trial Durations:\n');
    fprintf(fid, '----------------\n');
    fprintf(fid, 'Correct Trials - Mean: %.1f s, Std: %.1f s\n', ...
            metrics.correctDurationMean/1000, metrics.correctDurationStd/1000);
    fprintf(fid, 'Error Trials - Mean: %.1f s, Std: %.1f s\n', ...
            metrics.errorDurationMean/1000, metrics.errorDurationStd/1000);
    
    fclose(fid);
end

function display_summary(metrics, mouseName)
% DISPLAY_SUMMARY - Display analysis summary to console
%
% Variables:
%   metrics - performance metrics
%   mouseName - mouse identifier
%
% Goal: Print key results to console for immediate review

    fprintf('\n=== Analysis Summary for %s ===\n', mouseName);
    fprintf('Total Trials: %d\n', metrics.totalTrials);
    fprintf('Correct: %d (%.1f%%)\n', metrics.correctTrials, metrics.accuracy);
    fprintf('Early Errors: %d\n', metrics.earlyErrorTrials);
    fprintf('Aborts: %d\n', metrics.abortTrials);
    fprintf('Correct Trial Duration: %.1f ± %.1f s\n', ...
            metrics.correctDurationMean/1000, metrics.correctDurationStd/1000);
    fprintf('Error Trial Duration: %.1f ± %.1f s\n', ...
            metrics.errorDurationMean/1000, metrics.errorDurationStd/1000);
    fprintf('================================\n\n');
end
