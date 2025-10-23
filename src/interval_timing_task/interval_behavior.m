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
    logFile = 'log.txt';  % Use original TXT file
    
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
% PARSE_LOG_DATA - Parse CSV log file into structured data with mixed types
%
% Variables:
%   logFilePath - path to CSV log file
%   data - table containing all log data
%   stateIdx - logical index for state events
%   eventIdx - logical index for regular events
%   events - struct array for events (L, R, B)
%   states - struct array for states (S)
%
% Goal: Extract all events and states from CSV log file for analysis

    fprintf('Reading TXT log file...\n');
    
    % Check if CSV file exists
    if ~exist(logFilePath, 'file')
        error('CSV log file not found: %s', logFilePath);
    end
    
    % Read CSV file line by line to preserve mixed data types
    fid = fopen(logFilePath, 'r');
    if fid == -1
        error('Could not open TXT file: %s', logFilePath);
    end
    
    % Read all lines
    rawLines = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    rawLines = rawLines{1};
    
    % Parse each line manually to preserve data types
    nLines = length(rawLines);
    timestamps = zeros(nLines, 1);
    types = cell(nLines, 1);
    values = cell(nLines, 1);
    
    for i = 1:nLines
        line = rawLines{i};
        parts = strsplit(line, ' ');
        if length(parts) >= 3
            timestamps(i) = str2double(parts{1});
            types{i} = parts{2};
            % Join remaining parts for state descriptions
            values{i} = strjoin(parts(3:end), ' ');
        end
    end
    
    % Remove empty entries
    validIdx = ~cellfun(@isempty, types);
    timestamps = timestamps(validIdx);
    types = types(validIdx);
    values = values(validIdx);
    
    % Create data table with proper types
    data = table(timestamps, types, values, 'VariableNames', {'timestamp', 'type', 'value'});
    
    fprintf('Loaded %d rows from TXT\n', length(timestamps));
    
    % Separate state events from regular events
    stateIdx = strcmp(data.type, 'S');
    eventIdx = ~stateIdx;
    
    % Create events structure (L, R, B events)
    eventTimestamps = timestamps(eventIdx);
    eventTypes = types(eventIdx);
    eventValues = values(eventIdx);
    nEvents = length(eventTimestamps);
    
    events = struct('timestamp', cell(nEvents, 1), 'type', cell(nEvents, 1), 'value', cell(nEvents, 1));
    
    for i = 1:nEvents
        events(i).timestamp = eventTimestamps(i);
        events(i).type = eventTypes{i};
        
        % Try to convert to number, if fails keep as string
        valStr = eventValues{i};
        valNum = str2double(valStr);
        if ~isnan(valNum)
            events(i).value = valNum;
        else
            events(i).value = valStr;
        end
    end
    
    % Create states structure (S events)
    stateTimestamps = timestamps(stateIdx);
    stateValues = values(stateIdx);
    nStates = length(stateTimestamps);
    
    states = struct('timestamp', cell(nStates, 1), 'state', cell(nStates, 1));
    
    for i = 1:nStates
        states(i).timestamp = stateTimestamps(i);
        states(i).state = stateValues{i};
    end
    
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
%   itiStartTimes - timestamps when ITI starts (trial starts)
%   trialArmTimes - timestamps when trials are armed (reward window starts)
%   eventTimestamps - all event timestamps
%   eventTypes - all event types
%   eventValues - all event values
%
% Goal: Identify trial boundaries and classify trial outcomes correctly

    fprintf('Analyzing trials...\n');
    
    % Extract arrays for faster processing
    eventTimestamps = [events.timestamp];
    eventTypes = {events.type};
    eventValues = [events.value];
    
    % Find ITI_START times (trial starts)
    itiStartIdx = strcmp({states.state}, 'ITI_START');
    itiStartTimes = [states(itiStartIdx).timestamp];
    
    % Find TRIAL_ARMED times (reward window starts)
    trialArmIdx = strcmp({states.state}, 'TRIAL_ARMED');
    trialArmTimes = [states(trialArmIdx).timestamp];
    
    fprintf('Found %d ITI starts and %d trial arms\n', length(itiStartTimes), length(trialArmTimes));
    
    % Each trial starts with ITI_START and ends with the next ITI_START
    nTrials = length(itiStartTimes) - 1; % Last ITI_START has no corresponding end
    trials = struct('startTime', cell(nTrials, 1), 'endTime', cell(nTrials, 1), ...
                   'type', cell(nTrials, 1), 'duration', cell(nTrials, 1), ...
                   'rewardWindowStart', cell(nTrials, 1), 'beamBreakTime', cell(nTrials, 1));
    
    % Process each trial
    for i = 1:nTrials
        trialStartTime = itiStartTimes(i);
        trialEndTime = itiStartTimes(i + 1);
        
        % Find TRIAL_ARMED within this trial period
        trialArmIdx = find(trialArmTimes > trialStartTime & trialArmTimes < trialEndTime, 1);
        
        if isempty(trialArmIdx)
            % No TRIAL_ARMED found - this shouldn't happen in normal operation
            continue;
        end
        
        rewardWindowStart = trialArmTimes(trialArmIdx);
        
        % Find events within this trial period
        trialEventIdx = eventTimestamps >= trialStartTime & eventTimestamps < trialEndTime;
        trialTimestamps = eventTimestamps(trialEventIdx);
        trialTypes = eventTypes(trialEventIdx);
        trialValues = eventValues(trialEventIdx);
        
        % Check for beam breaks during reward window (correct trials)
        rewardWindowIdx = trialTimestamps >= rewardWindowStart;
        beamBreakIdx = find(rewardWindowIdx & strcmp(trialTypes, 'B') & trialValues == 1, 1);
        
        if ~isempty(beamBreakIdx)
            % Beam break during reward window - CORRECT trial
            beamBreakTime = trialTimestamps(beamBreakIdx);
            trialType = 'correct';
        else
            % No beam break during reward window - check for early errors
            beamBreakTime = NaN;
            
            % Check for beam break before reward window (early error)
            earlyIdx = trialTimestamps < rewardWindowStart;
            earlyBeamIdx = find(earlyIdx & strcmp(trialTypes, 'B') & trialValues == 1, 1);
            
            if ~isempty(earlyBeamIdx)
                trialType = 'early_error';
            else
                % No beam break at all - ABORT trial
                trialType = 'abort';
            end
        end
        
        % Store trial data
        trials(i).startTime = trialStartTime;
        trials(i).endTime = trialEndTime;
        trials(i).type = trialType;
        trials(i).duration = trialEndTime - trialStartTime;
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
        % Return empty metrics with default values
        metrics = struct('totalTrials', 0, 'correctTrials', 0, 'earlyErrorTrials', 0, ...
                       'abortTrials', 0, 'accuracy', 0, 'correctDurationMean', 0, ...
                       'correctDurationStd', 0, 'errorDurationMean', 0, 'errorDurationStd', 0);
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
    
    % Handle cases with no trials of certain types
    if ~isempty(correctDurations)
        metrics.correctDurationMean = mean(correctDurations);
        metrics.correctDurationStd = std(correctDurations);
    else
        metrics.correctDurationMean = 0;
        metrics.correctDurationStd = 0;
    end
    
    if ~isempty(errorDurations)
        metrics.errorDurationMean = mean(errorDurations);
        metrics.errorDurationStd = std(errorDurations);
    else
        metrics.errorDurationMean = 0;
        metrics.errorDurationStd = 0;
    end
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
