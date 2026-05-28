function logTable = interval_session_performance(subjectName, sessionName, sessionInterval)
% INTERVAL_SESSION_PERFORMANCE - Plot interval task performance for one session
%
% Variables:
%   subjectName - Subject folder under interval_timing_task/data (e.g. 'ey9387')
%   sessionName - Session folder under subject (e.g. 'ey9387_2026_05_21')
%   sessionInterval - Target interval duration (sec); default 5 if omitted
%
% Goal: Load revised interval CSV logs, extract trial outcomes (ERROR / REWARD),
% print accuracy (excluding fast errors and late corrects), count pokes near the
% target interval, and plot session performance.
% Narrowed accuracy uses trials with poke time in [sessionInterval - 2, sessionInterval + 3] s.
%
% Returns:
%   logTable - Parsed event table (timestampMs, event, value)

    if nargin < 3 || isempty(sessionInterval)
        sessionInterval = 5;   % sec; default matches INTERVAL_DURATION in Revised_Interval_Timing_Code_Processing.ino
    end
    minLeaveSec = 0.1;         % sec; MIN_LEAVE_TIME in Arduino sketch
    correctTimeMaxSec = 20;    % upper bound for correct poke-time histogram (sec)
    zoomHalfWidthSec = 3;      % half-width of zoomed trial-wise panel (sec)
    movAvgWinSec = 60;         % moving-average window for rewards/min (sec)
    rateBinSec = 1;            % time bin width for reward-rate estimation (sec)
    accuracyBeforeSec = 3;     % sec before sessionInterval for narrowed accuracy window
    accuracyAfterSec = 3;      % sec after sessionInterval for narrowed accuracy window
    rewardAttemptBeforeSec = 1;  % sec before sessionInterval for reward-seeking window
    rewardAttemptAfterSec = 2;   % sec after sessionInterval for reward-seeking window

    addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
    paths = get_paths();

    sessionDir = fullfile(paths.intervalDataPath, subjectName, sessionName);
    if ~exist(sessionDir, 'dir')
        error('interval_session_performance:SessionNotFound', ...
            'Session directory not found: %s', sessionDir);
    end

    csvPath = find_interval_csv(sessionDir);
    fprintf('Loading interval log: %s\n', csvPath);
    logTable = parse_interval_log(csvPath);
    sessionDurationMin = (max(logTable.timestampMs) - min(logTable.timestampMs)) / 1000 / 60;
    fprintf('Session duration: %.1f min\n', sessionDurationMin);

    trials = extract_interval_trials(logTable, minLeaveSec);
    nError = sum(trials.type == "error");
    nCorrect = sum(trials.type == "correct");
    fprintf('%d trials (%d error, %d correct)\n', height(trials), nError, nCorrect);

    errorMask = trials.type == "error";
    correctMask = trials.type == "correct";
    accuracyLowSec = sessionInterval - accuracyBeforeSec;
    accuracyHighSec = sessionInterval + accuracyAfterSec;
    accuracyMask = trials.pokeTimeSec >= accuracyLowSec & trials.pokeTimeSec <= accuracyHighSec;
    nAccuracyTrials = sum(accuracyMask);
    nCorrectForAccuracy = sum(correctMask & accuracyMask);
    if nAccuracyTrials > 0
        accuracyPct = 100 * nCorrectForAccuracy / nAccuracyTrials;
    else
        accuracyPct = NaN;
    end
    fprintf(['Accuracy: %.1f%% (%d/%d trials; poke window %.1f–%.1f s ', ...
        '(interval %.0f s, excludes < %.1f s before and > %.1f s after))\n'], ...
        accuracyPct, nCorrectForAccuracy, nAccuracyTrials, accuracyLowSec, accuracyHighSec, ...
        sessionInterval, accuracyBeforeSec, accuracyAfterSec);
    if nAccuracyTrials > 0
        medianAccuracyIntervalSec = median(trials.pokeTimeSec(accuracyMask));
        fprintf('Median poke time (narrowed accuracy trials): %.2f s\n', medianAccuracyIntervalSec);
    end

    rewardAttemptMask = trials.pokeTimeSec >= sessionInterval - rewardAttemptBeforeSec & ...
        trials.pokeTimeSec <= sessionInterval + rewardAttemptAfterSec;
    nRewardAttempt = sum(rewardAttemptMask);
    nRewardAttemptCorrect = sum(correctMask & rewardAttemptMask);
    nRewardAttemptError = sum(errorMask & rewardAttemptMask);
    fprintf(['Trials with poke %.1f–%.1f s since leave (around %.0f s interval): ', ...
        '%d total (%d correct [solenoid], %d error)\n'], ...
        sessionInterval - rewardAttemptBeforeSec, sessionInterval + rewardAttemptAfterSec, ...
        sessionInterval, nRewardAttempt, nRewardAttemptCorrect, nRewardAttemptError);

    plot_interval_session_performance(logTable, trials, subjectName, sessionName, ...
        sessionInterval, correctTimeMaxSec, zoomHalfWidthSec, movAvgWinSec, rateBinSec);
end

function csvPath = find_interval_csv(sessionDir)
% FIND_INTERVAL_CSV - Return path to the revised interval CSV in a session folder
%
% Variables:
%   sessionDir - Full path to session data directory
%
% Goal: Select the most recent revised_interval_*.csv log file

    csvFiles = dir(fullfile(sessionDir, 'revised_interval_*.csv'));
    if isempty(csvFiles)
        error('interval_session_performance:NoCsv', ...
            'No revised_interval_*.csv found in %s', sessionDir);
    end
    [~, newestIdx] = max([csvFiles.datenum]);
    csvPath = fullfile(sessionDir, csvFiles(newestIdx).name);
end

function logTable = parse_interval_log(csvPath)
% PARSE_INTERVAL_LOG - Read Arduino/Processing interval task CSV
%
% Variables:
%   csvPath - Path to revised_interval_*.csv
%
% Goal: Return sorted table with timestamp_ms, event, and value columns

    rawTable = readtable(csvPath, 'TextType', 'string');
    varNames = lower(string(rawTable.Properties.VariableNames));

    timeCol = find(contains(varNames, 'timestamp'), 1);
    eventCol = find(strcmp(varNames, 'event') | contains(varNames, 'event'), 1);
    valueCol = find(strcmp(varNames, 'value') | contains(varNames, 'value'), 1);

    if isempty(timeCol) || isempty(eventCol) || isempty(valueCol)
        error('interval_session_performance:BadCsv', ...
            'CSV must contain timestamp, event, and value columns: %s', csvPath);
    end

    timestampMs = rawTable{:, timeCol};
    if iscell(timestampMs)
        timestampMs = cellfun(@str2double, timestampMs);
    elseif isstring(timestampMs) || ischar(timestampMs)
        timestampMs = str2double(string(timestampMs));
    end
    timestampMs = double(timestampMs);

    eventNames = string(rawTable{:, eventCol});
    eventValues = rawTable{:, valueCol};
    if iscell(eventValues) || isstring(eventValues)
        eventValues = str2double(string(eventValues));
    end
    eventValues = double(eventValues);

    validRows = ~isnan(timestampMs) & eventNames ~= "" & ~ismissing(eventNames);
    logTable = table(timestampMs(validRows), eventNames(validRows), eventValues(validRows), ...
        'VariableNames', {'timestampMs', 'event', 'value'});
    logTable = sortrows(logTable, 'timestampMs');
end

function trials = extract_interval_trials(logTable, minLeaveSec)
% EXTRACT_INTERVAL_TRIALS - Segment ERROR/REWARD trials from event log
%
% Variables:
%   logTable - Parsed event table
%   minLeaveSec - Minimum confirmed leave duration (sec)
%
% Goal: For each ERROR or post-training REWARD, record poke time since last leave

    minLeaveMs = minLeaveSec * 1000;

    leavePending = false;
    leaveConfirmStartMs = NaN;
    initialExitMs = NaN;
    leaveTimeMs = NaN;
    timerArmed = false;
    beamState = 0;
    firstRewardSeen = false;

    trialTypes = strings(0, 1);
    pokeTimesSec = [];

    eventNames = logTable.event;
    timestampsMs = logTable.timestampMs;
    eventValues = logTable.value;
    nEvents = height(logTable);

    for eventIdx = 1:nEvents
        eventTimeMs = timestampsMs(eventIdx);
        eventName = eventNames(eventIdx);
        eventValue = eventValues(eventIdx);

        if leavePending && beamState == 0 && (eventTimeMs - leaveConfirmStartMs) >= minLeaveMs
            leaveTimeMs = initialExitMs;
            timerArmed = true;
            leavePending = false;
        end

        if eventName == "B"
            beamState = eventValue;
            if eventValue == 0
                leavePending = true;
                initialExitMs = eventTimeMs;
                leaveConfirmStartMs = eventTimeMs;
            else
                leavePending = false;
            end
        elseif eventName == "ERROR"
            if timerArmed && ~isnan(leaveTimeMs)
                trialTypes(end + 1, 1) = "error"; %#ok<AGROW>
                pokeTimesSec(end + 1, 1) = (eventTimeMs - leaveTimeMs) / 1000; %#ok<AGROW>
            end
            timerArmed = false;
            leaveTimeMs = NaN;
        elseif eventName == "REWARD"
            if ~firstRewardSeen
                firstRewardSeen = true;
            elseif timerArmed && ~isnan(leaveTimeMs)
                trialTypes(end + 1, 1) = "correct"; %#ok<AGROW>
                pokeTimesSec(end + 1, 1) = (eventTimeMs - leaveTimeMs) / 1000; %#ok<AGROW>
            end
            timerArmed = false;
            leaveTimeMs = NaN;
        end
    end

    trials = table(trialTypes, pokeTimesSec, 'VariableNames', {'type', 'pokeTimeSec'});
end

function fig = plot_interval_session_performance(logTable, trials, subjectName, sessionName, ...
        sessionInterval, correctTimeMaxSec, zoomHalfWidthSec, movAvgWinSec, rateBinSec)
% PLOT_INTERVAL_SESSION_PERFORMANCE - Build session performance figure
%
% Variables:
%   logTable, trials - Parsed log and trial table
%   subjectName, sessionName - Session identifiers for title
%   sessionInterval - Target interval (sec); vertical reference line
%   correctTimeMaxSec - Upper x-limit for correct-time histogram
%   zoomHalfWidthSec - Half-width of zoomed trial-wise panel
%   movAvgWinSec - Moving-average window for reward rate (sec)
%   rateBinSec - Bin width for reward-rate time series (sec)
%
% Goal: Four-panel figure of reward rate, trial outcomes, zoom, and combined distribution

    rewardTimesSec = logTable.timestampMs(logTable.event == "REWARD") / 1000;
    sessionEndSec = max(logTable.timestampMs) / 1000;
    timeBinsSec = (0:rateBinSec:sessionEndSec)';
    if numel(timeBinsSec) < 2
        timeBinsSec = [0; rateBinSec];
    end
    binEdges = [timeBinsSec; timeBinsSec(end) + rateBinSec];
    rewardCounts = histcounts(rewardTimesSec, binEdges);
    rewardsPerMin = rewardCounts * (60 / rateBinSec);
    smoothWinBins = max(1, round(movAvgWinSec / rateBinSec));
    rewardsPerMinSmooth = movmean(rewardsPerMin, smoothWinBins);

    plotErrorMinSec = 1;   % exclude fast errors from trial and histogram plots
    errorMask = trials.type == "error";
    correctMask = trials.type == "correct";
    errorPlotMask = errorMask & trials.pokeTimeSec >= plotErrorMinSec;
    errorTimesSec = trials.pokeTimeSec(errorPlotMask);
    correctTimesSec = trials.pokeTimeSec(correctMask);
    trialNumbers = (1:height(trials))';

    fig = figure('Name', sprintf('%s %s interval performance', subjectName, sessionName), ...
        'Position', [80, 80, 1100, 950]);
    layout = tiledlayout(fig, 4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    axReward = nexttile(layout);
    plot(axReward, timeBinsSec, rewardsPerMinSmooth, 'k-', 'LineWidth', 1.2);
    xlabel(axReward, 'Session time (s)');
    ylabel(axReward, 'Rewards/min (moving avg)');
    title(axReward, sprintf('%s | %s', subjectName, sessionName), 'interpreter', 'none');
    grid(axReward, 'on');
    xlim(axReward, [0, sessionEndSec]);

    axTrials = nexttile(layout);
    hold(axTrials, 'on');
    if any(errorPlotMask)
        plot(axTrials, errorTimesSec, trialNumbers(errorPlotMask), 'r.', 'MarkerSize', 15);
    end
    if any(correctMask)
        plot(axTrials, correctTimesSec, trialNumbers(correctMask), '.', ...
            'Color', [0, 0.7, 0], 'MarkerSize', 15);
    end
    xline(axTrials, sessionInterval, 'k--', 'LineWidth', 1.2);
    xlabel(axTrials, 'Poke time since leave (s)');
    ylabel(axTrials, 'Trial');
    xlim(axTrials, [0, correctTimeMaxSec]);
    ylim(axTrials, [0.5, max(height(trials) + 0.5, 1.5)]);
    grid(axTrials, 'on');
    hold(axTrials, 'off');

    axZoom = nexttile(layout);
    hold(axZoom, 'on');
    if any(errorPlotMask)
        plot(axZoom, errorTimesSec, trialNumbers(errorPlotMask), 'r.', 'MarkerSize', 15);
    end
    if any(correctMask)
        plot(axZoom, correctTimesSec, trialNumbers(correctMask), '.', ...
            'Color', [0, 0.7, 0], 'MarkerSize', 15);
    end
    xline(axZoom, sessionInterval, 'k--', 'LineWidth', 1.2);
    xlabel(axZoom, 'Poke time since leave (s)');
    ylabel(axZoom, 'Trial');
    xlim(axZoom, [sessionInterval - zoomHalfWidthSec, sessionInterval + zoomHalfWidthSec]);
    ylim(axZoom, [0.5, max(height(trials) + 0.5, 1.5)]);
    grid(axZoom, 'on');
    hold(axZoom, 'off');

    distBinWidthSec = 0.2;
    histBinEdges = 0:distBinWidthSec:correctTimeMaxSec;
    binCenters = histBinEdges(1:end-1) + distBinWidthSec / 2;
    errorCounts = histcounts(errorTimesSec, histBinEdges);
    correctCounts = histcounts(correctTimesSec, histBinEdges);
    errorBarMask = binCenters < sessionInterval;
    correctBarMask = binCenters >= sessionInterval;

    axHist = nexttile(layout);
    hold(axHist, 'on');
    bar(axHist, binCenters(errorBarMask), errorCounts(errorBarMask), 1, ...
        'FaceColor', [0.85, 0.2, 0.2], 'EdgeColor', 'none', 'DisplayName', 'Error');
    bar(axHist, binCenters(correctBarMask), correctCounts(correctBarMask), 1, ...
        'FaceColor', [0.2, 0.65, 0.25], 'EdgeColor', 'none', 'DisplayName', 'Correct');
    xline(axHist, sessionInterval, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    xlabel(axHist, 'Poke time since leave (s)');
    ylabel(axHist, 'Count');
    xlim(axHist, [0, correctTimeMaxSec]);
    histYMax = max([errorCounts, correctCounts, 1]);
    ylim(axHist, [0, histYMax * 1.05]);
    grid(axHist, 'on');
    hold(axHist, 'off');
    legend(axHist, 'Location', 'best');

    legend(axTrials, {'Error', 'Correct', sprintf('%g s interval', sessionInterval)}, ...
        'Location', 'best');
end
