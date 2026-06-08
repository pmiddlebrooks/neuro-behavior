function logTable = interval_session_performance(subjectName, sessionName, sessionInterval, nParts)
% INTERVAL_SESSION_PERFORMANCE - Plot interval task performance for one session
%
% Variables:
%   subjectName - Subject folder under interval_timing_task/data (e.g. 'ey9387')
%   sessionName - Session folder under subject (e.g. 'ey9387_2026_05_21')
%   sessionInterval - Target interval duration (sec); default 5 if omitted
%   nParts - Number of equal sequential session windows for part-wise histograms; default 3
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
    if nargin < 4 || isempty(nParts)
        nParts = 3;
    end
    minLeaveSec = 0.1;         % sec; MIN_LEAVE_TIME in Arduino sketch
    correctTimeMaxSec = 20;    % upper bound for correct poke-time histogram (sec)
    movAvgWinSec = 60;         % moving-average window for rewards/min and trials/min (sec)
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
        sessionInterval, correctTimeMaxSec, movAvgWinSec, rateBinSec, accuracyBeforeSec);
    plot_interval_session_part_histograms(trials, subjectName, sessionName, sessionInterval, ...
        correctTimeMaxSec, nParts);
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

    sessionOriginMs = min(logTable.timestampMs);
    trialTypes = strings(0, 1);
    pokeTimesSec = [];
    outcomeTimesSec = [];

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
                outcomeTimesSec(end + 1, 1) = (eventTimeMs - sessionOriginMs) / 1000; %#ok<AGROW>
            end
            timerArmed = false;
            leaveTimeMs = NaN;
        elseif eventName == "REWARD"
            if ~firstRewardSeen
                firstRewardSeen = true;
            elseif timerArmed && ~isnan(leaveTimeMs)
                trialTypes(end + 1, 1) = "correct"; %#ok<AGROW>
                pokeTimesSec(end + 1, 1) = (eventTimeMs - leaveTimeMs) / 1000; %#ok<AGROW>
                outcomeTimesSec(end + 1, 1) = (eventTimeMs - sessionOriginMs) / 1000; %#ok<AGROW>
            end
            timerArmed = false;
            leaveTimeMs = NaN;
        end
    end

    trials = table(trialTypes, pokeTimesSec, outcomeTimesSec, ...
        'VariableNames', {'type', 'pokeTimeSec', 'outcomeTimeSec'});
end

function fig = plot_interval_session_performance(logTable, trials, subjectName, sessionName, ...
        sessionInterval, correctTimeMaxSec, movAvgWinSec, rateBinSec, accuracyBeforeSec)
% PLOT_INTERVAL_SESSION_PERFORMANCE - Build session performance figure
%
% Variables:
%   logTable, trials - Parsed log and trial table
%   subjectName, sessionName - Session identifiers for title
%   sessionInterval - Target interval (sec); vertical reference line
%   correctTimeMaxSec - Upper x-limit for correct-time histogram
%   movAvgWinSec - Moving-average window for rate plots (sec)
%   rateBinSec - Bin width for session time-series (sec)
%   accuracyBeforeSec - Exclude errors with poke time earlier than interval minus this (sec)
%
% Goal: Four-panel figure of reward rate, trial completion rate, trial outcomes, and distribution

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

    accuracyLowSec = sessionInterval - accuracyBeforeSec;
    quickErrorMask = trials.type == "error" & trials.pokeTimeSec < accuracyLowSec;
    completionMask = ~quickErrorMask;
    completionTimesSec = trials.outcomeTimeSec(completionMask);
    trialCounts = histcounts(completionTimesSec, binEdges);
    trialsPerMin = trialCounts * (60 / rateBinSec);
    trialsPerMinSmooth = movmean(trialsPerMin, smoothWinBins);

    plotErrorMinSec = 1;   % exclude fast errors from trial and histogram plots
    errorMask = trials.type == "error";
    correctMask = trials.type == "correct";
    errorPlotMask = errorMask & trials.pokeTimeSec >= plotErrorMinSec;
    errorTimesSec = trials.pokeTimeSec(errorPlotMask);
    correctTimesSec = trials.pokeTimeSec(correctMask);
    trialNumbers = (1:height(trials))';

    fig = figure('Name', sprintf('%s %s interval performance', subjectName, sessionName), ...
        'Units', 'pixels');
    layout = tiledlayout(fig, 4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    axReward = nexttile(layout);
    plot(axReward, timeBinsSec, rewardsPerMinSmooth, 'k-', 'LineWidth', 1.2);
    xlabel(axReward, 'Session time (s)');
    ylabel(axReward, 'Rewards/min (moving avg)');
    title(axReward, sprintf('%s | %s', subjectName, sessionName), 'interpreter', 'none');
    grid(axReward, 'on');
    xlim(axReward, [0, sessionEndSec]);

    axCompletion = nexttile(layout);
    plot(axCompletion, timeBinsSec, trialsPerMinSmooth, 'k-', 'LineWidth', 1.2);
    xlabel(axCompletion, 'Session time (s)');
    ylabel(axCompletion, 'Trials/min (moving avg)');
    title(axCompletion, sprintf('Trial completion (excludes errors < %.1f s; keeps late corrects)', ...
        accuracyLowSec));
    grid(axCompletion, 'on');
    xlim(axCompletion, [0, sessionEndSec]);

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

    distBinWidthSec = 0.2;
    axHist = nexttile(layout);
    histYMax = plot_poke_time_histogram_ax(axHist, errorTimesSec, correctTimesSec, ...
        sessionInterval, correctTimeMaxSec, distBinWidthSec);
    xlabel(axHist, 'Poke time since leave (s)');
    ylabel(axHist, 'Count');
    ylim(axHist, [0, histYMax * 1.05]);
    grid(axHist, 'on');
    legend(axHist, 'Location', 'best');

    legend(axTrials, {'Error', 'Correct', sprintf('%g s interval', sessionInterval)}, ...
        'Location', 'best');
    fit_figure_on_screen(fig, 1100, 950);
end

function fig = plot_interval_session_part_histograms(trials, subjectName, sessionName, ...
        sessionInterval, correctTimeMaxSec, nParts)
% PLOT_INTERVAL_SESSION_PART_HISTOGRAMS - Poke-time distributions by session segment
%
% Variables:
%   trials - Trial table with type, pokeTimeSec, outcomeTimeSec
%   subjectName, sessionName - Session identifiers
%   sessionInterval - Target interval (sec); vertical reference line
%   correctTimeMaxSec - Upper x-limit for histograms (sec)
%   nParts - Number of equal sequential session time windows
%
% Goal: One-column figure with poke-time histograms per session part (shared y-limits)

    plotErrorMinSec = 1;
    distBinWidthSec = 0.2;
    if isempty(trials)
        sessionEndSec = 1;
    else
        sessionEndSec = max(trials.outcomeTimeSec);
    end
    if sessionEndSec <= 0
        sessionEndSec = 1;
    end
    partEdgesSec = linspace(0, sessionEndSec, nParts + 1);

    fig = figure('Name', sprintf('%s %s interval by part', subjectName, sessionName), ...
        'Units', 'pixels');
    layout = tiledlayout(fig, nParts, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(layout, sprintf('%s | %s — poke time by session part (n=%d)', ...
        subjectName, sessionName, nParts), 'interpreter', 'none');

    partAxes = gobjects(nParts, 1);
    histYMaxAll = 1;

    for partIdx = 1:nParts
        if partIdx < nParts
            partMask = trials.outcomeTimeSec >= partEdgesSec(partIdx) & ...
                trials.outcomeTimeSec < partEdgesSec(partIdx + 1);
        else
            partMask = trials.outcomeTimeSec >= partEdgesSec(partIdx) & ...
                trials.outcomeTimeSec <= partEdgesSec(partIdx + 1);
        end

        partTrials = trials(partMask, :);
        errorMask = partTrials.type == "error";
        correctMask = partTrials.type == "correct";
        errorPlotMask = errorMask & partTrials.pokeTimeSec >= plotErrorMinSec;
        errorTimesSec = partTrials.pokeTimeSec(errorPlotMask);
        correctTimesSec = partTrials.pokeTimeSec(correctMask);

        partAxes(partIdx) = nexttile(layout);
        histYMax = plot_poke_time_histogram_ax(partAxes(partIdx), errorTimesSec, correctTimesSec, ...
            sessionInterval, correctTimeMaxSec, distBinWidthSec);
        histYMaxAll = max(histYMaxAll, histYMax);

        partStartMin = partEdgesSec(partIdx) / 60;
        partEndMin = partEdgesSec(partIdx + 1) / 60;
        title(partAxes(partIdx), sprintf('Part %d (%.1f–%.1f min, %d trials)', ...
            partIdx, partStartMin, partEndMin, height(partTrials)));
        ylabel(partAxes(partIdx), 'Count');
        grid(partAxes(partIdx), 'on');

        if partIdx == nParts
            xlabel(partAxes(partIdx), 'Poke time since leave (s)');
        end
    end

    for partIdx = 1:nParts
        ylim(partAxes(partIdx), [0, histYMaxAll * 1.05]);
    end

    legend(partAxes(1), 'Location', 'best');
    fit_figure_on_screen(fig, 700, 280 * nParts);
end

function fit_figure_on_screen(fig, prefWidth, prefHeight)
% FIT_FIGURE_ON_SCREEN - Size and position a figure fully within the monitor
%
% Variables:
%   fig - Figure handle
%   prefWidth, prefHeight - Preferred inner size (pixels)
%
% Goal: Keep the title bar and menu bar on screen; shrink the figure if needed

    screenSize = get(0, 'ScreenSize');
    margin = 48;
    maxOuterWidth = screenSize(3) - 2 * margin;
    maxOuterHeight = screenSize(4) - 2 * margin;

    set(fig, 'Units', 'pixels');
    set(fig, 'Position', [screenSize(1) + margin, screenSize(2) + margin, prefWidth, prefHeight]);
    drawnow;

    outerPos = get(fig, 'OuterPosition');
    scale = min([1, maxOuterWidth / outerPos(3), maxOuterHeight / outerPos(4)]);
    if scale < 1
        innerPos = get(fig, 'Position');
        newWidth = max(300, round(innerPos(3) * scale));
        newHeight = max(300, round(innerPos(4) * scale));
        set(fig, 'Position', [innerPos(1), innerPos(2), newWidth, newHeight]);
        drawnow;
    end

    movegui(fig, 'onscreen');
end

function histYMax = plot_poke_time_histogram_ax(ax, errorTimesSec, correctTimesSec, ...
        sessionInterval, correctTimeMaxSec, distBinWidthSec)
% PLOT_POKE_TIME_HISTOGRAM_AX - Combined error/correct poke-time histogram on one axes
%
% Variables:
%   ax - Target axes
%   errorTimesSec, correctTimesSec - Poke times since leave (sec)
%   sessionInterval - Target interval; errors plotted below, corrects at/above
%   correctTimeMaxSec, distBinWidthSec - Histogram range and bin width
%
% Goal: Red error bars and green correct bars with interval reference line

    histBinEdges = 0:distBinWidthSec:correctTimeMaxSec;
    binCenters = histBinEdges(1:end-1) + distBinWidthSec / 2;
    errorCounts = histcounts(errorTimesSec, histBinEdges);
    correctCounts = histcounts(correctTimesSec, histBinEdges);
    errorBarMask = binCenters < sessionInterval;
    correctBarMask = binCenters >= sessionInterval;

    hold(ax, 'on');
    bar(ax, binCenters(errorBarMask), errorCounts(errorBarMask), 1, ...
        'FaceColor', [0.85, 0.2, 0.2], 'EdgeColor', 'none', 'DisplayName', 'Error');
    bar(ax, binCenters(correctBarMask), correctCounts(correctBarMask), 1, ...
        'FaceColor', [0.2, 0.65, 0.25], 'EdgeColor', 'none', 'DisplayName', 'Correct');
    xline(ax, sessionInterval, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    xlim(ax, [0, correctTimeMaxSec]);
    histYMax = max([errorCounts, correctCounts, 1]);
    hold(ax, 'off');
end
