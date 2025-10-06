%%
% Peri-Reach Criticality Analysis
% Loads results from criticality_compare.m and analyzes d2 criticality values
% around reach onset times for each brain area
%
% Variables:
%   results - loaded criticality analysis results
%   dataR - reach behavioral data
%   areas - brain areas to analyze
%   reachStartFrame - reach start times in frame units for each area
%   d2Windows - d2 criticality values in 3-second windows around each reach
%   meanD2PeriReach - mean d2 values across all reaches for each area

%% Load existing results if requested
slidingWindowSize = 1;% For d2, use a small window to try to optimize temporal resolution


%%
paths = get_paths;
resultsPathWin = fullfile(paths.dropPath, sprintf('criticality/criticality_compare_results_win%gs.mat', slidingWindowSize));

% Load criticality analysis results
fprintf('Loading criticality analysis results...\n');
% results = load(fullfile(paths.dropPath, 'criticality/criticality_compare_results.mat'));
results = load(resultsPathWin);
results = results.results;

% Extract areas and parameters
areas = results.areas;
areasToTest = 2:4;
optimalBinSize = results.reach.optimalBinSize;
d2Rea = results.reach.d2;
startS = results.reach.startS;

%% Load reach behavioral data
fprintf('Loading reach behavioral data...\n');
dataR = load(fullfile(paths.dropPath, 'reach_data/Copy_of_Y4_100623_Spiketimes_idchan_BEH.mat'));

reachStart = dataR.R(:,1); % In seconds
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

reachClass = dataR.BlockWithErrors(:,4);
corrIdx = ismember(reachClass, [1 2]);
reachStartCorr = reachStart(corrIdx);
reachStartErr = reachStart(~corrIdx);
% % Convert reach start times to frame units for each area
% reachStartFrame = cell(1, length(areas));
% for a = areasToTest
%     % Round reach to nearest frame in criticality analysis bin sizes
%     reachStartFrame{a} = round(reachStart ./ optimalBinSize(a));
% end

%% ==============================================     Peri-Reach Analysis     ==============================================

% Define window parameters
windowDurationSec = 40; % 3-second window around each reach
windowDurationFrames = cell(1, length(areas));
for a = areasToTest
    windowDurationFrames{a} = round(windowDurationSec / optimalBinSize(a));
end

% Initialize storage for peri-reach d2 values (separate correct vs error)
d2WindowsCorr = cell(1, length(areas));
d2WindowsErr = cell(1, length(areas));
meanD2PeriReachCorr = cell(1, length(areas));
meanD2PeriReachErr = cell(1, length(areas));
timeAxisPeriReach = cell(1, length(areas));

fprintf('\n=== Peri-Reach d2 Criticality Analysis ===\n');

for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});

    % Get d2 values and time points for this area
    d2Values = d2Rea{a};
    timePoints = startS{a};
    % timePoints = startS{a} - results.reach.d2WindowSize(a)/2; % Adjust the times so the leading edge of the analyzed window aligns with reach onset

    % Initialize arrays for this area
    numReachesCorr = length(reachStartCorr);
    numReachesErr = length(reachStartErr);
    halfWindow = floor(windowDurationFrames{a} / 2);

    % Create time axis for peri-reach window (centered on reach onset)
    timeAxisPeriReach{a} = (-halfWindow:halfWindow) * optimalBinSize(a);

    % Initialize storage for all reach windows
    d2WindowsCorr{a} = nan(numReachesCorr, windowDurationFrames{a} + 1);
    d2WindowsErr{a} = nan(numReachesErr, windowDurationFrames{a} + 1);

    % Extract d2 values around each correct reach
    validCorr = 0;
    for r = 1:numReachesCorr
        reachTime = reachStartCorr(r)/1000; % adjust if needed based on units
        [~, closestIdx] = min(abs(timePoints - reachTime));
        winStart = closestIdx - halfWindow;
        winEnd = closestIdx + halfWindow;
        if winStart >= 1 && winEnd <= length(d2Values)
            d2WindowsCorr{a}(r, :) = d2Values(winStart:winEnd);
            validCorr = validCorr + 1;
        else
            d2WindowsCorr{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
        end
    end

    % Extract d2 values around each error reach
    validErr = 0;
    for r = 1:numReachesErr
        reachTime = reachStartErr(r)/1000; % adjust if needed based on units
        [~, closestIdx] = min(abs(timePoints - reachTime));
        winStart = closestIdx - halfWindow;
        winEnd = closestIdx + halfWindow;
        if winStart >= 1 && winEnd <= length(d2Values)
            d2WindowsErr{a}(r, :) = d2Values(winStart:winEnd);
            validErr = validErr + 1;
        else
            d2WindowsErr{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
        end
    end

    % Calculate mean d2 values across all valid reaches
    meanD2PeriReachCorr{a} = nanmean(d2WindowsCorr{a}, 1);
    meanD2PeriReachErr{a} = nanmean(d2WindowsErr{a}, 1);

    fprintf('Area %s: %d correct, %d error valid reaches\n', areas{a}, validCorr, validErr);
end

 %% ==============================================     Plotting Results     ==============================================

% Toggle plotting error reaches (single traces and mean)
plotErrors = true;

% Create peri-reach plots for each area
figure(300 + slidingWindowSize); clf;
set(gcf, 'Position', [100, 100, 1200, 800]);

% Use tight_subplot for layout
ha = tight_subplot(2, 2, [0.1 0.05], [0.08 0.1], [0.1 0.05]);

colors = {'k', 'b', 'r', [0 0.75 0]};


% Compute global y-limits across areas (use mean traces; include errors if enabled)
allMeanVals = [];
for a = areasToTest
    if ~isempty(meanD2PeriReachCorr{a})
        allMeanVals = [allMeanVals; meanD2PeriReachCorr{a}(:)]; %#ok<AGROW>
    end
    if plotErrors && ~isempty(meanD2PeriReachErr{a})
        allMeanVals = [allMeanVals; meanD2PeriReachErr{a}(:)]; %#ok<AGROW>
    end
end
globalYMin = nanmin(allMeanVals);
globalYMax = nanmax(allMeanVals);
if isempty(globalYMin) || isempty(globalYMax) || isnan(globalYMin) || isnan(globalYMax)
    globalYMin = 0; globalYMax = 1;
end
% Add small padding
pad = 0.05 * (globalYMax - globalYMin + eps);
yLimCommon = [globalYMin - pad, globalYMax + pad];

for a = areasToTest
    axes(ha(a));
    hold on;

    % % Plot individual reach windows (all single traces in light gray)
    % for r = 1:size(d2WindowsCorr{a}, 1)
    %     if ~all(isnan(d2WindowsCorr{a}(r, :)))
    %         plot(timeAxisPeriReach{a}, d2WindowsCorr{a}(r, :), 'Color', [.7 .7 .7], 'LineWidth', 0.5);
    %     end
    % end
    % if plotErrors
    %     for r = 1:size(d2WindowsErr{a}, 1)
    %         if ~all(isnan(d2WindowsErr{a}(r, :)))
    %             plot(timeAxisPeriReach{a}, d2WindowsErr{a}(r, :), 'Color', [.7 .7 .7], 'LineWidth', 0.5);
    %         end
    %     end
    % end

    % Calculate SEM for mean traces
    semCorr = nanstd(d2WindowsCorr{a}, 0, 1) / sqrt(sum(~all(isnan(d2WindowsCorr{a}), 2)));
    if plotErrors
        semErr = nanstd(d2WindowsErr{a}, 0, 1) / sqrt(sum(~all(isnan(d2WindowsErr{a}), 2)));
    end
    
    % Plot SEM ribbons first (behind the mean lines)
    fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
         [meanD2PeriReachCorr{a} + semCorr, fliplr(meanD2PeriReachCorr{a} - semCorr)], ...
         colors{a}, 'FaceAlpha', 0.4, 'EdgeColor', 'none');
    
    if plotErrors
        fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
             [meanD2PeriReachErr{a} + semErr, fliplr(meanD2PeriReachErr{a} - semErr)], ...
             colors{a}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
    
    % Plot mean d2 values: correct (solid area color), errors (dashed area color)
    hMeanCorr = plot(timeAxisPeriReach{a}, meanD2PeriReachCorr{a}, 'Color', colors{a}, 'LineWidth', 3, 'LineStyle', '-');
    if plotErrors
        hMeanErr = plot(timeAxisPeriReach{a}, meanD2PeriReachErr{a}, 'Color', colors{a}, 'LineWidth', 3, 'LineStyle', '--');
    else
        hMeanErr = []; %#ok<NASGU>
    end

    % Add vertical line at reach onset
    plot([0 0], ylim, 'k--', 'LineWidth', 2);

    % Formatting
    xlabel('Time relative to reach onset (s)', 'FontSize', 12);
    ylabel('d2 Criticality', 'FontSize', 12);
    title(sprintf('%s - Peri-Reach d2 Criticality (Window: %gs)', areas{a}, slidingWindowSize), 'FontSize', 14);
    grid on;

    % Set x-axis ticks and limits in seconds
    xMin = min(timeAxisPeriReach{a});
    xMax = max(timeAxisPeriReach{a});
    xlim([xMin xMax]);
    xTicks = ceil(xMin):floor(xMax);
    if isempty(xTicks)
        % Fallback to 5 ticks across range
        xTicks = linspace(xMin, xMax, 5);
    end
    xticks(xTicks);
    xticklabels(string(xTicks));

    % Set consistent y-axis limits and ticks across all subplots
    ylim(yLimCommon);
    yTicks = linspace(yLimCommon(1), yLimCommon(2), 5);
    yticks(yTicks);
    yticklabels(string(round(yTicks, 3)));

    % Add legend for mean traces
    if plotErrors
        legend([hMeanCorr hMeanErr], {'Mean d2 (Correct)', 'Mean d2 (Error)'}, 'Location', 'best', 'FontSize', 10);
    else
        legend(hMeanCorr, 'Mean d2 (Correct)', 'Location', 'best', 'FontSize', 10);
    end
end

sgtitle(sprintf('Peri-Reach d2 Criticality Analysis (Sliding Window: %gs)', slidingWindowSize), 'FontSize', 16);

% Save plot
filename = fullfile(paths.dropPath, sprintf('criticality/peri_reach_d2_criticality_win%gs.png', slidingWindowSize));
exportgraphics(gcf, filename, 'Resolution', 300);
fprintf('Saved peri-reach plot to: %s\n', filename);

%% ==============================================     Summary Statistics     ==============================================

fprintf('\n=== Peri-Reach Analysis Summary ===\n');
for a = areasToTest
    fprintf('\nArea %s:\n', areas{a});
    fprintf('  Total reaches: %d\n', length(reachStartFrame{a}));
    fprintf('  Valid reaches: %d\n', sum(~all(isnan(d2Windows{a}), 2)));
    fprintf('  Mean d2 at reach onset: %.4f\n', meanD2PeriReach{a}(halfWindow + 1));
    fprintf('  Mean d2 pre-reach (-1s): %.4f\n', meanD2PeriReach{a}(halfWindow - round(1/optimalBinSize(a)) + 1));
    fprintf('  Mean d2 post-reach (+1s): %.4f\n', meanD2PeriReach{a}(halfWindow + round(1/optimalBinSize(a)) + 1));
end

fprintf('\nPeri-reach analysis complete!\n');
