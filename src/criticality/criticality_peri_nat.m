%%
% Peri-Behavior Criticality Analysis (Naturalistic Data)
% Loads results from criticality_compare.m and analyzes d2 criticality values
% around behavior onset times for naturalistic data
%
% Variables:
%   results - loaded criticality analysis results
%   bhvID - behavior labels vector (user provided)
%   binSizeBhv - bin size for behavior labels in seconds (user provided)
%   areas - brain areas to analyze
%   d2Windows - d2 criticality values around each behavior onset
%   meanD2PeriBhv - mean d2 values across all behaviors for each set

%% Load behavior 
opts = neuro_behavior_options;
opts.frameSize = 1/opts.fsBhv;
opts.frameSize = .02;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectEnd = 45 * 60; % seconds
getDataType = 'behavior';
get_standard_data

binSizeBhv = opts.frameSize;
%% Load existing results if requested
slidingWindowSize = 1;% For d2, use a small window to try to optimize temporal resolution

%%
paths = get_paths;
resultsPathWin = fullfile(paths.dropPath, sprintf('criticality/criticality_compare_results_win%gs.mat', slidingWindowSize));

% Load criticality analysis results
fprintf('Loading criticality analysis results...\n');
results = load(resultsPathWin);
results = results.results;

% Extract areas and parameters
areas = results.areas;
areasToTest = 2:4;
optimalBinSize = results.naturalistic.optimalBinSize;
d2Nat = results.naturalistic.d2;
startS = results.naturalistic.startS;


%% User Input: Behavior Labels and Selector
% User should have already loaded:
%   bhvID - vector of behavior labels at behavior bin resolution
%   binSizeBhv - bin size for bhvID in seconds

% Validate user inputs
if ~exist('bhvID', 'var') || ~exist('binSizeBhv', 'var')
    error('bhvID and binSizeBhv variables must be defined before running this script');
end

% Define behavior sets to analyze
% Each cell contains labels that belong to the same behavior set
behaviorSets = {
    [15],          % Set 1: behaviors 1, 2
    [5:8],          % Set 2: behaviors 3, 4  
    [9:10]           % Set 3: behaviors 5, 6
    [11:12]           % Set 3: behaviors 5, 6
};

behaviorSetNames = {
    'Loco',         % Name for Set 1
    'Head Groom',      % Name for Set 2
    'Body Groom'    % Name for Set 3
    'Itch'    % Name for Set 3
};

behaviorSetColors = {
    [0 0.6 0],       % Green for Set 1
    [0 0.8 0.8],     % Cyan for Set 2
    [0.8 0.2 0.8]    % Magenta for Set 3
    [1 0 0]    % Magenta for Set 3
};

fprintf('\nBehavior sets defined:\n');
for s = 1:length(behaviorSets)
    fprintf('  Set %d (%s): labels [%s]\n', s, behaviorSetNames{s}, ...
        mat2str(behaviorSets{s}));
end

%% ==============================================     Peri-Behavior Analysis     ==============================================

% Define window parameters
windowDurationSec = 40; % seconds around each behavior onset
windowDurationFrames = cell(1, length(areas));
for a = areasToTest
    windowDurationFrames{a} = ceil(windowDurationSec / optimalBinSize(a));
end

% Initialize storage for peri-behavior d2 values (separate by behavior set)
d2WindowsBhv = cell(length(behaviorSets), length(areas));
meanD2PeriBhv = cell(length(behaviorSets), length(areas));

fprintf('\n=== Peri-Behavior d2 Criticality Analysis ===\n');
fprintf('Window duration: %.1f seconds\n', windowDurationSec);
fprintf('Behavior bin size: %.3f seconds\n', binSizeBhv);

% Find behavior onsets for each behavior set (outside area loop)
bhvOnsetsBySet = cell(1, length(behaviorSets));
bhvOnsetTimesBySet = cell(1, length(behaviorSets));

for s = 1:length(behaviorSets)
    fprintf('\nFinding behavior onsets for set %d (%s)...\n', s, behaviorSetNames{s});
    
    % Find behavior onsets (transitions TO this behavior set)
    % Look for transitions from any other behavior to behaviors in this set
    bhvOnsets = [];
    % Find all transitions into any of the target behaviors in this set
    bhvOnsets = [];
    for b = 1:length(behaviorSets{s})
        thisBhv = behaviorSets{s}(b);
        isThisBhv = (bhvID == thisBhv);
        transitions = find(diff([0; isThisBhv]) == 1); % transitions into this behavior
        bhvOnsets = [bhvOnsets; transitions];
    end
    bhvOnsets = sort(unique(bhvOnsets)); % Remove duplicates and sort
    if isempty(bhvOnsets)
        fprintf('  No behaviors found for set %d\n', s);
        bhvOnsetsBySet{s} = [];
        bhvOnsetTimesBySet{s} = [];
        continue;
    end
    
    % Convert behavior onset indices to time (seconds)
    bhvOnsetTimes = (bhvOnsets - 1) * binSizeBhv;
    
    fprintf('  Found %d behavior onsets\n', length(bhvOnsets));
    
    % Store for use in area loop
    bhvOnsetsBySet{s} = bhvOnsets;
    bhvOnsetTimesBySet{s} = bhvOnsetTimes;
end

% Process each area
for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});
    
    % Get d2 values and time points for this area
    d2Values = d2Nat{a};
    timePoints = startS{a};
    
    % Initialize arrays for this area
    halfWindow = floor(windowDurationFrames{a} / 2);
    
    % Create time axis for peri-behavior window (centered on behavior onset)
    timeAxisPeriBhv{a} = (-halfWindow:halfWindow) * optimalBinSize(a);
    
    % Process each behavior set
    for s = 1:length(behaviorSets)
        if isempty(bhvOnsetsBySet{s})
            d2WindowsBhv{s, a} = [];
            meanD2PeriBhv{s, a} = [];
            continue;
        end
        
        fprintf('  Processing behavior set %d (%s)...\n', s, behaviorSetNames{s});
        
        % Get pre-computed behavior onsets and times
        bhvOnsets = bhvOnsetsBySet{s};
        bhvOnsetTimes = bhvOnsetTimesBySet{s};
        
        % Initialize storage for this behavior set
        numBehaviors = length(bhvOnsets);
        d2WindowsBhv{s, a} = nan(numBehaviors, windowDurationFrames{a} + 1);
        
        % Extract d2 values around each behavior
        validBehaviors = 0;
        for b = 1:numBehaviors
            bhvTime = bhvOnsetTimes(b);
            [~, closestIdx] = min(abs(timePoints - bhvTime));
            
            winStart = closestIdx - halfWindow;
            winEnd = closestIdx + halfWindow;
            
            % Check if window is within bounds
            if winStart >= 1 && winEnd <= length(d2Values)
                d2WindowsBhv{s, a}(b, :) = d2Values(winStart:winEnd);
                validBehaviors = validBehaviors + 1;
            end
        end
        
        % Calculate mean d2 values across all valid behaviors
        if validBehaviors > 0
            meanD2PeriBhv{s, a} = nanmean(d2WindowsBhv{s, a}, 1);
            fprintf('    %d valid behaviors processed\n', validBehaviors);
        else
            meanD2PeriBhv{s, a} = [];
            fprintf('    No valid behaviors found\n');
        end
    end
end

%% ==============================================     Plotting Results     ==============================================

% Toggle plotting individual behavior traces
plotIndividualTraces = false;

% Create peri-behavior plots for each area
figure(400 + slidingWindowSize); clf;
set(gcf, 'Position', [100, 100, 1400, 1000]);

% Use tight_subplot for layout
ha = tight_subplot(2, 2, [0.15 0.1], [0.1 0.08], [0.1 0.05]);

% Compute global y-limits across areas and behavior sets
allMeanVals = [];
for a = areasToTest
    for s = 1:length(behaviorSets)
        if ~isempty(meanD2PeriBhv{s, a})
            allMeanVals = [allMeanVals; meanD2PeriBhv{s, a}(:)]; %#ok<AGROW>
        end
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
    
    % Plot individual behavior traces (if enabled)
    if plotIndividualTraces
        for s = 1:length(behaviorSets)
            if ~isempty(d2WindowsBhv{s, a})
                for b = 1:size(d2WindowsBhv{s, a}, 1)
                    if ~all(isnan(d2WindowsBhv{s, a}(b, :)))
                        plot(timeAxisPeriBhv{a}, d2WindowsBhv{s, a}(b, :), ...
                            'Color', [.7 .7 .7], 'LineWidth', 0.5);
                    end
                end
            end
        end
    end
    
    % Plot SEM ribbons and means for each behavior set
    legendHandles = [];
    legendLabels = {};
    
    for s = 1:length(behaviorSets)
        if ~isempty(meanD2PeriBhv{s, a}) && ~isempty(d2WindowsBhv{s, a})
            % Calculate SEM
            semBhv = nanstd(d2WindowsBhv{s, a}, 0, 1) / sqrt(sum(~all(isnan(d2WindowsBhv{s, a}), 2)));
            
            % Plot SEM ribbon
            fill([timeAxisPeriBhv{a}, fliplr(timeAxisPeriBhv{a})], ...
                 [meanD2PeriBhv{s, a} + semBhv, fliplr(meanD2PeriBhv{s, a} - semBhv)], ...
                 behaviorSetColors{s}, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            
            % Plot mean line
            hMean = plot(timeAxisPeriBhv{a}, meanD2PeriBhv{s, a}, 'Color', behaviorSetColors{s}, ...
                'LineWidth', 3, 'LineStyle', '-');
            
            legendHandles = [legendHandles, hMean]; %#ok<AGROW>
            legendLabels = [legendLabels, sprintf('%s (n=%d)', behaviorSetNames{s}, ...
                sum(~all(isnan(d2WindowsBhv{s, a}), 2)))]; %#ok<AGROW>
        end
    end
    
    % Add vertical line at behavior onset
    plot([0 0], ylim, 'k--', 'LineWidth', 2);
    
    % Formatting
    xlabel('Time relative to behavior onset (s)', 'FontSize', 12);
    ylabel('d2 Criticality', 'FontSize', 12);
    title(sprintf('%s - Peri-Behavior d2 Criticality (Window: %gs)', areas{a}, slidingWindowSize), 'FontSize', 14);
    grid on;
    
    % Set x-axis ticks and limits in seconds
    xMin = min(timeAxisPeriBhv{a});
    xMax = max(timeAxisPeriBhv{a});
    xlim([xMin xMax]);
    xTicks = ceil(xMin):floor(xMax);
    if isempty(xTicks)
        xTicks = linspace(xMin, xMax, 5);
    end
    xticks(xTicks);
    xticklabels(string(xTicks));
    
    % Set consistent y-axis limits and ticks across all subplots
    ylim(yLimCommon);
    yTicks = linspace(yLimCommon(1), yLimCommon(2), 5);
    yticks(yTicks);
    yticklabels(string(round(yTicks, 3)));
    
    % Add legend
    if ~isempty(legendHandles)
        legend(legendHandles, legendLabels, 'Location', 'best', 'FontSize', 10);
    end
end

sgtitle(sprintf('Peri-Behavior d2 Criticality Analysis (Sliding Window: %gs)', slidingWindowSize), 'FontSize', 16);

% Save plot
filename = fullfile(paths.dropPath, sprintf('criticality/peri_behavior_d2_criticality_win%gs.png', slidingWindowSize));
exportgraphics(gcf, filename, 'Resolution', 300);
fprintf('Saved peri-behavior plot to: %s\n', filename);

%% ==============================================     Summary Statistics     ==============================================

fprintf('\n=== Peri-Behavior Analysis Summary ===\n');
for a = areasToTest
    fprintf('\nArea %s:\n', areas{a});
    for s = 1:length(behaviorSets)
        if ~isempty(meanD2PeriBhv{s, a})
            validBehaviors = sum(~all(isnan(d2WindowsBhv{s, a}), 2));
            fprintf('  %s: %d valid behaviors\n', behaviorSetNames{s}, validBehaviors);
            if validBehaviors > 0
                fprintf('    Mean d2 at behavior onset: %.4f\n', meanD2PeriBhv{s, a}(halfWindow + 1));
            end
        end
    end
end

fprintf('\nPeri-behavior analysis complete!\n');
