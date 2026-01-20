%%
% RQA_SINGLE_PER_SESSION (Single Large Window)
% Combines Recurrence Quantitative Analysis (RQA) metrics in a single large window
% Compares metrics between spontaneous and reach sessions.
%
% Variables:
%   timeRange - Time range in seconds [startTime, endTime] (default: [0, 20*60])
%   areasToPlot - Cell array of area names to plot (default: {} = all common areas)
%                 Example: {'M23', 'M56'} or {} for all areas
%   natColor - Color for spontaneous sessions (default: [0 191 255]/255)
%   reachColor - Color for reach sessions (default: [255 215 0]/255)
%
% Goal:
%   Load data for each session, analyze a single large window, and compare
%   RQA metrics (recurrence rate, determinism) between spontaneous and reach sessions.

%% Configuration
paths = get_paths;

timeRange = [0, 20*60];  % Time range in seconds [startTime, endTime]
areasToPlot = {};        % Cell array of area names to plot (empty = all common areas)
                         % Example: {'M23', 'M56'} or {} for all areas
natColor = [0 191 255] ./ 255;   % Color for spontaneous sessions
reachColor = [255 215 0] ./ 255; % Color for reach sessions

% Calculate window parameters from timeRange
windowStartTime = timeRange(1);
windowEndTime = timeRange(2);
windowDuration = windowEndTime - windowStartTime;  % Window duration in seconds
nMin = windowDuration / 60;  % Window size in minutes (for display purposes)

%% Get session lists
fprintf('\n=== RQA Analysis Across Sessions (Single Window) ===\n');
fprintf('Time range: [%.1f, %.1f] seconds (%.1f minutes)\n', ...
    windowStartTime, windowEndTime, nMin);

spontaneousSessions = spontaneous_session_list();
reachSessions = reach_session_list();

fprintf('Spontaneous sessions: %d\n', length(spontaneousSessions));
fprintf('Reach sessions: %d\n', length(reachSessions));

%% Configure analysis options
opts = neuro_behavior_options;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = windowStartTime;
opts.collectEnd = windowEndTime;  % Restrict data to timeRange
opts.minFiringRate = .1;
opts.maxFiringRate = 100;

% RQA Analysis config - use single large window
config = struct();
config.slidingWindowSize = windowDuration;     % Window size in seconds
config.stepSize = windowDuration + 1;          % Larger than window size so only one window
config.nShuffles = 3;                          % Number of shuffles for normalization
config.binSize = 0.03;                         % Bin size for spikes (seconds)
config.minTimeBins = 10000;                    % Minimum bins per window (used in defaults)
config.nPCADim = 3;                            % Number of PCA dimensions
config.recurrenceThreshold = 0.02;             % Target recurrence rate (2%)
config.distanceMetric = 'euclidean';           % Distance metric
config.nMinNeurons = 10;                       % Minimum neurons per area
config.makePlots = false;                      % Don't create per-session plots
config.useBernoulliControl = true;            % Compute Bernoulli normalized metric
config.saveRecurrencePlots = false;           % Don't store recurrence plots
config.includeM2356 = true;                   % Include combined M23+M56 area
config.useOptimalBinSize = false;             % We provide binSize explicitly
config.saveData = false;                      % Don't save individual session results
config.usePerWindowPCA = false;               % PCA on full session

%% Initialize results storage
spontaneousData = struct();
spontaneousData.rrRaw = {};              % Recurrence rate raw
spontaneousData.rrNormalized = {};       % Recurrence rate normalized (shuffled)
spontaneousData.detRaw = {};             % Determinism raw
spontaneousData.detNormalized = {};      % Determinism normalized (shuffled)
spontaneousData.sessionNames = {};
spontaneousData.areas = [];

reachData = struct();
reachData.rrRaw = {};
reachData.rrNormalized = {};
reachData.detRaw = {};
reachData.detNormalized = {};
reachData.sessionNames = {};
reachData.areas = [];

%% Process spontaneous sessions
fprintf('\n=== Processing Spontaneous Sessions ===\n');
for s = 1:length(spontaneousSessions)
    sessionName = spontaneousSessions{s};
    fprintf('\nProcessing session %d/%d: %s\n', s, length(spontaneousSessions), sessionName);
    
    try
        % Load data
        dataStruct = load_sliding_window_data('spontaneous', 'spikes', ...
            'sessionName', sessionName, 'opts', opts);
        
        % Run RQA analysis
        fprintf('  Running RQA analysis...\n');
        results = rqa_sliding_analysis(dataStruct, config);
        
        % Extract areas if not yet set
        if isempty(spontaneousData.areas) && isfield(results, 'areas')
            spontaneousData.areas = results.areas;
            numAreas = length(spontaneousData.areas);
            spontaneousData.rrRaw = cell(1, numAreas);
            spontaneousData.rrNormalized = cell(1, numAreas);
            spontaneousData.detRaw = cell(1, numAreas);
            spontaneousData.detNormalized = cell(1, numAreas);
            for a = 1:numAreas
                spontaneousData.rrRaw{a} = [];
                spontaneousData.rrNormalized{a} = [];
                spontaneousData.detRaw{a} = [];
                spontaneousData.detNormalized{a} = [];
            end
        end
        
        % Extract metrics per area (single value per area)
        if isfield(results, 'recurrenceRate') && isfield(results, 'determinism')
            for a = 1:length(results.areas)
                % Recurrence rate raw
                if a <= length(results.recurrenceRate) && ~isempty(results.recurrenceRate{a})
                    rrValues = results.recurrenceRate{a}(~isnan(results.recurrenceRate{a}));
                    if ~isempty(rrValues)
                        spontaneousData.rrRaw{a} = [spontaneousData.rrRaw{a}, rrValues(1)];
                    else
                        spontaneousData.rrRaw{a} = [spontaneousData.rrRaw{a}, nan];
                    end
                else
                    spontaneousData.rrRaw{a} = [spontaneousData.rrRaw{a}, nan];
                end
                
                % Recurrence rate normalized
                if isfield(results, 'recurrenceRateNormalized') && ...
                        a <= length(results.recurrenceRateNormalized) && ...
                        ~isempty(results.recurrenceRateNormalized{a})
                    rrNormValues = results.recurrenceRateNormalized{a}(~isnan(results.recurrenceRateNormalized{a}));
                    if ~isempty(rrNormValues)
                        spontaneousData.rrNormalized{a} = [spontaneousData.rrNormalized{a}, rrNormValues(1)];
                    else
                        spontaneousData.rrNormalized{a} = [spontaneousData.rrNormalized{a}, nan];
                    end
                else
                    spontaneousData.rrNormalized{a} = [spontaneousData.rrNormalized{a}, nan];
                end
                
                % Determinism raw
                if a <= length(results.determinism) && ~isempty(results.determinism{a})
                    detValues = results.determinism{a}(~isnan(results.determinism{a}));
                    if ~isempty(detValues)
                        spontaneousData.detRaw{a} = [spontaneousData.detRaw{a}, detValues(1)];
                    else
                        spontaneousData.detRaw{a} = [spontaneousData.detRaw{a}, nan];
                    end
                else
                    spontaneousData.detRaw{a} = [spontaneousData.detRaw{a}, nan];
                end
                
                % Determinism normalized
                if isfield(results, 'determinismNormalized') && ...
                        a <= length(results.determinismNormalized) && ...
                        ~isempty(results.determinismNormalized{a})
                    detNormValues = results.determinismNormalized{a}(~isnan(results.determinismNormalized{a}));
                    if ~isempty(detNormValues)
                        spontaneousData.detNormalized{a} = [spontaneousData.detNormalized{a}, detNormValues(1)];
                    else
                        spontaneousData.detNormalized{a} = [spontaneousData.detNormalized{a}, nan];
                    end
                else
                    spontaneousData.detNormalized{a} = [spontaneousData.detNormalized{a}, nan];
                end
            end
        else
            warning('RQA metrics not found in results for session %s', sessionName);
            for a = 1:length(spontaneousData.areas)
                spontaneousData.rrRaw{a} = [spontaneousData.rrRaw{a}, nan];
                spontaneousData.rrNormalized{a} = [spontaneousData.rrNormalized{a}, nan];
                spontaneousData.detRaw{a} = [spontaneousData.detRaw{a}, nan];
                spontaneousData.detNormalized{a} = [spontaneousData.detNormalized{a}, nan];
            end
        end
        
        spontaneousData.sessionNames{end+1} = sessionName;
        fprintf('  ✓ Session completed successfully\n');
        
    catch ME
        fprintf('  ✗ Error processing session: %s\n', ME.message);
        % Add NaN for all areas
        if ~isempty(spontaneousData.areas)
            for a = 1:length(spontaneousData.areas)
                spontaneousData.rrRaw{a} = [spontaneousData.rrRaw{a}, nan];
                spontaneousData.rrNormalized{a} = [spontaneousData.rrNormalized{a}, nan];
                spontaneousData.detRaw{a} = [spontaneousData.detRaw{a}, nan];
                spontaneousData.detNormalized{a} = [spontaneousData.detNormalized{a}, nan];
            end
        end
        spontaneousData.sessionNames{end+1} = sessionName;
    end
end

%% Process reach sessions
fprintf('\n=== Processing Reach Sessions ===\n');
% Reset opts for reach sessions
opts.collectStart = windowStartTime;
opts.collectEnd = windowEndTime;  % End time from timeRange

for s = 1:length(reachSessions)
    sessionName = reachSessions{s};
    fprintf('\nProcessing session %d/%d: %s\n', s, length(reachSessions), sessionName);
    
    try
        % Load data
        dataStruct = load_sliding_window_data('reach', 'spikes', ...
            'sessionName', sessionName, 'opts', opts);
        
        % Run RQA analysis
        fprintf('  Running RQA analysis...\n');
        results = rqa_sliding_analysis(dataStruct, config);
        
        % Extract areas if not yet set
        if isempty(reachData.areas) && isfield(results, 'areas')
            reachData.areas = results.areas;
            numAreas = length(reachData.areas);
            reachData.rrRaw = cell(1, numAreas);
            reachData.rrNormalized = cell(1, numAreas);
            reachData.detRaw = cell(1, numAreas);
            reachData.detNormalized = cell(1, numAreas);
            for a = 1:numAreas
                reachData.rrRaw{a} = [];
                reachData.rrNormalized{a} = [];
                reachData.detRaw{a} = [];
                reachData.detNormalized{a} = [];
            end
        end
        
        % Extract metrics per area (single value per area)
        if isfield(results, 'recurrenceRate') && isfield(results, 'determinism')
            for a = 1:length(results.areas)
                % Recurrence rate raw
                if a <= length(results.recurrenceRate) && ~isempty(results.recurrenceRate{a})
                    rrValues = results.recurrenceRate{a}(~isnan(results.recurrenceRate{a}));
                    if ~isempty(rrValues)
                        reachData.rrRaw{a} = [reachData.rrRaw{a}, rrValues(1)];
                    else
                        reachData.rrRaw{a} = [reachData.rrRaw{a}, nan];
                    end
                else
                    reachData.rrRaw{a} = [reachData.rrRaw{a}, nan];
                end
                
                % Recurrence rate normalized
                if isfield(results, 'recurrenceRateNormalized') && ...
                        a <= length(results.recurrenceRateNormalized) && ...
                        ~isempty(results.recurrenceRateNormalized{a})
                    rrNormValues = results.recurrenceRateNormalized{a}(~isnan(results.recurrenceRateNormalized{a}));
                    if ~isempty(rrNormValues)
                        reachData.rrNormalized{a} = [reachData.rrNormalized{a}, rrNormValues(1)];
                    else
                        reachData.rrNormalized{a} = [reachData.rrNormalized{a}, nan];
                    end
                else
                    reachData.rrNormalized{a} = [reachData.rrNormalized{a}, nan];
                end
                
                % Determinism raw
                if a <= length(results.determinism) && ~isempty(results.determinism{a})
                    detValues = results.determinism{a}(~isnan(results.determinism{a}));
                    if ~isempty(detValues)
                        reachData.detRaw{a} = [reachData.detRaw{a}, detValues(1)];
                    else
                        reachData.detRaw{a} = [reachData.detRaw{a}, nan];
                    end
                else
                    reachData.detRaw{a} = [reachData.detRaw{a}, nan];
                end
                
                % Determinism normalized
                if isfield(results, 'determinismNormalized') && ...
                        a <= length(results.determinismNormalized) && ...
                        ~isempty(results.determinismNormalized{a})
                    detNormValues = results.determinismNormalized{a}(~isnan(results.determinismNormalized{a}));
                    if ~isempty(detNormValues)
                        reachData.detNormalized{a} = [reachData.detNormalized{a}, detNormValues(1)];
                    else
                        reachData.detNormalized{a} = [reachData.detNormalized{a}, nan];
                    end
                else
                    reachData.detNormalized{a} = [reachData.detNormalized{a}, nan];
                end
            end
        else
            warning('RQA metrics not found in results for session %s', sessionName);
            for a = 1:length(reachData.areas)
                reachData.rrRaw{a} = [reachData.rrRaw{a}, nan];
                reachData.rrNormalized{a} = [reachData.rrNormalized{a}, nan];
                reachData.detRaw{a} = [reachData.detRaw{a}, nan];
                reachData.detNormalized{a} = [reachData.detNormalized{a}, nan];
            end
        end
        
        reachData.sessionNames{end+1} = sessionName;
        fprintf('  ✓ Session completed successfully\n');
        
    catch ME
        fprintf('  ✗ Error processing session: %s\n', ME.message);
        % Add NaN for all areas
        if ~isempty(reachData.areas)
            for a = 1:length(reachData.areas)
                reachData.rrRaw{a} = [reachData.rrRaw{a}, nan];
                reachData.rrNormalized{a} = [reachData.rrNormalized{a}, nan];
                reachData.detRaw{a} = [reachData.detRaw{a}, nan];
                reachData.detNormalized{a} = [reachData.detNormalized{a}, nan];
            end
        end
        reachData.sessionNames{end+1} = sessionName;
    end
end

%% Determine common areas
if isempty(spontaneousData.areas) || isempty(reachData.areas)
    error('No data loaded. Check that sessions were processed successfully.');
end

% Find common areas
commonAreas = intersect(spontaneousData.areas, reachData.areas);
if isempty(commonAreas)
    error('No common areas found between spontaneous and reach sessions.');
end

fprintf('\n=== Common Areas Found ===\n');
for a = 1:length(commonAreas)
    fprintf('  %s\n', commonAreas{a});
end

%% Create plots
fprintf('\n=== Creating RQA Plots ===\n');

% Determine which areas to plot
if isempty(areasToPlot)
    % Use all common areas if not specified
    areasToPlot = commonAreas;
else
    % Filter to only include areas that exist in common areas
    areasToPlot = intersect(areasToPlot, commonAreas);
    if isempty(areasToPlot)
        error('None of the specified areas to plot are found in common areas.');
    end
    fprintf('Plotting specified areas: %s\n', strjoin(areasToPlot, ', '));
end
numAreasToPlot = length(areasToPlot);

% Collect all values to determine global y-axis limits
allRrRaw = [];
allRrNorm = [];
allDetRaw = [];
allDetNorm = [];

for a = 1:numAreasToPlot
    areaName = areasToPlot{a};
    
    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));
    
    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end
    
    % Recurrence rate
    natRr = spontaneousData.rrRaw{natAreaIdx};
    reachRr = reachData.rrRaw{reachAreaIdx};
    natRr = natRr(~isnan(natRr));
    reachRr = reachRr(~isnan(reachRr));
    allRrRaw = [allRrRaw, natRr, reachRr];
    
    natRrNorm = spontaneousData.rrNormalized{natAreaIdx};
    reachRrNorm = reachData.rrNormalized{reachAreaIdx};
    natRrNorm = natRrNorm(~isnan(natRrNorm));
    reachRrNorm = reachRrNorm(~isnan(reachRrNorm));
    allRrNorm = [allRrNorm, natRrNorm, reachRrNorm];
    
    % Determinism
    natDet = spontaneousData.detRaw{natAreaIdx};
    reachDet = reachData.detRaw{reachAreaIdx};
    natDet = natDet(~isnan(natDet));
    reachDet = reachDet(~isnan(reachDet));
    allDetRaw = [allDetRaw, natDet, reachDet];
    
    natDetNorm = spontaneousData.detNormalized{natAreaIdx};
    reachDetNorm = reachData.detNormalized{reachAreaIdx};
    natDetNorm = natDetNorm(~isnan(natDetNorm));
    reachDetNorm = reachDetNorm(~isnan(reachDetNorm));
    allDetNorm = [allDetNorm, natDetNorm, reachDetNorm];
end

% Calculate global y-axis limits with some padding
if ~isempty(allRrRaw)
    ylimRrRaw = [0, max(allRrRaw) * 1.05];
    if ylimRrRaw(1) == ylimRrRaw(2)
        ylimRrRaw = ylimRrRaw(1) + [-0.1, 0.1];
    end
else
    ylimRrRaw = [0, 1];
end

if ~isempty(allRrNorm)
    ylimRrNorm = [0, max(allRrNorm) * 1.05];
    if ylimRrNorm(1) == ylimRrNorm(2)
        ylimRrNorm = ylimRrNorm(1) + [-0.1, 0.1];
    end
else
    ylimRrNorm = [0, 1];
end

if ~isempty(allDetRaw)
    ylimDetRaw = [0, max(allDetRaw) * 1.05];
    if ylimDetRaw(1) == ylimDetRaw(2)
        ylimDetRaw = ylimDetRaw(1) + [-0.1, 0.1];
    end
else
    ylimDetRaw = [0, 1];
end

if ~isempty(allDetNorm)
    ylimDetNorm = [0, max(allDetNorm) * 1.05];
    if ylimDetNorm(1) == ylimDetNorm(2)
        ylimDetNorm = ylimDetNorm(1) + [-0.1, 0.1];
    end
else
    ylimDetNorm = [0, 1];
end

% Create figure with 4 columns: RR raw, RR normalized, DET raw, DET normalized
figure(3200); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [100, 100, 2200, 300 * numAreasToPlot]);

for a = 1:numAreasToPlot
    areaName = areasToPlot{a};
    
    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));
    
    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end
    
    % Recurrence rate
    natRr = spontaneousData.rrRaw{natAreaIdx};
    reachRr = reachData.rrRaw{reachAreaIdx};
    natRr = natRr(~isnan(natRr));
    reachRr = reachRr(~isnan(reachRr));
    
    natRrNorm = spontaneousData.rrNormalized{natAreaIdx};
    reachRrNorm = reachData.rrNormalized{reachAreaIdx};
    natRrNorm = natRrNorm(~isnan(natRrNorm));
    reachRrNorm = reachRrNorm(~isnan(reachRrNorm));
    
    % Determinism
    natDet = spontaneousData.detRaw{natAreaIdx};
    reachDet = reachData.detRaw{reachAreaIdx};
    natDet = natDet(~isnan(natDet));
    reachDet = reachDet(~isnan(reachDet));
    
    natDetNorm = spontaneousData.detNormalized{natAreaIdx};
    reachDetNorm = reachData.detNormalized{reachAreaIdx};
    natDetNorm = natDetNorm(~isnan(natDetNorm));
    reachDetNorm = reachDetNorm(~isnan(reachDetNorm));
    
    % RR Raw
    subplot(numAreasToPlot, 4, (a-1)*4 + 1);
    hold on;
    numNat = length(natRr);
    numReach = length(reachRr);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natRr, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachRr, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    xlim([0.5, numNat + numReach + 0.5]);
    if numNat > 0 && numReach > 0
        xticks([mean(xNat), mean(xReach)]);
        xticklabels({'Spontaneous', 'Reach'});
    elseif numNat > 0
        xticks(mean(xNat));
        xticklabels({'Spontaneous'});
    elseif numReach > 0
        xticks(mean(xReach));
        xticklabels({'Reach'});
    end
    ylabel('RR Raw');
    title(sprintf('%s - RR Raw', areaName));
    grid on;
    ylim(ylimRrRaw);
    if ~isempty(natRr)
        yline(mean(natRr), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachRr)
        yline(mean(reachRr), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % RR Normalized
    subplot(numAreasToPlot, 4, (a-1)*4 + 2);
    hold on;
    numNat = length(natRrNorm);
    numReach = length(reachRrNorm);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natRrNorm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachRrNorm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    xlim([0.5, numNat + numReach + 0.5]);
    if numNat > 0 && numReach > 0
        xticks([mean(xNat), mean(xReach)]);
        xticklabels({'Spontaneous', 'Reach'});
    elseif numNat > 0
        xticks(mean(xNat));
        xticklabels({'Spontaneous'});
    elseif numReach > 0
        xticks(mean(xReach));
        xticklabels({'Reach'});
    end
    ylabel('RR Normalized');
    title(sprintf('%s - RR Normalized', areaName));
    grid on;
    ylim(ylimRrNorm);
    if ~isempty(natRrNorm)
        yline(mean(natRrNorm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachRrNorm)
        yline(mean(reachRrNorm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % DET Raw
    subplot(numAreasToPlot, 4, (a-1)*4 + 3);
    hold on;
    numNat = length(natDet);
    numReach = length(reachDet);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natDet, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachDet, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    xlim([0.5, numNat + numReach + 0.5]);
    if numNat > 0 && numReach > 0
        xticks([mean(xNat), mean(xReach)]);
        xticklabels({'Spontaneous', 'Reach'});
    elseif numNat > 0
        xticks(mean(xNat));
        xticklabels({'Spontaneous'});
    elseif numReach > 0
        xticks(mean(xReach));
        xticklabels({'Reach'});
    end
    ylabel('DET Raw');
    title(sprintf('%s - DET Raw', areaName));
    grid on;
    ylim(ylimDetRaw);
    if ~isempty(natDet)
        yline(mean(natDet), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachDet)
        yline(mean(reachDet), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % DET Normalized
    subplot(numAreasToPlot, 4, (a-1)*4 + 4);
    hold on;
    numNat = length(natDetNorm);
    numReach = length(reachDetNorm);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natDetNorm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachDetNorm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    xlim([0.5, numNat + numReach + 0.5]);
    if numNat > 0 && numReach > 0
        xticks([mean(xNat), mean(xReach)]);
        xticklabels({'Spontaneous', 'Reach'});
    elseif numNat > 0
        xticks(mean(xNat));
        xticklabels({'Spontaneous'});
    elseif numReach > 0
        xticks(mean(xReach));
        xticklabels({'Reach'});
    end
    ylabel('DET Normalized');
    title(sprintf('%s - DET Normalized', areaName));
    grid on;
    ylim(ylimDetNorm);
    if ~isempty(natDetNorm)
        yline(mean(natDetNorm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachDetNorm)
        yline(mean(reachDetNorm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
end

% Add overall title
sgtitle(sprintf('RQA Comparison: Spontaneous vs Reach (Time Range: %.1f-%.1f s)', ...
    windowStartTime, windowEndTime), 'FontSize', 14, 'FontWeight', 'bold');

%% Save figure
saveDir = fullfile(paths.dropPath, 'sliding_window_comparisons');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

plotFilenamePng = sprintf('rqa_single_per_session_%.0f-%.0fs.png', windowStartTime, windowEndTime);
plotPathPng = fullfile(saveDir, plotFilenamePng);

exportgraphics(gcf, plotPathPng, 'Resolution', 300);
fprintf('\nSaved PNG plot to: %s\n', plotPathPng);

plotFilenameEps = sprintf('rqa_single_per_session_%.0f-%.0fs.eps', windowStartTime, windowEndTime);
plotPathEps = fullfile(saveDir, plotFilenameEps);

exportgraphics(gcf, plotPathEps, 'ContentType', 'vector');
fprintf('Saved EPS plot to: %s\n', plotPathEps);

fprintf('\n=== RQA Single-Window Analysis Complete ===\n');

