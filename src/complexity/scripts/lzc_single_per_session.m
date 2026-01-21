%%
% LZC_SINGLE_PER_SESSION (Single Large Window)
% Combines Lempel-Ziv complexity (LZC) analyses in a single large window
% Compares metrics between spontaneous and reach sessions.
%
% Variables:
%   timeRange - Time range in seconds [startTime, endTime] (default: [0, 20*60])
%   useNormalized - Whether to use normalized metrics (default: true)
%   areasToPlot - Cell array of area names to plot (default: {} = all common areas)
%                 Example: {'M23', 'M56'} or {} for all areas
%   natColor - Color for spontaneous sessions (default: [0 191 255]/255)
%   reachColor - Color for reach sessions (default: [255 215 0]/255)
%
% Goal:
%   Load data for each session, analyze a single large window, and compare
%   LZC metrics between spontaneous and reach sessions.

%% Configuration
paths = get_paths;

timeRange = [5*60, 10*60];  % Time range in seconds [startTime, endTime]
useNormalized = true;    % Use normalized LZC if available
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
fprintf('\n=== LZC Analysis Across Sessions (Single Window) ===\n');
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

% LZC Analysis config - use single large window
config = struct();
config.slidingWindowSize = windowDuration;    % Window size in seconds
config.stepSize = windowDuration + 1;         % Larger than window size so only one window
config.nShuffles = 3;                         % Number of shuffles for normalization
config.binSize = 0.03;                        % Bin size for spikes (seconds)
config.makePlots = false;                     % Don't create individual plots
config.minDataPoints = 2e5;                   % Minimum data points per window (for optimization)
config.useBernoulliControl = false;           % Compute Bernoulli normalized metric
config.includeM2356 = true;                  % Include combined M23+M56 area
config.useOptimalBinSize = true;             % Set to true to automatically calculate optimal bin size per area
config.saveData = false;                     % Don't save individual session results
config.nMinNeurons = 15;                     % Minimum neurons per area

%% Initialize results storage
spontaneousData = struct();
spontaneousData.lzRaw = {};              % Cell array: {areaIdx}{sessionIdx} = LZC value
spontaneousData.lzNormalized = {};       % Cell array: {areaIdx}{sessionIdx} = LZC normalized by shuffles
spontaneousData.lzNormalizedBern = {};   % Cell array: {areaIdx}{sessionIdx} = LZC normalized by Bernoulli
spontaneousData.sessionNames = {};
spontaneousData.areas = [];

reachData = struct();
reachData.lzRaw = {};              % Cell array: {areaIdx}{sessionIdx} = LZC value
reachData.lzNormalized = {};       % Cell array: {areaIdx}{sessionIdx} = LZC normalized by shuffles
reachData.lzNormalizedBern = {};   % Cell array: {areaIdx}{sessionIdx} = LZC normalized by Bernoulli
reachData.sessionNames = {};
reachData.areas = [];

%% Set up parallel pool if running parallel in lzc_sliding_analysis
runParallel = 1;
% Check if parpool is already running, start one if not
if runParallel
    currentPool = gcp('nocreate');
if isempty(currentPool)
    NumWorkers = min(4, length(dataStruct.areas));
    parpool('local', NumWorkers);
    fprintf('Started parallel pool with %d workers\n', NumWorkers);
else
    fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
end
end

%% Process spontaneous sessions
fprintf('\n=== Processing Spontaneous Sessions ===\n');
for s = 1:length(spontaneousSessions)
    sessionName = spontaneousSessions{s};
    fprintf('\nProcessing session %d/%d: %s\n', s, length(spontaneousSessions), sessionName);
    
    try
        % Load data
        dataStruct = load_sliding_window_data('spontaneous', 'spikes', ...
            'sessionName', sessionName, 'opts', opts);
        
        % Run LZC analysis
        fprintf('  Running LZC analysis...\n');
        results = lzc_sliding_analysis(dataStruct, config);
        
        % Extract areas if not yet set
        if isempty(spontaneousData.areas) && isfield(results, 'areas')
            spontaneousData.areas = results.areas;
            numAreas = length(spontaneousData.areas);
            spontaneousData.lzRaw = cell(1, numAreas);
            spontaneousData.lzNormalized = cell(1, numAreas);
            spontaneousData.lzNormalizedBern = cell(1, numAreas);
            for a = 1:numAreas
                spontaneousData.lzRaw{a} = [];
                spontaneousData.lzNormalized{a} = [];
                spontaneousData.lzNormalizedBern{a} = [];
            end
        end
        
        % Extract LZC values per area (should be single value per area)
        if isfield(results, 'lzComplexity')
            for a = 1:length(results.areas)
                % Raw LZC
                if a <= length(results.lzComplexity) && ~isempty(results.lzComplexity{a})
                    lzValues = results.lzComplexity{a}(~isnan(results.lzComplexity{a}));
                    if ~isempty(lzValues)
                        spontaneousData.lzRaw{a} = [spontaneousData.lzRaw{a}, lzValues(1)];
                    else
                        spontaneousData.lzRaw{a} = [spontaneousData.lzRaw{a}, nan];
                    end
                else
                    spontaneousData.lzRaw{a} = [spontaneousData.lzRaw{a}, nan];
                end
                
                % Shuffled-normalized LZC
                if isfield(results, 'lzComplexityNormalized') && ...
                        a <= length(results.lzComplexityNormalized) && ...
                        ~isempty(results.lzComplexityNormalized{a})
                    lzNormValues = results.lzComplexityNormalized{a}(~isnan(results.lzComplexityNormalized{a}));
                    if ~isempty(lzNormValues)
                        spontaneousData.lzNormalized{a} = [spontaneousData.lzNormalized{a}, lzNormValues(1)];
                    else
                        spontaneousData.lzNormalized{a} = [spontaneousData.lzNormalized{a}, nan];
                    end
                else
                    spontaneousData.lzNormalized{a} = [spontaneousData.lzNormalized{a}, nan];
                end
                
                % Bernoulli-normalized LZC
                if isfield(results, 'lzComplexityNormalizedBernoulli') && ...
                        a <= length(results.lzComplexityNormalizedBernoulli) && ...
                        ~isempty(results.lzComplexityNormalizedBernoulli{a})
                    lzBernValues = results.lzComplexityNormalizedBernoulli{a}(~isnan(results.lzComplexityNormalizedBernoulli{a}));
                    if ~isempty(lzBernValues)
                        spontaneousData.lzNormalizedBern{a} = [spontaneousData.lzNormalizedBern{a}, lzBernValues(1)];
                    else
                        spontaneousData.lzNormalizedBern{a} = [spontaneousData.lzNormalizedBern{a}, nan];
                    end
                else
                    spontaneousData.lzNormalizedBern{a} = [spontaneousData.lzNormalizedBern{a}, nan];
                end
            end
        else
            warning('lzComplexity not found in results for session %s', sessionName);
            for a = 1:length(spontaneousData.areas)
                spontaneousData.lzRaw{a} = [spontaneousData.lzRaw{a}, nan];
                spontaneousData.lzNormalized{a} = [spontaneousData.lzNormalized{a}, nan];
                spontaneousData.lzNormalizedBern{a} = [spontaneousData.lzNormalizedBern{a}, nan];
            end
        end
        
        spontaneousData.sessionNames{end+1} = sessionName;
        fprintf('  ✓ Session completed successfully\n');
        
    catch ME
        fprintf('  ✗ Error processing session: %s\n', ME.message);
        % Add NaN for all areas
        if ~isempty(spontaneousData.areas)
            for a = 1:length(spontaneousData.areas)
                spontaneousData.lzRaw{a} = [spontaneousData.lzRaw{a}, nan];
                spontaneousData.lzNormalized{a} = [spontaneousData.lzNormalized{a}, nan];
                spontaneousData.lzNormalizedBern{a} = [spontaneousData.lzNormalizedBern{a}, nan];
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
        
        % Run LZC analysis
        fprintf('  Running LZC analysis...\n');
        results = lzc_sliding_analysis(dataStruct, config);
        
        % Extract areas if not yet set
        if isempty(reachData.areas) && isfield(results, 'areas')
            reachData.areas = results.areas;
            numAreas = length(reachData.areas);
            reachData.lzRaw = cell(1, numAreas);
            reachData.lzNormalized = cell(1, numAreas);
            reachData.lzNormalizedBern = cell(1, numAreas);
            for a = 1:numAreas
                reachData.lzRaw{a} = [];
                reachData.lzNormalized{a} = [];
                reachData.lzNormalizedBern{a} = [];
            end
        end
        
        % Extract LZC values per area (should be single value per area)
        if isfield(results, 'lzComplexity')
            for a = 1:length(results.areas)
                % Raw LZC
                if a <= length(results.lzComplexity) && ~isempty(results.lzComplexity{a})
                    lzValues = results.lzComplexity{a}(~isnan(results.lzComplexity{a}));
                    if ~isempty(lzValues)
                        reachData.lzRaw{a} = [reachData.lzRaw{a}, lzValues(1)];
                    else
                        reachData.lzRaw{a} = [reachData.lzRaw{a}, nan];
                    end
                else
                    reachData.lzRaw{a} = [reachData.lzRaw{a}, nan];
                end
                
                % Shuffled-normalized LZC
                if isfield(results, 'lzComplexityNormalized') && ...
                        a <= length(results.lzComplexityNormalized) && ...
                        ~isempty(results.lzComplexityNormalized{a})
                    lzNormValues = results.lzComplexityNormalized{a}(~isnan(results.lzComplexityNormalized{a}));
                    if ~isempty(lzNormValues)
                        reachData.lzNormalized{a} = [reachData.lzNormalized{a}, lzNormValues(1)];
                    else
                        reachData.lzNormalized{a} = [reachData.lzNormalized{a}, nan];
                    end
                else
                    reachData.lzNormalized{a} = [reachData.lzNormalized{a}, nan];
                end
                
                % Bernoulli-normalized LZC
                if isfield(results, 'lzComplexityNormalizedBernoulli') && ...
                        a <= length(results.lzComplexityNormalizedBernoulli) && ...
                        ~isempty(results.lzComplexityNormalizedBernoulli{a})
                    lzBernValues = results.lzComplexityNormalizedBernoulli{a}(~isnan(results.lzComplexityNormalizedBernoulli{a}));
                    if ~isempty(lzBernValues)
                        reachData.lzNormalizedBern{a} = [reachData.lzNormalizedBern{a}, lzBernValues(1)];
                    else
                        reachData.lzNormalizedBern{a} = [reachData.lzNormalizedBern{a}, nan];
                    end
                else
                    reachData.lzNormalizedBern{a} = [reachData.lzNormalizedBern{a}, nan];
                end
            end
        else
            warning('lzComplexity not found in results for session %s', sessionName);
            for a = 1:length(reachData.areas)
                reachData.lzRaw{a} = [reachData.lzRaw{a}, nan];
                reachData.lzNormalized{a} = [reachData.lzNormalized{a}, nan];
                reachData.lzNormalizedBern{a} = [reachData.lzNormalizedBern{a}, nan];
            end
        end
        
        reachData.sessionNames{end+1} = sessionName;
        fprintf('  ✓ Session completed successfully\n');
        
    catch ME
        fprintf('  ✗ Error processing session: %s\n', ME.message);
        % Add NaN for all areas
        if ~isempty(reachData.areas)
            for a = 1:length(reachData.areas)
                reachData.lzRaw{a} = [reachData.lzRaw{a}, nan];
                reachData.lzNormalized{a} = [reachData.lzNormalized{a}, nan];
                reachData.lzNormalizedBern{a} = [reachData.lzNormalizedBern{a}, nan];
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
fprintf('\n=== Creating LZC Plots ===\n');

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
allLzRaw = [];
allLzNorm = [];

for a = 1:numAreasToPlot
    areaName = areasToPlot{a};
    
    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));
    
    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end
    
    % Collect raw LZC values
    natLz = spontaneousData.lzRaw{natAreaIdx};
    reachLz = reachData.lzRaw{reachAreaIdx};
    natLz = natLz(~isnan(natLz));
    reachLz = reachLz(~isnan(reachLz));
    allLzRaw = [allLzRaw, natLz, reachLz];
    
    % Collect normalized LZC values (shuffled)
    natLzNorm = spontaneousData.lzNormalized{natAreaIdx};
    reachLzNorm = reachData.lzNormalized{reachAreaIdx};
    natLzNorm = natLzNorm(~isnan(natLzNorm));
    reachLzNorm = reachLzNorm(~isnan(reachLzNorm));
    allLzNorm = [allLzNorm, natLzNorm, reachLzNorm];
end

% Calculate global y-axis limits with some padding
if ~isempty(allLzRaw)
    ylimLzRaw = [0, max(allLzRaw) * 1.05];
    if ylimLzRaw(1) == ylimLzRaw(2)
        ylimLzRaw = ylimLzRaw(1) + [-0.1, 0.1];
    end
else
    ylimLzRaw = [0, 1];
end

if ~isempty(allLzNorm)
    minLzNorm = min(allLzNorm);
    maxLzNorm = max(allLzNorm);
    rangeLzNorm = maxLzNorm - minLzNorm;
    if rangeLzNorm == 0
        % If all values are identical, create a small symmetric range
        ylimLzNorm = minLzNorm + [-0.1, 0.1];
    else
        % Add a small buffer on both sides of the data range
        bufferLzNorm = 0.05 * rangeLzNorm;
        ylimLzNorm = [minLzNorm - bufferLzNorm, maxLzNorm + bufferLzNorm];
    end
else
    ylimLzNorm = [0, 1];
end

% Create figure with 2 columns: LZC raw, LZC normalized
figure(3100); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [100, 100, 1600, 300 * numAreasToPlot]);

for a = 1:numAreasToPlot
    areaName = areasToPlot{a};
    
    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));
    
    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end
    
    % Get values for this area
    natLz = spontaneousData.lzRaw{natAreaIdx};
    reachLz = reachData.lzRaw{reachAreaIdx};
    natLz = natLz(~isnan(natLz));
    reachLz = reachLz(~isnan(reachLz));
    
    natLzNorm = spontaneousData.lzNormalized{natAreaIdx};
    reachLzNorm = reachData.lzNormalized{reachAreaIdx};
    natLzNorm = natLzNorm(~isnan(natLzNorm));
    reachLzNorm = reachLzNorm(~isnan(reachLzNorm));
    
    % LZC Raw
    subplot(numAreasToPlot, 2, (a-1)*2 + 1);
    hold on;
    numNat = length(natLz);
    numReach = length(reachLz);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natLz, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachLz, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('LZC Raw');
    title(sprintf('%s - LZC Raw', areaName));
    grid on;
    ylim(ylimLzRaw);
    if ~isempty(natLz)
        yline(mean(natLz), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachLz)
        yline(mean(reachLz), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % LZC Normalized (shuffled)
    subplot(numAreasToPlot, 2, (a-1)*2 + 2);
    hold on;
    numNat = length(natLzNorm);
    numReach = length(reachLzNorm);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natLzNorm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachLzNorm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('LZC Normalized');
    title(sprintf('%s - LZC Normalized', areaName));
    grid on;
    ylim(ylimLzNorm);
    if ~isempty(natLzNorm)
        yline(mean(natLzNorm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachLzNorm)
        yline(mean(reachLzNorm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
end

% Add overall title
sgtitle(sprintf('LZC Comparison: Spontaneous vs Reach (Time Range: %.1f-%.1f s)', ...
    windowStartTime, windowEndTime), 'FontSize', 14, 'FontWeight', 'bold');

%% Save figure
saveDir = fullfile(paths.dropPath, 'sliding_window_comparisons');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

plotFilenamePng = sprintf('lzc_single_per_session_%.0f-%.0fs.png', windowStartTime, windowEndTime);
plotPathPng = fullfile(saveDir, plotFilenamePng);

exportgraphics(gcf, plotPathPng, 'Resolution', 300);
fprintf('\nSaved PNG plot to: %s\n', plotPathPng);

plotFilenameEps = sprintf('lzc_single_per_session_%.0f-%.0fs.eps', windowStartTime, windowEndTime);
plotPathEps = fullfile(saveDir, plotFilenameEps);

exportgraphics(gcf, plotPathEps, 'ContentType', 'vector');
fprintf('Saved EPS plot to: %s\n', plotPathEps);

fprintf('\n=== LZC Single-Window Analysis Complete ===\n');

