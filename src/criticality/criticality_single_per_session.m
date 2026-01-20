%%
% Criticality Single Per Session (Single Large Window)
% Combines d2 (AR) and avalanche (dcc, kappa) analyses in a single large window
% Compares metrics between spontaneous and reach sessions
%
% Variables:
%   timeRange - Time range in seconds [startTime, endTime] (default: [0, 20*60])
%   useNormalized - Whether to use normalized metrics (default: true)
%   areasToPlot - Cell array of area names to plot (default: [] = all common areas)
%                 Example: {'M1', 'PMd'} or [] for all areas
%   natColor - Color for spontaneous sessions (default: [0 191 255]/255)
%   reachColor - Color for reach sessions (default: [255 215 0]/255)
%
% Goal:
%   Load data for each session, analyze a single large window, and compare
%   d2, dcc, and kappa metrics between spontaneous and reach sessions.

%% Configuration
paths = get_paths;

timeRange = [0, 20*60];  % Time range in seconds [startTime, endTime]
useNormalized = true;  % Use normalized metrics if available
areasToPlot = {};  % Cell array of area names to plot (empty = all common areas)
                   % Example: {'M1', 'PMd'} or [] for all areas
natColor = [0 191 255] ./ 255;  % Color for spontaneous sessions
reachColor = [255 215 0] ./ 255;  % Color for reach sessions

% Calculate window parameters from timeRange
windowStartTime = timeRange(1);
windowEndTime = timeRange(2);
windowDuration = windowEndTime - windowStartTime;  % Window duration in seconds
nMin = windowDuration / 60;  % Window size in minutes (for display purposes)

%% Get session lists
fprintf('\n=== Criticality Analysis Across Sessions (Single Window) ===\n');
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
opts.collectEnd = windowEndTime;  % End time from timeRange
opts.minFiringRate = .1;
opts.maxFiringRate = 100;

% AR Analysis config - use single large window
configAR = struct();
configAR.binSize = .03;
configAR.slidingWindowSize = windowDuration;  % Window size in seconds
configAR.stepSize = windowDuration + 1;  % Larger than window size so only one window
configAR.minSpikesPerBin = 2.5;
configAR.minBinsPerWindow = 1000;
configAR.analyzeD2 = true;
configAR.analyzeMrBr = false;
configAR.pcaFlag = 1;
configAR.pcaFirstFlag = 1;
configAR.nDim = 4;
configAR.enablePermutations = true;
configAR.nShuffles = 20;
configAR.analyzeModulation = false;
configAR.makePlots = false;  % Don't create individual plots
configAR.saveData = false;  % Don't save individual session results
configAR.useOptimalBinWindowFunction = false;
configAR.pOrder = 10;
configAR.critType = 2;
configAR.normalizeD2 = true;
configAR.maxSpikesPerBin = 50;
configAR.nMinNeurons = 15;
configAR.includeM2356 = true;

% AV Analysis config - use single large window
configAV = struct();
configAV.binSize = .03;
configAV.slidingWindowSize = windowDuration;  % Window size in seconds
configAV.avStepSize = windowDuration + 1;  % Larger than window size so only one window
configAV.minSpikesPerBin = 2.5;
configAV.minBinsPerWindow = 1000;
configAV.analyzeDcc = true;
configAV.analyzeKappa = true;
configAV.pcaFlag = 0;
configAV.pcaFirstFlag = 1;
configAV.nDim = 4;
configAV.enablePermutations = true;
configAV.nShuffles = 10;
configAV.makePlots = false;  % Don't create individual plots
configAV.saveData = false;  % Don't save individual session results
configAV.useOptimalBinWindowFunction = false;
configAV.thresholdFlag = 1;
configAV.thresholdPct = 1;
configAV.nMinNeurons = 10;
configAV.normalizeMetrics = true;  % Normalize metrics by shuffled values
configAV.includeM2356 = true;

%% Initialize results storage
spontaneousData = struct();
spontaneousData.d2Raw = {};  % Cell array: {areaIdx}{sessionIdx} = d2 value
spontaneousData.d2Normalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized d2 value
spontaneousData.dcc = {};  % Cell array: {areaIdx}{sessionIdx} = dcc value
spontaneousData.dccNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized dcc value
spontaneousData.kappa = {};  % Cell array: {areaIdx}{sessionIdx} = kappa value
spontaneousData.kappaNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized kappa value
spontaneousData.sessionNames = {};
spontaneousData.areas = [];

reachData = struct();
reachData.d2Raw = {};  % Cell array: {areaIdx}{sessionIdx} = d2 value
reachData.d2Normalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized d2 value
reachData.dcc = {};  % Cell array: {areaIdx}{sessionIdx} = dcc value
reachData.dccNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized dcc value
reachData.kappa = {};  % Cell array: {areaIdx}{sessionIdx} = kappa value
reachData.kappaNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized kappa value
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
        
        % Run AR analysis
        fprintf('  Running AR analysis...\n');
        resultsAR = criticality_ar_analysis(dataStruct, configAR);
        
        % Run AV analysis
        fprintf('  Running AV analysis...\n');
        resultsAV = criticality_av_analysis(dataStruct, configAV);
        
        % Extract areas if not yet set
        if isempty(spontaneousData.areas) && isfield(resultsAR, 'areas')
            spontaneousData.areas = resultsAR.areas;
            numAreas = length(spontaneousData.areas);
            spontaneousData.d2Raw = cell(1, numAreas);
            spontaneousData.d2Normalized = cell(1, numAreas);
            spontaneousData.dcc = cell(1, numAreas);
            spontaneousData.dccNormalized = cell(1, numAreas);
            spontaneousData.kappa = cell(1, numAreas);
            spontaneousData.kappaNormalized = cell(1, numAreas);
            for a = 1:numAreas
                spontaneousData.d2Raw{a} = [];
                spontaneousData.d2Normalized{a} = [];
                spontaneousData.dcc{a} = [];
                spontaneousData.dccNormalized{a} = [];
                spontaneousData.kappa{a} = [];
                spontaneousData.kappaNormalized{a} = [];
            end
        end
        
        % Extract d2 values per area (should be single value per area)
        if isfield(resultsAR, 'd2') && isfield(resultsAR, 'd2Normalized')
            for a = 1:length(resultsAR.areas)
                if a <= length(resultsAR.d2) && ~isempty(resultsAR.d2{a})
                    d2Values = resultsAR.d2{a}(~isnan(resultsAR.d2{a}));
                    if ~isempty(d2Values)
                        spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, d2Values(1)];
                    else
                        spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, nan];
                    end
                else
                    spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, nan];
                end
                
                if a <= length(resultsAR.d2Normalized) && ~isempty(resultsAR.d2Normalized{a})
                    d2NormValues = resultsAR.d2Normalized{a}(~isnan(resultsAR.d2Normalized{a}));
                    if ~isempty(d2NormValues)
                        spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, d2NormValues(1)];
                    else
                        spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, nan];
                    end
                else
                    spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, nan];
                end
            end
        else
            warning('d2 or d2Normalized not found in AR results for session %s', sessionName);
            for a = 1:length(spontaneousData.areas)
                spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, nan];
                spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, nan];
            end
        end
        
        % Extract avalanche values per area (should be single value per area)
        if isfield(resultsAV, 'dcc') && isfield(resultsAV, 'kappa')
            for a = 1:length(resultsAV.areas)
                % Extract dcc raw
                if a <= length(resultsAV.dcc) && ~isempty(resultsAV.dcc{a})
                    dccValues = resultsAV.dcc{a}(~isnan(resultsAV.dcc{a}));
                    if ~isempty(dccValues)
                        spontaneousData.dcc{a} = [spontaneousData.dcc{a}, dccValues(1)];
                    else
                        spontaneousData.dcc{a} = [spontaneousData.dcc{a}, nan];
                    end
                else
                    spontaneousData.dcc{a} = [spontaneousData.dcc{a}, nan];
                end
                
                % Extract dcc normalized
                if a <= length(resultsAV.dccNormalized) && ~isempty(resultsAV.dccNormalized{a})
                    dccNormValues = resultsAV.dccNormalized{a}(~isnan(resultsAV.dccNormalized{a}));
                    if ~isempty(dccNormValues)
                        spontaneousData.dccNormalized{a} = [spontaneousData.dccNormalized{a}, dccNormValues(1)];
                    else
                        spontaneousData.dccNormalized{a} = [spontaneousData.dccNormalized{a}, nan];
                    end
                else
                    spontaneousData.dccNormalized{a} = [spontaneousData.dccNormalized{a}, nan];
                end
                
                % Extract kappa raw
                if a <= length(resultsAV.kappa) && ~isempty(resultsAV.kappa{a})
                    kappaValues = resultsAV.kappa{a}(~isnan(resultsAV.kappa{a}));
                    if ~isempty(kappaValues)
                        spontaneousData.kappa{a} = [spontaneousData.kappa{a}, kappaValues(1)];
                    else
                        spontaneousData.kappa{a} = [spontaneousData.kappa{a}, nan];
                    end
                else
                    spontaneousData.kappa{a} = [spontaneousData.kappa{a}, nan];
                end
                
                % Extract kappa normalized
                if a <= length(resultsAV.kappaNormalized) && ~isempty(resultsAV.kappaNormalized{a})
                    kappaNormValues = resultsAV.kappaNormalized{a}(~isnan(resultsAV.kappaNormalized{a}));
                    if ~isempty(kappaNormValues)
                        spontaneousData.kappaNormalized{a} = [spontaneousData.kappaNormalized{a}, kappaNormValues(1)];
                    else
                        spontaneousData.kappaNormalized{a} = [spontaneousData.kappaNormalized{a}, nan];
                    end
                else
                    spontaneousData.kappaNormalized{a} = [spontaneousData.kappaNormalized{a}, nan];
                end
            end
        else
            warning('dcc or kappa not found in AV results for session %s', sessionName);
            for a = 1:length(spontaneousData.areas)
                spontaneousData.dcc{a} = [spontaneousData.dcc{a}, nan];
                spontaneousData.dccNormalized{a} = [spontaneousData.dccNormalized{a}, nan];
                spontaneousData.kappa{a} = [spontaneousData.kappa{a}, nan];
                spontaneousData.kappaNormalized{a} = [spontaneousData.kappaNormalized{a}, nan];
            end
        end
        
        spontaneousData.sessionNames{end+1} = sessionName;
        fprintf('  ✓ Session completed successfully\n');
        
    catch ME
        fprintf('  ✗ Error processing session: %s\n', ME.message);
        % Add NaN for all areas
        if ~isempty(spontaneousData.areas)
            for a = 1:length(spontaneousData.areas)
                spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, nan];
                spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, nan];
                spontaneousData.dcc{a} = [spontaneousData.dcc{a}, nan];
                spontaneousData.dccNormalized{a} = [spontaneousData.dccNormalized{a}, nan];
                spontaneousData.kappa{a} = [spontaneousData.kappa{a}, nan];
                spontaneousData.kappaNormalized{a} = [spontaneousData.kappaNormalized{a}, nan];
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
        
        % Run AR analysis
        fprintf('  Running AR analysis...\n');
        resultsAR = criticality_ar_analysis(dataStruct, configAR);
        
        % Run AV analysis
        fprintf('  Running AV analysis...\n');
        resultsAV = criticality_av_analysis(dataStruct, configAV);
        
        % Extract areas if not yet set
        if isempty(reachData.areas) && isfield(resultsAR, 'areas')
            reachData.areas = resultsAR.areas;
            numAreas = length(reachData.areas);
            reachData.d2Raw = cell(1, numAreas);
            reachData.d2Normalized = cell(1, numAreas);
            reachData.dcc = cell(1, numAreas);
            reachData.dccNormalized = cell(1, numAreas);
            reachData.kappa = cell(1, numAreas);
            reachData.kappaNormalized = cell(1, numAreas);
            for a = 1:numAreas
                reachData.d2Raw{a} = [];
                reachData.d2Normalized{a} = [];
                reachData.dcc{a} = [];
                reachData.dccNormalized{a} = [];
                reachData.kappa{a} = [];
                reachData.kappaNormalized{a} = [];
            end
        end
        
        % Extract d2 values per area (should be single value per area)
        if isfield(resultsAR, 'd2') && isfield(resultsAR, 'd2Normalized')
            for a = 1:length(resultsAR.areas)
                if a <= length(resultsAR.d2) && ~isempty(resultsAR.d2{a})
                    d2Values = resultsAR.d2{a}(~isnan(resultsAR.d2{a}));
                    if ~isempty(d2Values)
                        reachData.d2Raw{a} = [reachData.d2Raw{a}, d2Values(1)];
                    else
                        reachData.d2Raw{a} = [reachData.d2Raw{a}, nan];
                    end
                else
                    reachData.d2Raw{a} = [reachData.d2Raw{a}, nan];
                end
                
                if a <= length(resultsAR.d2Normalized) && ~isempty(resultsAR.d2Normalized{a})
                    d2NormValues = resultsAR.d2Normalized{a}(~isnan(resultsAR.d2Normalized{a}));
                    if ~isempty(d2NormValues)
                        reachData.d2Normalized{a} = [reachData.d2Normalized{a}, d2NormValues(1)];
                    else
                        reachData.d2Normalized{a} = [reachData.d2Normalized{a}, nan];
                    end
                else
                    reachData.d2Normalized{a} = [reachData.d2Normalized{a}, nan];
                end
            end
        else
            warning('d2 or d2Normalized not found in AR results for session %s', sessionName);
            for a = 1:length(reachData.areas)
                reachData.d2Raw{a} = [reachData.d2Raw{a}, nan];
                reachData.d2Normalized{a} = [reachData.d2Normalized{a}, nan];
            end
        end
        
        % Extract avalanche values per area (should be single value per area)
        if isfield(resultsAV, 'dcc') && isfield(resultsAV, 'kappa')
            for a = 1:length(resultsAV.areas)
                % Extract dcc raw
                if a <= length(resultsAV.dcc) && ~isempty(resultsAV.dcc{a})
                    dccValues = resultsAV.dcc{a}(~isnan(resultsAV.dcc{a}));
                    if ~isempty(dccValues)
                        reachData.dcc{a} = [reachData.dcc{a}, dccValues(1)];
                    else
                        reachData.dcc{a} = [reachData.dcc{a}, nan];
                    end
                else
                    reachData.dcc{a} = [reachData.dcc{a}, nan];
                end
                
                % Extract dcc normalized
                if a <= length(resultsAV.dccNormalized) && ~isempty(resultsAV.dccNormalized{a})
                    dccNormValues = resultsAV.dccNormalized{a}(~isnan(resultsAV.dccNormalized{a}));
                    if ~isempty(dccNormValues)
                        reachData.dccNormalized{a} = [reachData.dccNormalized{a}, dccNormValues(1)];
                    else
                        reachData.dccNormalized{a} = [reachData.dccNormalized{a}, nan];
                    end
                else
                    reachData.dccNormalized{a} = [reachData.dccNormalized{a}, nan];
                end
                
                % Extract kappa raw
                if a <= length(resultsAV.kappa) && ~isempty(resultsAV.kappa{a})
                    kappaValues = resultsAV.kappa{a}(~isnan(resultsAV.kappa{a}));
                    if ~isempty(kappaValues)
                        reachData.kappa{a} = [reachData.kappa{a}, kappaValues(1)];
                    else
                        reachData.kappa{a} = [reachData.kappa{a}, nan];
                    end
                else
                    reachData.kappa{a} = [reachData.kappa{a}, nan];
                end
                
                % Extract kappa normalized
                if a <= length(resultsAV.kappaNormalized) && ~isempty(resultsAV.kappaNormalized{a})
                    kappaNormValues = resultsAV.kappaNormalized{a}(~isnan(resultsAV.kappaNormalized{a}));
                    if ~isempty(kappaNormValues)
                        reachData.kappaNormalized{a} = [reachData.kappaNormalized{a}, kappaNormValues(1)];
                    else
                        reachData.kappaNormalized{a} = [reachData.kappaNormalized{a}, nan];
                    end
                else
                    reachData.kappaNormalized{a} = [reachData.kappaNormalized{a}, nan];
                end
            end
        else
            warning('dcc or kappa not found in AV results for session %s', sessionName);
            for a = 1:length(reachData.areas)
                reachData.dcc{a} = [reachData.dcc{a}, nan];
                reachData.dccNormalized{a} = [reachData.dccNormalized{a}, nan];
                reachData.kappa{a} = [reachData.kappa{a}, nan];
                reachData.kappaNormalized{a} = [reachData.kappaNormalized{a}, nan];
            end
        end
        
        reachData.sessionNames{end+1} = sessionName;
        fprintf('  ✓ Session completed successfully\n');
        
    catch ME
        fprintf('  ✗ Error processing session: %s\n', ME.message);
        % Add NaN for all areas
        if ~isempty(reachData.areas)
            for a = 1:length(reachData.areas)
                reachData.d2Raw{a} = [reachData.d2Raw{a}, nan];
                reachData.d2Normalized{a} = [reachData.d2Normalized{a}, nan];
                reachData.dcc{a} = [reachData.dcc{a}, nan];
                reachData.dccNormalized{a} = [reachData.dccNormalized{a}, nan];
                reachData.kappa{a} = [reachData.kappa{a}, nan];
                reachData.kappaNormalized{a} = [reachData.kappaNormalized{a}, nan];
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
fprintf('\n=== Creating Plots ===\n');

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
allD2Raw = [];
allD2Normalized = [];
allDccRaw = [];
allDccNormalized = [];
allKappaRaw = [];
allKappaNormalized = [];

for a = 1:numAreasToPlot
    areaName = areasToPlot{a};
    
    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));
    
    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end
    
    % Collect d2 values
    natD2 = spontaneousData.d2Raw{natAreaIdx};
    reachD2 = reachData.d2Raw{reachAreaIdx};
    natD2 = natD2(~isnan(natD2));
    reachD2 = reachD2(~isnan(reachD2));
    allD2Raw = [allD2Raw, natD2, reachD2];
    
    natD2Norm = spontaneousData.d2Normalized{natAreaIdx};
    reachD2Norm = reachData.d2Normalized{reachAreaIdx};
    natD2Norm = natD2Norm(~isnan(natD2Norm));
    reachD2Norm = reachD2Norm(~isnan(reachD2Norm));
    allD2Normalized = [allD2Normalized, natD2Norm, reachD2Norm];
    
    % Collect dcc values
    natDcc = spontaneousData.dcc{natAreaIdx};
    reachDcc = reachData.dcc{reachAreaIdx};
    natDcc = natDcc(~isnan(natDcc));
    reachDcc = reachDcc(~isnan(reachDcc));
    allDccRaw = [allDccRaw, natDcc, reachDcc];
    
    natDccNorm = spontaneousData.dccNormalized{natAreaIdx};
    reachDccNorm = reachData.dccNormalized{reachAreaIdx};
    natDccNorm = natDccNorm(~isnan(natDccNorm));
    reachDccNorm = reachDccNorm(~isnan(reachDccNorm));
    allDccNormalized = [allDccNormalized, natDccNorm, reachDccNorm];
    
    % Collect kappa values
    natKappa = spontaneousData.kappa{natAreaIdx};
    reachKappa = reachData.kappa{reachAreaIdx};
    natKappa = natKappa(~isnan(natKappa));
    reachKappa = reachKappa(~isnan(reachKappa));
    allKappaRaw = [allKappaRaw, natKappa, reachKappa];
    
    natKappaNorm = spontaneousData.kappaNormalized{natAreaIdx};
    reachKappaNorm = reachData.kappaNormalized{reachAreaIdx};
    natKappaNorm = natKappaNorm(~isnan(natKappaNorm));
    reachKappaNorm = reachKappaNorm(~isnan(reachKappaNorm));
    allKappaNormalized = [allKappaNormalized, natKappaNorm, reachKappaNorm];
end

% Calculate global y-axis limits with some padding
if ~isempty(allD2Raw)
    ylimD2Raw = [0, min(.25, max(allD2Raw) * 1.05)];
    if ylimD2Raw(1) == ylimD2Raw(2)
        ylimD2Raw = ylimD2Raw(1) + [-0.1, 0.1];
    end
else
    ylimD2Raw = [0, 1];
end

if ~isempty(allD2Normalized)
    ylimD2Normalized = [0, max(allD2Normalized) * 1.05];
    if ylimD2Normalized(1) == ylimD2Normalized(2)
        ylimD2Normalized = ylimD2Normalized(1) + [-0.1, 0.1];
    end
else
    ylimD2Normalized = [0, 1];
end

if ~isempty(allDccRaw)
    ylimDccRaw = [0, max(allDccRaw) * 1.05];
    if ylimDccRaw(1) == ylimDccRaw(2)
        ylimDccRaw = ylimDccRaw(1) + [-0.1, 0.1];
    end
else
    ylimDccRaw = [0, 1];
end

if ~isempty(allDccNormalized)
    ylimDccNormalized = [0, max(allDccNormalized) * 1.05];
    if ylimDccNormalized(1) == ylimDccNormalized(2)
        ylimDccNormalized = ylimDccNormalized(1) + [-0.1, 0.1];
    end
else
    ylimDccNormalized = [0, 1];
end

if ~isempty(allKappaRaw)
    ylimKappaRaw = [0, max(allKappaRaw) * 1.05];
    if ylimKappaRaw(1) == ylimKappaRaw(2)
        ylimKappaRaw = ylimKappaRaw(1) + [-0.1, 0.1];
    end
else
    ylimKappaRaw = [0, 1];
end

if ~isempty(allKappaNormalized)
    ylimKappaNormalized = [0, max(allKappaNormalized) * 1.05];
    if ylimKappaNormalized(1) == ylimKappaNormalized(2)
        ylimKappaNormalized = ylimKappaNormalized(1) + [-0.1, 0.1];
    end
else
    ylimKappaNormalized = [0, 1];
end

% Create figure with 6 columns: d2 raw, d2 normalized, dcc raw, dcc normalized, kappa raw, kappa normalized
figure(3002); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [100, 100, 3000, 300 * numAreasToPlot]);

% Create subplots for all metrics
for a = 1:numAreasToPlot
    areaName = areasToPlot{a};
    
    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));
    
    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end
    
    % Get values for this area
    natD2 = spontaneousData.d2Raw{natAreaIdx};
    reachD2 = reachData.d2Raw{reachAreaIdx};
    natD2 = natD2(~isnan(natD2));
    reachD2 = reachD2(~isnan(reachD2));
    
    natD2Norm = spontaneousData.d2Normalized{natAreaIdx};
    reachD2Norm = reachData.d2Normalized{reachAreaIdx};
    natD2Norm = natD2Norm(~isnan(natD2Norm));
    reachD2Norm = reachD2Norm(~isnan(reachD2Norm));
    
    natDcc = spontaneousData.dcc{natAreaIdx};
    reachDcc = reachData.dcc{reachAreaIdx};
    natDcc = natDcc(~isnan(natDcc));
    reachDcc = reachDcc(~isnan(reachDcc));
    
    natDccNorm = spontaneousData.dccNormalized{natAreaIdx};
    reachDccNorm = reachData.dccNormalized{reachAreaIdx};
    natDccNorm = natDccNorm(~isnan(natDccNorm));
    reachDccNorm = reachDccNorm(~isnan(reachDccNorm));
    
    natKappa = spontaneousData.kappa{natAreaIdx};
    reachKappa = reachData.kappa{reachAreaIdx};
    natKappa = natKappa(~isnan(natKappa));
    reachKappa = reachKappa(~isnan(reachKappa));
    
    natKappaNorm = spontaneousData.kappaNormalized{natAreaIdx};
    reachKappaNorm = reachData.kappaNormalized{reachAreaIdx};
    natKappaNorm = natKappaNorm(~isnan(natKappaNorm));
    reachKappaNorm = reachKappaNorm(~isnan(reachKappaNorm));
    
    % Create all 6 subplots
    % d2 Raw
    subplot(numAreasToPlot, 6, (a-1)*6 + 1);
    hold on;
    numNat = length(natD2);
    numReach = length(reachD2);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natD2, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachD2, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('d2 Raw');
    title(sprintf('%s - d2 Raw', areaName));
    grid on;
    ylim(ylimD2Raw);
    if ~isempty(natD2)
        yline(mean(natD2), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachD2)
        yline(mean(reachD2), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % d2 Normalized
    subplot(numAreasToPlot, 6, (a-1)*6 + 2);
    hold on;
    numNat = length(natD2Norm);
    numReach = length(reachD2Norm);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natD2Norm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachD2Norm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('d2 Normalized');
    title(sprintf('%s - d2 Normalized', areaName));
    grid on;
    ylim(ylimD2Normalized);
    if ~isempty(natD2Norm)
        yline(mean(natD2Norm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachD2Norm)
        yline(mean(reachD2Norm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % dcc Raw
    subplot(numAreasToPlot, 6, (a-1)*6 + 3);
    hold on;
    numNat = length(natDcc);
    numReach = length(reachDcc);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natDcc, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachDcc, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('dcc Raw');
    title(sprintf('%s - dcc Raw', areaName));
    grid on;
    ylim(ylimDccRaw);
    if ~isempty(natDcc)
        yline(mean(natDcc), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachDcc)
        yline(mean(reachDcc), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % dcc Normalized
    subplot(numAreasToPlot, 6, (a-1)*6 + 4);
    hold on;
    numNat = length(natDccNorm);
    numReach = length(reachDccNorm);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natDccNorm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachDccNorm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('dcc Normalized');
    title(sprintf('%s - dcc Normalized', areaName));
    grid on;
    ylim(ylimDccNormalized);
    if ~isempty(natDccNorm)
        yline(mean(natDccNorm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachDccNorm)
        yline(mean(reachDccNorm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % kappa Raw
    subplot(numAreasToPlot, 6, (a-1)*6 + 5);
    hold on;
    numNat = length(natKappa);
    numReach = length(reachKappa);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natKappa, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachKappa, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('kappa Raw');
    title(sprintf('%s - kappa Raw', areaName));
    grid on;
    ylim(ylimKappaRaw);
    if ~isempty(natKappa)
        yline(mean(natKappa), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachKappa)
        yline(mean(reachKappa), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % kappa Normalized
    subplot(numAreasToPlot, 6, (a-1)*6 + 6);
    hold on;
    numNat = length(natKappaNorm);
    numReach = length(reachKappaNorm);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natKappaNorm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachKappaNorm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('kappa Normalized');
    title(sprintf('%s - kappa Normalized', areaName));
    grid on;
    ylim(ylimKappaNormalized);
    if ~isempty(natKappaNorm)
        yline(mean(natKappaNorm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachKappaNorm)
        yline(mean(reachKappaNorm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
end

% Add overall title
sgtitle(sprintf('Criticality Comparison: Spontaneous vs Reach (Time Range: %.1f-%.1f s)', ...
    windowStartTime, windowEndTime), 'FontSize', 14, 'FontWeight', 'bold');

%% Save figure
saveDir = fullfile(paths.dropPath, 'sliding_window_comparisons');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

plotFilenamePng = sprintf('criticality_across_full_sessions_%.0f-%.0fs.png', windowStartTime, windowEndTime);
plotPathPng = fullfile(saveDir, plotFilenamePng);

exportgraphics(gcf, plotPathPng, 'Resolution', 300);
fprintf('\nSaved PNG plot to: %s\n', plotPathPng);

plotFilenameEps = sprintf('criticality_across_full_sessions_%.0f-%.0fs.eps', windowStartTime, windowEndTime);
plotPathEps = fullfile(saveDir, plotFilenameEps);

exportgraphics(gcf, plotPathEps, 'ContentType', 'vector');
fprintf('Saved EPS plot to: %s\n', plotPathEps);

fprintf('\n=== Analysis Complete ===\n');
