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
configAR.nDim = 5;
configAR.enablePermutations = true;
configAR.nShuffles = 20;
configAR.analyzeModulation = false;
configAR.makePlots = false;  % Don't create individual plots
configAR.saveData = false;  % Don't save individual session results
configAR.useOptimalBinWindowFunction = true;
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
spontaneousData.decades = {};  % Cell array: {areaIdx}{sessionIdx} = decades value
spontaneousData.decadesNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized decades value
spontaneousData.tau = {};  % Cell array: {areaIdx}{sessionIdx} = tau value
spontaneousData.tauNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized tau value
spontaneousData.alpha = {};  % Cell array: {areaIdx}{sessionIdx} = alpha value
spontaneousData.alphaNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized alpha value
spontaneousData.paramSD = {};  % Cell array: {areaIdx}{sessionIdx} = paramSD value
spontaneousData.paramSDNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized paramSD value
spontaneousData.sessionNames = {};
spontaneousData.areas = [];

reachData = struct();
reachData.d2Raw = {};  % Cell array: {areaIdx}{sessionIdx} = d2 value
reachData.d2Normalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized d2 value
reachData.dcc = {};  % Cell array: {areaIdx}{sessionIdx} = dcc value
reachData.dccNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized dcc value
reachData.kappa = {};  % Cell array: {areaIdx}{sessionIdx} = kappa value
reachData.kappaNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized kappa value
reachData.decades = {};  % Cell array: {areaIdx}{sessionIdx} = decades value
reachData.decadesNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized decades value
reachData.tau = {};  % Cell array: {areaIdx}{sessionIdx} = tau value
reachData.tauNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized tau value
reachData.alpha = {};  % Cell array: {areaIdx}{sessionIdx} = alpha value
reachData.alphaNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized alpha value
reachData.paramSD = {};  % Cell array: {areaIdx}{sessionIdx} = paramSD value
reachData.paramSDNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized paramSD value
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
            spontaneousData.decades = cell(1, numAreas);
            spontaneousData.decadesNormalized = cell(1, numAreas);
            spontaneousData.tau = cell(1, numAreas);
            spontaneousData.tauNormalized = cell(1, numAreas);
            spontaneousData.alpha = cell(1, numAreas);
            spontaneousData.alphaNormalized = cell(1, numAreas);
            spontaneousData.paramSD = cell(1, numAreas);
            spontaneousData.paramSDNormalized = cell(1, numAreas);
            for a = 1:numAreas
                spontaneousData.d2Raw{a} = [];
                spontaneousData.d2Normalized{a} = [];
                spontaneousData.dcc{a} = [];
                spontaneousData.dccNormalized{a} = [];
                spontaneousData.kappa{a} = [];
                spontaneousData.kappaNormalized{a} = [];
                spontaneousData.decades{a} = [];
                spontaneousData.decadesNormalized{a} = [];
                spontaneousData.tau{a} = [];
                spontaneousData.tauNormalized{a} = [];
                spontaneousData.alpha{a} = [];
                spontaneousData.alphaNormalized{a} = [];
                spontaneousData.paramSD{a} = [];
                spontaneousData.paramSDNormalized{a} = [];
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
                
                % Extract decades raw
                if a <= length(resultsAV.decades) && ~isempty(resultsAV.decades{a})
                    decadesValues = resultsAV.decades{a}(~isnan(resultsAV.decades{a}));
                    if ~isempty(decadesValues)
                        spontaneousData.decades{a} = [spontaneousData.decades{a}, decadesValues(1)];
                    else
                        spontaneousData.decades{a} = [spontaneousData.decades{a}, nan];
                    end
                else
                    spontaneousData.decades{a} = [spontaneousData.decades{a}, nan];
                end
                
                % Extract decades normalized
                if a <= length(resultsAV.decadesNormalized) && ~isempty(resultsAV.decadesNormalized{a})
                    decadesNormValues = resultsAV.decadesNormalized{a}(~isnan(resultsAV.decadesNormalized{a}));
                    if ~isempty(decadesNormValues)
                        spontaneousData.decadesNormalized{a} = [spontaneousData.decadesNormalized{a}, decadesNormValues(1)];
                    else
                        spontaneousData.decadesNormalized{a} = [spontaneousData.decadesNormalized{a}, nan];
                    end
                else
                    spontaneousData.decadesNormalized{a} = [spontaneousData.decadesNormalized{a}, nan];
                end
                
                % Extract tau raw
                if a <= length(resultsAV.tau) && ~isempty(resultsAV.tau{a})
                    tauValues = resultsAV.tau{a}(~isnan(resultsAV.tau{a}));
                    if ~isempty(tauValues)
                        spontaneousData.tau{a} = [spontaneousData.tau{a}, tauValues(1)];
                    else
                        spontaneousData.tau{a} = [spontaneousData.tau{a}, nan];
                    end
                else
                    spontaneousData.tau{a} = [spontaneousData.tau{a}, nan];
                end
                
                % Extract tau normalized
                if a <= length(resultsAV.tauNormalized) && ~isempty(resultsAV.tauNormalized{a})
                    tauNormValues = resultsAV.tauNormalized{a}(~isnan(resultsAV.tauNormalized{a}));
                    if ~isempty(tauNormValues)
                        spontaneousData.tauNormalized{a} = [spontaneousData.tauNormalized{a}, tauNormValues(1)];
                    else
                        spontaneousData.tauNormalized{a} = [spontaneousData.tauNormalized{a}, nan];
                    end
                else
                    spontaneousData.tauNormalized{a} = [spontaneousData.tauNormalized{a}, nan];
                end
                
                % Extract alpha raw
                if a <= length(resultsAV.alpha) && ~isempty(resultsAV.alpha{a})
                    alphaValues = resultsAV.alpha{a}(~isnan(resultsAV.alpha{a}));
                    if ~isempty(alphaValues)
                        spontaneousData.alpha{a} = [spontaneousData.alpha{a}, alphaValues(1)];
                    else
                        spontaneousData.alpha{a} = [spontaneousData.alpha{a}, nan];
                    end
                else
                    spontaneousData.alpha{a} = [spontaneousData.alpha{a}, nan];
                end
                
                % Extract alpha normalized
                if a <= length(resultsAV.alphaNormalized) && ~isempty(resultsAV.alphaNormalized{a})
                    alphaNormValues = resultsAV.alphaNormalized{a}(~isnan(resultsAV.alphaNormalized{a}));
                    if ~isempty(alphaNormValues)
                        spontaneousData.alphaNormalized{a} = [spontaneousData.alphaNormalized{a}, alphaNormValues(1)];
                    else
                        spontaneousData.alphaNormalized{a} = [spontaneousData.alphaNormalized{a}, nan];
                    end
                else
                    spontaneousData.alphaNormalized{a} = [spontaneousData.alphaNormalized{a}, nan];
                end
                
                % Extract paramSD raw
                if a <= length(resultsAV.paramSD) && ~isempty(resultsAV.paramSD{a})
                    paramSDValues = resultsAV.paramSD{a}(~isnan(resultsAV.paramSD{a}));
                    if ~isempty(paramSDValues)
                        spontaneousData.paramSD{a} = [spontaneousData.paramSD{a}, paramSDValues(1)];
                    else
                        spontaneousData.paramSD{a} = [spontaneousData.paramSD{a}, nan];
                    end
                else
                    spontaneousData.paramSD{a} = [spontaneousData.paramSD{a}, nan];
                end
                
                % Extract paramSD normalized
                if a <= length(resultsAV.paramSDNormalized) && ~isempty(resultsAV.paramSDNormalized{a})
                    paramSDNormValues = resultsAV.paramSDNormalized{a}(~isnan(resultsAV.paramSDNormalized{a}));
                    if ~isempty(paramSDNormValues)
                        spontaneousData.paramSDNormalized{a} = [spontaneousData.paramSDNormalized{a}, paramSDNormValues(1)];
                    else
                        spontaneousData.paramSDNormalized{a} = [spontaneousData.paramSDNormalized{a}, nan];
                    end
                else
                    spontaneousData.paramSDNormalized{a} = [spontaneousData.paramSDNormalized{a}, nan];
                end
            end
        else
            warning('dcc or kappa not found in AV results for session %s', sessionName);
            for a = 1:length(spontaneousData.areas)
                spontaneousData.dcc{a} = [spontaneousData.dcc{a}, nan];
                spontaneousData.dccNormalized{a} = [spontaneousData.dccNormalized{a}, nan];
                spontaneousData.kappa{a} = [spontaneousData.kappa{a}, nan];
                spontaneousData.kappaNormalized{a} = [spontaneousData.kappaNormalized{a}, nan];
                spontaneousData.decades{a} = [spontaneousData.decades{a}, nan];
                spontaneousData.decadesNormalized{a} = [spontaneousData.decadesNormalized{a}, nan];
                spontaneousData.tau{a} = [spontaneousData.tau{a}, nan];
                spontaneousData.tauNormalized{a} = [spontaneousData.tauNormalized{a}, nan];
                spontaneousData.alpha{a} = [spontaneousData.alpha{a}, nan];
                spontaneousData.alphaNormalized{a} = [spontaneousData.alphaNormalized{a}, nan];
                spontaneousData.paramSD{a} = [spontaneousData.paramSD{a}, nan];
                spontaneousData.paramSDNormalized{a} = [spontaneousData.paramSDNormalized{a}, nan];
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
                spontaneousData.decades{a} = [spontaneousData.decades{a}, nan];
                spontaneousData.decadesNormalized{a} = [spontaneousData.decadesNormalized{a}, nan];
                spontaneousData.tau{a} = [spontaneousData.tau{a}, nan];
                spontaneousData.tauNormalized{a} = [spontaneousData.tauNormalized{a}, nan];
                spontaneousData.alpha{a} = [spontaneousData.alpha{a}, nan];
                spontaneousData.alphaNormalized{a} = [spontaneousData.alphaNormalized{a}, nan];
                spontaneousData.paramSD{a} = [spontaneousData.paramSD{a}, nan];
                spontaneousData.paramSDNormalized{a} = [spontaneousData.paramSDNormalized{a}, nan];
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
            reachData.decades = cell(1, numAreas);
            reachData.decadesNormalized = cell(1, numAreas);
            reachData.tau = cell(1, numAreas);
            reachData.tauNormalized = cell(1, numAreas);
            reachData.alpha = cell(1, numAreas);
            reachData.alphaNormalized = cell(1, numAreas);
            reachData.paramSD = cell(1, numAreas);
            reachData.paramSDNormalized = cell(1, numAreas);
            for a = 1:numAreas
                reachData.d2Raw{a} = [];
                reachData.d2Normalized{a} = [];
                reachData.dcc{a} = [];
                reachData.dccNormalized{a} = [];
                reachData.kappa{a} = [];
                reachData.kappaNormalized{a} = [];
                reachData.decades{a} = [];
                reachData.decadesNormalized{a} = [];
                reachData.tau{a} = [];
                reachData.tauNormalized{a} = [];
                reachData.alpha{a} = [];
                reachData.alphaNormalized{a} = [];
                reachData.paramSD{a} = [];
                reachData.paramSDNormalized{a} = [];
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
                
                % Extract decades raw
                if a <= length(resultsAV.decades) && ~isempty(resultsAV.decades{a})
                    decadesValues = resultsAV.decades{a}(~isnan(resultsAV.decades{a}));
                    if ~isempty(decadesValues)
                        reachData.decades{a} = [reachData.decades{a}, decadesValues(1)];
                    else
                        reachData.decades{a} = [reachData.decades{a}, nan];
                    end
                else
                    reachData.decades{a} = [reachData.decades{a}, nan];
                end
                
                % Extract decades normalized
                if a <= length(resultsAV.decadesNormalized) && ~isempty(resultsAV.decadesNormalized{a})
                    decadesNormValues = resultsAV.decadesNormalized{a}(~isnan(resultsAV.decadesNormalized{a}));
                    if ~isempty(decadesNormValues)
                        reachData.decadesNormalized{a} = [reachData.decadesNormalized{a}, decadesNormValues(1)];
                    else
                        reachData.decadesNormalized{a} = [reachData.decadesNormalized{a}, nan];
                    end
                else
                    reachData.decadesNormalized{a} = [reachData.decadesNormalized{a}, nan];
                end
                
                % Extract tau raw
                if a <= length(resultsAV.tau) && ~isempty(resultsAV.tau{a})
                    tauValues = resultsAV.tau{a}(~isnan(resultsAV.tau{a}));
                    if ~isempty(tauValues)
                        reachData.tau{a} = [reachData.tau{a}, tauValues(1)];
                    else
                        reachData.tau{a} = [reachData.tau{a}, nan];
                    end
                else
                    reachData.tau{a} = [reachData.tau{a}, nan];
                end
                
                % Extract tau normalized
                if a <= length(resultsAV.tauNormalized) && ~isempty(resultsAV.tauNormalized{a})
                    tauNormValues = resultsAV.tauNormalized{a}(~isnan(resultsAV.tauNormalized{a}));
                    if ~isempty(tauNormValues)
                        reachData.tauNormalized{a} = [reachData.tauNormalized{a}, tauNormValues(1)];
                    else
                        reachData.tauNormalized{a} = [reachData.tauNormalized{a}, nan];
                    end
                else
                    reachData.tauNormalized{a} = [reachData.tauNormalized{a}, nan];
                end
                
                % Extract alpha raw
                if a <= length(resultsAV.alpha) && ~isempty(resultsAV.alpha{a})
                    alphaValues = resultsAV.alpha{a}(~isnan(resultsAV.alpha{a}));
                    if ~isempty(alphaValues)
                        reachData.alpha{a} = [reachData.alpha{a}, alphaValues(1)];
                    else
                        reachData.alpha{a} = [reachData.alpha{a}, nan];
                    end
                else
                    reachData.alpha{a} = [reachData.alpha{a}, nan];
                end
                
                % Extract alpha normalized
                if a <= length(resultsAV.alphaNormalized) && ~isempty(resultsAV.alphaNormalized{a})
                    alphaNormValues = resultsAV.alphaNormalized{a}(~isnan(resultsAV.alphaNormalized{a}));
                    if ~isempty(alphaNormValues)
                        reachData.alphaNormalized{a} = [reachData.alphaNormalized{a}, alphaNormValues(1)];
                    else
                        reachData.alphaNormalized{a} = [reachData.alphaNormalized{a}, nan];
                    end
                else
                    reachData.alphaNormalized{a} = [reachData.alphaNormalized{a}, nan];
                end
                
                % Extract paramSD raw
                if a <= length(resultsAV.paramSD) && ~isempty(resultsAV.paramSD{a})
                    paramSDValues = resultsAV.paramSD{a}(~isnan(resultsAV.paramSD{a}));
                    if ~isempty(paramSDValues)
                        reachData.paramSD{a} = [reachData.paramSD{a}, paramSDValues(1)];
                    else
                        reachData.paramSD{a} = [reachData.paramSD{a}, nan];
                    end
                else
                    reachData.paramSD{a} = [reachData.paramSD{a}, nan];
                end
                
                % Extract paramSD normalized
                if a <= length(resultsAV.paramSDNormalized) && ~isempty(resultsAV.paramSDNormalized{a})
                    paramSDNormValues = resultsAV.paramSDNormalized{a}(~isnan(resultsAV.paramSDNormalized{a}));
                    if ~isempty(paramSDNormValues)
                        reachData.paramSDNormalized{a} = [reachData.paramSDNormalized{a}, paramSDNormValues(1)];
                    else
                        reachData.paramSDNormalized{a} = [reachData.paramSDNormalized{a}, nan];
                    end
                else
                    reachData.paramSDNormalized{a} = [reachData.paramSDNormalized{a}, nan];
                end
            end
        else
            warning('dcc or kappa not found in AV results for session %s', sessionName);
            for a = 1:length(reachData.areas)
                reachData.dcc{a} = [reachData.dcc{a}, nan];
                reachData.dccNormalized{a} = [reachData.dccNormalized{a}, nan];
                reachData.kappa{a} = [reachData.kappa{a}, nan];
                reachData.kappaNormalized{a} = [reachData.kappaNormalized{a}, nan];
                reachData.decades{a} = [reachData.decades{a}, nan];
                reachData.decadesNormalized{a} = [reachData.decadesNormalized{a}, nan];
                reachData.tau{a} = [reachData.tau{a}, nan];
                reachData.tauNormalized{a} = [reachData.tauNormalized{a}, nan];
                reachData.alpha{a} = [reachData.alpha{a}, nan];
                reachData.alphaNormalized{a} = [reachData.alphaNormalized{a}, nan];
                reachData.paramSD{a} = [reachData.paramSD{a}, nan];
                reachData.paramSDNormalized{a} = [reachData.paramSDNormalized{a}, nan];
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
                reachData.decades{a} = [reachData.decades{a}, nan];
                reachData.decadesNormalized{a} = [reachData.decadesNormalized{a}, nan];
                reachData.tau{a} = [reachData.tau{a}, nan];
                reachData.tauNormalized{a} = [reachData.tauNormalized{a}, nan];
                reachData.alpha{a} = [reachData.alpha{a}, nan];
                reachData.alphaNormalized{a} = [reachData.alphaNormalized{a}, nan];
                reachData.paramSD{a} = [reachData.paramSD{a}, nan];
                reachData.paramSDNormalized{a} = [reachData.paramSDNormalized{a}, nan];
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
% Buffer is 5% of the range
bufferPct = 0.05;  % 5% buffer

if ~isempty(allD2Raw)
    yMin = min(allD2Raw);
    yMax = min(.25, max(allD2Raw));
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimD2Raw = [yMin - buffer, yMax + buffer];
else
    ylimD2Raw = [0, 1];
end

if ~isempty(allD2Normalized)
    yMin = min(allD2Normalized);
    yMax = max(allD2Normalized);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimD2Normalized = [yMin - buffer, yMax + buffer];
else
    ylimD2Normalized = [0, 1];
end

if ~isempty(allDccRaw)
    yMin = min(allDccRaw);
    yMax = max(allDccRaw);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimDccRaw = [yMin - buffer, yMax + buffer];
else
    ylimDccRaw = [0, 1];
end

if ~isempty(allDccNormalized)
    yMin = min(allDccNormalized);
    yMax = max(allDccNormalized);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimDccNormalized = [yMin - buffer, yMax + buffer];
else
    ylimDccNormalized = [0, 1];
end

if ~isempty(allKappaRaw)
    yMin = min(allKappaRaw);
    yMax = max(allKappaRaw);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimKappaRaw = [yMin - buffer, yMax + buffer];
else
    ylimKappaRaw = [0, 1];
end

if ~isempty(allKappaNormalized)
    yMin = min(allKappaNormalized);
    yMax = max(allKappaNormalized);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimKappaNormalized = [yMin - buffer, yMax + buffer];
else
    ylimKappaNormalized = [0, 1];
end

% Detect monitors and size figure to full screen (prefer second monitor if present)
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);
if size(monitorPositions, 1) >= 2
    targetPos = monitorTwo;
else
    targetPos = monitorOne;
end

% Create figure with 6 columns: d2 raw, d2 normalized, dcc raw, dcc normalized, kappa raw, kappa normalized
figure(3002); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', targetPos);

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
titleStr = sprintf('Criticality Comparison: Spontaneous vs Reach (Time Range: %.1f-%.1f s)', ...
    windowStartTime, windowEndTime);
if configAR.pcaFlag
    titleStr = sprintf('%s [AR: PCA, nDim=%d]', titleStr, configAR.nDim);
end
if configAV.pcaFlag
    titleStr = sprintf('%s [AV: PCA, nDim=%d]', titleStr, configAV.nDim);
end
sgtitle(titleStr, 'FontSize', 14, 'FontWeight', 'bold');

%% Create second figure with decades, tau, alpha, and paramSD
fprintf('\n=== Creating Second Figure (decades, tau, alpha, paramSD) ===\n');

% Collect all values to determine global y-axis limits
allDecadesRaw = [];
allDecadesNormalized = [];
allTauRaw = [];
allTauNormalized = [];
allAlphaRaw = [];
allAlphaNormalized = [];
allParamSDRaw = [];
allParamSDNormalized = [];

for a = 1:numAreasToPlot
    areaName = areasToPlot{a};
    
    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));
    
    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end
    
    % Collect decades values
    natDecades = spontaneousData.decades{natAreaIdx};
    reachDecades = reachData.decades{reachAreaIdx};
    natDecades = natDecades(~isnan(natDecades));
    reachDecades = reachDecades(~isnan(reachDecades));
    allDecadesRaw = [allDecadesRaw, natDecades, reachDecades];
    
    natDecadesNorm = spontaneousData.decadesNormalized{natAreaIdx};
    reachDecadesNorm = reachData.decadesNormalized{reachAreaIdx};
    natDecadesNorm = natDecadesNorm(~isnan(natDecadesNorm));
    reachDecadesNorm = reachDecadesNorm(~isnan(reachDecadesNorm));
    allDecadesNormalized = [allDecadesNormalized, natDecadesNorm, reachDecadesNorm];
    
    % Collect tau values
    natTau = spontaneousData.tau{natAreaIdx};
    reachTau = reachData.tau{reachAreaIdx};
    natTau = natTau(~isnan(natTau));
    reachTau = reachTau(~isnan(reachTau));
    allTauRaw = [allTauRaw, natTau, reachTau];
    
    natTauNorm = spontaneousData.tauNormalized{natAreaIdx};
    reachTauNorm = reachData.tauNormalized{reachAreaIdx};
    natTauNorm = natTauNorm(~isnan(natTauNorm));
    reachTauNorm = reachTauNorm(~isnan(reachTauNorm));
    allTauNormalized = [allTauNormalized, natTauNorm, reachTauNorm];
    
    % Collect alpha values
    natAlpha = spontaneousData.alpha{natAreaIdx};
    reachAlpha = reachData.alpha{reachAreaIdx};
    natAlpha = natAlpha(~isnan(natAlpha));
    reachAlpha = reachAlpha(~isnan(reachAlpha));
    allAlphaRaw = [allAlphaRaw, natAlpha, reachAlpha];
    
    natAlphaNorm = spontaneousData.alphaNormalized{natAreaIdx};
    reachAlphaNorm = reachData.alphaNormalized{reachAreaIdx};
    natAlphaNorm = natAlphaNorm(~isnan(natAlphaNorm));
    reachAlphaNorm = reachAlphaNorm(~isnan(reachAlphaNorm));
    allAlphaNormalized = [allAlphaNormalized, natAlphaNorm, reachAlphaNorm];
    
    % Collect paramSD values
    natParamSD = spontaneousData.paramSD{natAreaIdx};
    reachParamSD = reachData.paramSD{reachAreaIdx};
    natParamSD = natParamSD(~isnan(natParamSD));
    reachParamSD = reachParamSD(~isnan(reachParamSD));
    allParamSDRaw = [allParamSDRaw, natParamSD, reachParamSD];
    
    natParamSDNorm = spontaneousData.paramSDNormalized{natAreaIdx};
    reachParamSDNorm = reachData.paramSDNormalized{reachAreaIdx};
    natParamSDNorm = natParamSDNorm(~isnan(natParamSDNorm));
    reachParamSDNorm = reachParamSDNorm(~isnan(reachParamSDNorm));
    allParamSDNormalized = [allParamSDNormalized, natParamSDNorm, reachParamSDNorm];
end

% Calculate global y-axis limits with some padding
% Buffer is 5% of the range
bufferPct = 0.05;  % 5% buffer

if ~isempty(allDecadesRaw)
    yMin = min(allDecadesRaw);
    yMax = max(allDecadesRaw);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimDecadesRaw = [yMin - buffer, yMax + buffer];
else
    ylimDecadesRaw = [0, 1];
end

if ~isempty(allDecadesNormalized)
    yMin = min(allDecadesNormalized);
    yMax = max(allDecadesNormalized);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimDecadesNormalized = [yMin - buffer, yMax + buffer];
else
    ylimDecadesNormalized = [0, 1];
end

if ~isempty(allTauRaw)
    yMin = min(allTauRaw);
    yMax = max(allTauRaw);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimTauRaw = [yMin - buffer, yMax + buffer];
else
    ylimTauRaw = [0, 1];
end

if ~isempty(allTauNormalized)
    yMin = min(allTauNormalized);
    yMax = max(allTauNormalized);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimTauNormalized = [yMin - buffer, yMax + buffer];
else
    ylimTauNormalized = [0, 1];
end

if ~isempty(allAlphaRaw)
    yMin = min(allAlphaRaw);
    yMax = max(allAlphaRaw);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimAlphaRaw = [yMin - buffer, yMax + buffer];
else
    ylimAlphaRaw = [0, 1];
end

if ~isempty(allAlphaNormalized)
    yMin = min(allAlphaNormalized);
    yMax = max(allAlphaNormalized);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimAlphaNormalized = [yMin - buffer, yMax + buffer];
else
    ylimAlphaNormalized = [0, 1];
end

if ~isempty(allParamSDRaw)
    yMin = min(allParamSDRaw);
    yMax = max(allParamSDRaw);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimParamSDRaw = [yMin - buffer, yMax + buffer];
else
    ylimParamSDRaw = [0, 1];
end

if ~isempty(allParamSDNormalized)
    yMin = min(allParamSDNormalized);
    yMax = max(allParamSDNormalized);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 0.1;  % Avoid zero range
    end
    buffer = bufferPct * yRange;
    ylimParamSDNormalized = [yMin - buffer, yMax + buffer];
else
    ylimParamSDNormalized = [0, 1];
end

% Create figure with 8 columns: decades raw, decades normalized, tau raw, tau normalized, alpha raw, alpha normalized, paramSD raw, paramSD normalized
figure(3003); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', targetPos);

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
    natDecades = spontaneousData.decades{natAreaIdx};
    reachDecades = reachData.decades{reachAreaIdx};
    natDecades = natDecades(~isnan(natDecades));
    reachDecades = reachDecades(~isnan(reachDecades));
    
    natDecadesNorm = spontaneousData.decadesNormalized{natAreaIdx};
    reachDecadesNorm = reachData.decadesNormalized{reachAreaIdx};
    natDecadesNorm = natDecadesNorm(~isnan(natDecadesNorm));
    reachDecadesNorm = reachDecadesNorm(~isnan(reachDecadesNorm));
    
    natTau = spontaneousData.tau{natAreaIdx};
    reachTau = reachData.tau{reachAreaIdx};
    natTau = natTau(~isnan(natTau));
    reachTau = reachTau(~isnan(reachTau));
    
    natTauNorm = spontaneousData.tauNormalized{natAreaIdx};
    reachTauNorm = reachData.tauNormalized{reachAreaIdx};
    natTauNorm = natTauNorm(~isnan(natTauNorm));
    reachTauNorm = reachTauNorm(~isnan(reachTauNorm));
    
    natAlpha = spontaneousData.alpha{natAreaIdx};
    reachAlpha = reachData.alpha{reachAreaIdx};
    natAlpha = natAlpha(~isnan(natAlpha));
    reachAlpha = reachAlpha(~isnan(reachAlpha));
    
    natAlphaNorm = spontaneousData.alphaNormalized{natAreaIdx};
    reachAlphaNorm = reachData.alphaNormalized{reachAreaIdx};
    natAlphaNorm = natAlphaNorm(~isnan(natAlphaNorm));
    reachAlphaNorm = reachAlphaNorm(~isnan(reachAlphaNorm));
    
    natParamSD = spontaneousData.paramSD{natAreaIdx};
    reachParamSD = reachData.paramSD{reachAreaIdx};
    natParamSD = natParamSD(~isnan(natParamSD));
    reachParamSD = reachParamSD(~isnan(reachParamSD));
    
    natParamSDNorm = spontaneousData.paramSDNormalized{natAreaIdx};
    reachParamSDNorm = reachData.paramSDNormalized{reachAreaIdx};
    natParamSDNorm = natParamSDNorm(~isnan(natParamSDNorm));
    reachParamSDNorm = reachParamSDNorm(~isnan(reachParamSDNorm));
    
    % Create all 8 subplots
    % decades Raw
    subplot(numAreasToPlot, 8, (a-1)*8 + 1);
    hold on;
    numNat = length(natDecades);
    numReach = length(reachDecades);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natDecades, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachDecades, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('decades Raw');
    title(sprintf('%s - decades Raw', areaName));
    grid on;
    ylim(ylimDecadesRaw);
    if ~isempty(natDecades)
        yline(mean(natDecades), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachDecades)
        yline(mean(reachDecades), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % decades Normalized
    subplot(numAreasToPlot, 8, (a-1)*8 + 2);
    hold on;
    numNat = length(natDecadesNorm);
    numReach = length(reachDecadesNorm);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natDecadesNorm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachDecadesNorm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('decades Normalized');
    title(sprintf('%s - decades Normalized', areaName));
    grid on;
    ylim(ylimDecadesNormalized);
    if ~isempty(natDecadesNorm)
        yline(mean(natDecadesNorm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachDecadesNorm)
        yline(mean(reachDecadesNorm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % tau Raw
    subplot(numAreasToPlot, 8, (a-1)*8 + 3);
    hold on;
    numNat = length(natTau);
    numReach = length(reachTau);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natTau, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachTau, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('tau Raw');
    title(sprintf('%s - tau Raw', areaName));
    grid on;
    ylim(ylimTauRaw);
    if ~isempty(natTau)
        yline(mean(natTau), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachTau)
        yline(mean(reachTau), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % tau Normalized
    subplot(numAreasToPlot, 8, (a-1)*8 + 4);
    hold on;
    numNat = length(natTauNorm);
    numReach = length(reachTauNorm);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natTauNorm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachTauNorm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('tau Normalized');
    title(sprintf('%s - tau Normalized', areaName));
    grid on;
    ylim(ylimTauNormalized);
    if ~isempty(natTauNorm)
        yline(mean(natTauNorm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachTauNorm)
        yline(mean(reachTauNorm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % alpha Raw
    subplot(numAreasToPlot, 8, (a-1)*8 + 5);
    hold on;
    numNat = length(natAlpha);
    numReach = length(reachAlpha);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natAlpha, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachAlpha, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('alpha Raw');
    title(sprintf('%s - alpha Raw', areaName));
    grid on;
    ylim(ylimAlphaRaw);
    if ~isempty(natAlpha)
        yline(mean(natAlpha), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachAlpha)
        yline(mean(reachAlpha), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % alpha Normalized
    subplot(numAreasToPlot, 8, (a-1)*8 + 6);
    hold on;
    numNat = length(natAlphaNorm);
    numReach = length(reachAlphaNorm);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natAlphaNorm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachAlphaNorm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('alpha Normalized');
    title(sprintf('%s - alpha Normalized', areaName));
    grid on;
    ylim(ylimAlphaNormalized);
    if ~isempty(natAlphaNorm)
        yline(mean(natAlphaNorm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachAlphaNorm)
        yline(mean(reachAlphaNorm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % paramSD Raw
    subplot(numAreasToPlot, 8, (a-1)*8 + 7);
    hold on;
    numNat = length(natParamSD);
    numReach = length(reachParamSD);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natParamSD, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachParamSD, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('paramSD Raw');
    title(sprintf('%s - paramSD Raw', areaName));
    grid on;
    ylim(ylimParamSDRaw);
    if ~isempty(natParamSD)
        yline(mean(natParamSD), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachParamSD)
        yline(mean(reachParamSD), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
    
    % paramSD Normalized
    subplot(numAreasToPlot, 8, (a-1)*8 + 8);
    hold on;
    numNat = length(natParamSDNorm);
    numReach = length(reachParamSDNorm);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natParamSDNorm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachParamSDNorm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    ylabel('paramSD Normalized');
    title(sprintf('%s - paramSD Normalized', areaName));
    grid on;
    ylim(ylimParamSDNormalized);
    if ~isempty(natParamSDNorm)
        yline(mean(natParamSDNorm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachParamSDNorm)
        yline(mean(reachParamSDNorm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
end

% Add overall title
titleStr2 = sprintf('Criticality Comparison: Spontaneous vs Reach - decades, tau, alpha, paramSD (Time Range: %.1f-%.1f s)', ...
    windowStartTime, windowEndTime);
if configAR.pcaFlag
    titleStr2 = sprintf('%s [AR: PCA, nDim=%d]', titleStr2, configAR.nDim);
end
if configAV.pcaFlag
    titleStr2 = sprintf('%s [AV: PCA, nDim=%d]', titleStr2, configAV.nDim);
end
sgtitle(titleStr2, 'FontSize', 14, 'FontWeight', 'bold');

%% Save figures
saveDir = fullfile(paths.dropPath, 'sliding_window_comparisons');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

% Create filename suffix based on PCA flags
filenameSuffix = '';
if configAR.pcaFlag
    filenameSuffix = sprintf('%s_pca%d', filenameSuffix, configAR.nDim);
end
if configAV.pcaFlag
    if isempty(filenameSuffix)
        filenameSuffix = sprintf('_pca%d', configAV.nDim);
    else
        filenameSuffix = sprintf('%s_av_pca%d', filenameSuffix, configAV.nDim);
    end
end

% Save first figure (d2, dcc, kappa)
figure(3002);
plotFilenamePng = sprintf('criticality_across_full_sessions_%.0f-%.0fs%s.png', windowStartTime, windowEndTime, filenameSuffix);
plotPathPng = fullfile(saveDir, plotFilenamePng);

exportgraphics(gcf, plotPathPng, 'Resolution', 300);
fprintf('\nSaved PNG plot (d2, dcc, kappa) to: %s\n', plotPathPng);

plotFilenameEps = sprintf('criticality_across_full_sessions_%.0f-%.0fs%s.eps', windowStartTime, windowEndTime, filenameSuffix);
plotPathEps = fullfile(saveDir, plotFilenameEps);

exportgraphics(gcf, plotPathEps, 'ContentType', 'vector');
fprintf('Saved EPS plot (d2, dcc, kappa) to: %s\n', plotPathEps);

% Save second figure (decades, tau, paramSD)
figure(3003);
plotFilenamePng2 = sprintf('criticality_across_full_sessions_decades_tau_paramSD_%.0f-%.0fs%s.png', windowStartTime, windowEndTime, filenameSuffix);
plotPathPng2 = fullfile(saveDir, plotFilenamePng2);

exportgraphics(gcf, plotPathPng2, 'Resolution', 300);
fprintf('Saved PNG plot (decades, tau, paramSD) to: %s\n', plotPathPng2);

plotFilenameEps2 = sprintf('criticality_across_full_sessions_decades_tau_paramSD_%.0f-%.0fs%s.eps', windowStartTime, windowEndTime, filenameSuffix);
plotPathEps2 = fullfile(saveDir, plotFilenameEps2);

exportgraphics(gcf, plotPathEps2, 'ContentType', 'vector');
fprintf('Saved EPS plot (decades, tau, paramSD) to: %s\n', plotPathEps2);

fprintf('\n=== Analysis Complete ===\n');
