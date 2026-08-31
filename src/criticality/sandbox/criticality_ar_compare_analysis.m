function results = criticality_ar_compare_analysis(dataStruct, config)
% CRITICALITY_AR_COMPARE_ANALYSIS Compare Euclidean d2 vs KL-rate d2
%
% Variables:
%   dataStruct - Data structure from load_sliding_window_data()
%   config     - Configuration structure (same core fields as
%                criticality_ar_analysis, plus):
%     .klFitMethod  - 'MaxLikelihood' (needed for error bars) or 'YuleWalker'
%     .klErrBars    - If true, compute S2.5 Hessian error bars (slow)
%     .klParallel   - If true, parfor inside calc_db gradient (error bars)
%
% Goal:
%   Sandbox comparison of the current Euclidean d2 (myYuleWalker3 +
%   getFixedPointDistance2) against the Sooter et al. KL-rate d2 from
%   prox_crit_toolkit (calc_db), including optional Delta d2 error bars.
%   Windowing / binning / SDF / PCA match criticality_ar_analysis; modulation
%   and subsampling are omitted to keep this testbed focused.

    sandboxDir = fileparts(mfilename('fullpath'));
    criticalityDir = fileparts(sandboxDir);
    srcRoot = fileparts(criticalityDir);
    addpath(srcRoot);
    add_figure_tools_path();
    addpath(fullfile(srcRoot, 'criticality'));
    addpath(fullfile(srcRoot, 'criticality', 'analyses'));
    addpath(fullfile(srcRoot, 'sliding_window_prep', 'utils'));
    addpath(fullfile(srcRoot, 'session_prep', 'utils'));
    addpath(fullfile(srcRoot, 'data_prep'));
    add_compare_toolbox_paths(srcRoot);

    validate_workspace_vars({'sessionType', 'spikeTimes', 'spikeClusters', 'areas', 'idMatIdx'}, ...
        dataStruct, 'errorMsg', 'Required field', 'source', 'load_sliding_window_data');

    if nargin < 2 || isempty(config) || ~isstruct(config)
        config = struct();
    end
    config = set_compare_config_defaults(config);
    if config.sdfFlag
        if isempty(config.sdfSigmaMs) || ~isfinite(config.sdfSigmaMs) || config.sdfSigmaMs <= 0
            error('config.sdfSigmaMs must be a positive scalar when config.sdfFlag is true.');
        end
    end
    if config.klErrBars && ~strcmp(config.klFitMethod, 'MaxLikelihood')
        error(['KL error bars require config.klFitMethod = ''MaxLikelihood'' ', ...
            '(S2.5 Hessian is not defined for Yule-Walker).']);
    end

    sessionType = dataStruct.sessionType;
    areas = dataStruct.areas;
    numAreas = length(areas);

    if isfield(dataStruct, 'areasToTest')
        areasToTest = dataStruct.areasToTest;
    else
        areasToTest = 1:numAreas;
    end

    if isfield(config, 'brainAreas') && ~isempty(config.brainAreas)
        if ischar(config.brainAreas)
            desiredAreas = {config.brainAreas};
        else
            desiredAreas = config.brainAreas;
        end
        selectedIdx = [];
        for iArea = 1:numel(desiredAreas)
            thisName = desiredAreas{iArea};
            idx = find(strcmp(areas, thisName));
            if isempty(idx)
                fprintf('Warning: requested brain area "%s" not found. Skipping.\n', thisName);
            else
                selectedIdx = [selectedIdx, idx(:)']; %#ok<AGROW>
            end
        end
        selectedIdx = unique(selectedIdx, 'stable');
        if isempty(selectedIdx)
            error('config.brainAreas was specified but none of the requested areas were found.');
        end
        areasToTest = selectedIdx;
        fprintf('Restricting analysis to brainAreas: %s\n', strjoin(areas(areasToTest), ', '));
    end

    fprintf('\n=== Criticality AR Compare Setup ===\n');
    fprintf('Data type: %s\n', sessionType);
    fprintf('Window size: %.2f s, step: %.3f s\n', config.slidingWindowSize, config.stepSize);
    fprintf('Old d2: Euclidean (Yule-Walker). New d2: KL-rate (%s), errBars=%d\n', ...
        config.klFitMethod, config.klErrBars);

    filenameSuffix = format_ar_file_suffix(config);
    filenameSuffix = [filenameSuffix, '_klcompare'];

    if ~isfield(config, 'saveDir') || isempty(config.saveDir)
        config.saveDir = dataStruct.saveDir;
    end

    sessionNameForPath = '';
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        sessionNameForPath = dataStruct.sessionName;
    end

    resultsPath = create_results_path('criticality_ar_compare', sessionType, ...
        sessionNameForPath, config.saveDir, 'filenameSuffix', filenameSuffix);

    fprintf('\n--- Using spike times for on-demand binning ---\n');
    if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
        timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
    else
        timeRange = [0, max(dataStruct.spikeTimes)];
    end

    [binSize, slidingWindowSize] = find_compare_bin_window(dataStruct, config, areasToTest, timeRange);
    slidingWindowSize(slidingWindowSize < 10) = 10;

    if ~isfield(config, 'stepSize') || isempty(config.stepSize)
        error('stepSize must be provided in config');
    end

    totalTime = timeRange(2) - timeRange(1);
    windowToleranceSec = 1;
    if config.slidingWindowSize > (totalTime + windowToleranceSec)
        fprintf(['  slidingWindowSize %.1f s exceeds session duration %.1f s; ', ...
            'using full session for d2 window.\n'], config.slidingWindowSize, totalTime);
        config.slidingWindowSize = totalTime;
        config.stepSize = totalTime;
    end
    if config.stepSize > (totalTime + windowToleranceSec)
        config.stepSize = totalTime;
    end
    oversizedMask = isfinite(slidingWindowSize) & ...
        (slidingWindowSize > (totalTime + windowToleranceSec));
    if any(oversizedMask)
        slidingWindowSize(oversizedMask) = totalTime;
    end

    timeOrigin = timeRange(1);
    firstCenterTime = timeOrigin + config.slidingWindowSize / 2;
    lastCenterTime = timeOrigin + totalTime - config.slidingWindowSize / 2;
    commonCenterTimes = firstCenterTime:config.stepSize:lastCenterTime;
    if isempty(commonCenterTimes)
        error('No valid windows found. Check slidingWindowSize and stepSize relative to total time.');
    end
    numWindows = length(commonCenterTimes);
    fprintf('\nCommon window centers: %d windows from %.2f s to %.2f s (stepSize=%.3f s)\n', ...
        numWindows, firstCenterTime, lastCenterTime, config.stepSize);

    bhvTimeOrigin = session_time_origin(dataStruct);

    [popActivity, d2, d2Kl, d2KlErr, d2KlExit, startS, popActivityWindows, popActivityFull] = ...
        deal(cell(1, numAreas));
    nNeuronsPerArea = nan(1, numAreas);

    if strcmp(sessionType, 'spontaneous') && isfield(config, 'behaviorNumeratorIDs') && ...
            isfield(config, 'behaviorDenominatorIDs') && ...
            ~isempty(config.behaviorNumeratorIDs) && ~isempty(config.behaviorDenominatorIDs)
        behaviorProportion = cell(1, numAreas);
    else
        behaviorProportion = cell(1, numAreas);
        for a = 1:numAreas
            behaviorProportion{a} = [];
        end
    end

    fprintf('\n=== Filtering Areas to Process ===\n');
    areasToProcess = [];
    minNeuronsRequired = config.nMinNeurons;
    for a = areasToTest
        aID = dataStruct.idMatIdx{a};
        nNeurons = length(aID);
        nNeuronsPerArea(a) = nNeurons;
        if nNeurons < minNeuronsRequired
            fprintf('  Will skip area %s: only %d neurons (minimum %d)\n', ...
                areas{a}, nNeurons, minNeuronsRequired);
            popActivity{a} = [];
            d2{a} = [];
            d2Kl{a} = [];
            d2KlErr{a} = [];
            d2KlExit{a} = [];
            startS{a} = [];
            popActivityWindows{a} = [];
            popActivityFull{a} = [];
            behaviorProportion{a} = [];
        elseif isnan(binSize(a))
            fprintf('  Will skip area %s: invalid bin size\n', areas{a});
            popActivity{a} = [];
            d2{a} = [];
            d2Kl{a} = [];
            d2KlErr{a} = [];
            d2KlExit{a} = [];
            startS{a} = [];
            popActivityWindows{a} = [];
            popActivityFull{a} = [];
            behaviorProportion{a} = [];
        else
            areasToProcess = [areasToProcess, a]; %#ok<AGROW>
        end
    end
    if isempty(areasToProcess)
        error('No valid areas to process.');
    end
    fprintf('  Will process %d area(s): %s\n', length(areasToProcess), strjoin(areas(areasToProcess), ', '));

    fprintf('\n=== Processing Areas ===\n');
    nAreasToProcess = length(areasToProcess);
    tempPopActivity = cell(1, nAreasToProcess);
    tempD2 = cell(1, nAreasToProcess);
    tempD2Kl = cell(1, nAreasToProcess);
    tempD2KlErr = cell(1, nAreasToProcess);
    tempD2KlExit = cell(1, nAreasToProcess);
    tempStartS = cell(1, nAreasToProcess);
    tempPopActivityWindows = cell(1, nAreasToProcess);
    tempPopActivityFull = cell(1, nAreasToProcess);
    tempBehaviorProportion = cell(1, nAreasToProcess);

    for idx = 1:nAreasToProcess
        a = areasToProcess(idx);
        fprintf('\nProcessing area %s (%s)...\n', areas{a}, sessionType);
        tic;

        neuronIDs = dataStruct.idLabel{a};
        if config.sdfFlag
            aDataMat = bin_spikes_with_sdf(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
                neuronIDs, timeRange, binSize(a), config.sdfSigmaMs);
        else
            aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
                neuronIDs, timeRange, binSize(a));
        end
        if config.pcaFlag
            aDataMat = apply_config_pca_reconstruction(aDataMat, config);
        end
        numTimePoints = size(aDataMat, 1);
        popActivityLocal = sum(aDataMat, 2);

        [startSLocal, d2Local, d2KlLocal, d2KlErrLocal] = deal(nan(1, numWindows));
        d2KlExitLocal = nan(1, numWindows);
        popActivityWindowsLocal = nan(1, numWindows);
        popActivityFullLocal = nan(1, numWindows);

        if strcmp(sessionType, 'spontaneous') && isfield(dataStruct, 'bhvID') && ...
                ~isempty(dataStruct.bhvID) && isfield(config, 'behaviorNumeratorIDs') && ...
                isfield(config, 'behaviorDenominatorIDs') && ...
                ~isempty(config.behaviorNumeratorIDs) && ~isempty(config.behaviorDenominatorIDs)
            tempBehaviorProportion{idx} = nan(1, numWindows);
        else
            tempBehaviorProportion{idx} = [];
        end

        nFailedKl = 0;
        for w = 1:numWindows
            centerTime = commonCenterTimes(w);
            startSLocal(w) = centerTime;

            [startIdx, endIdx] = calculate_window_indices_from_center(...
                centerTime, slidingWindowSize(a), binSize(a), numTimePoints, timeOrigin);
            if startIdx < 1 || endIdx > numTimePoints || startIdx > endIdx
                continue;
            end

            wPopActivity = popActivityLocal(startIdx:endIdx);
            popActivityWindowsLocal(w) = mean(wPopActivity);
            midIdx = startIdx + floor((endIdx - startIdx) / 2);
            popActivityFullLocal(w) = popActivityLocal(midIdx);

            if ~isempty(tempBehaviorProportion{idx}) && isfield(dataStruct, 'fsBhv') && ...
                    ~isempty(dataStruct.fsBhv)
                fsBhv = dataStruct.fsBhv;
                bhvBinSize = 1 / fsBhv;
                winStartTime = centerTime - slidingWindowSize(a) / 2;
                winEndTime = centerTime + slidingWindowSize(a) / 2;
                bhvStartIdx = round((winStartTime - bhvTimeOrigin) / bhvBinSize) + 1;
                bhvEndIdx = round((winEndTime - bhvTimeOrigin) / bhvBinSize);
                bhvStartIdx = max(1, bhvStartIdx);
                bhvEndIdx = min(length(dataStruct.bhvID), bhvEndIdx);
                if bhvStartIdx <= bhvEndIdx
                    windowBhvID = dataStruct.bhvID(bhvStartIdx:bhvEndIdx);
                    numeratorCount = sum(ismember(windowBhvID, config.behaviorNumeratorIDs));
                    denominatorCount = sum(ismember(windowBhvID, config.behaviorDenominatorIDs));
                    if denominatorCount > 0
                        tempBehaviorProportion{idx}(w) = numeratorCount / denominatorCount;
                    end
                end
            end

            [d2Local(w), d2KlLocal(w), d2KlErrLocal(w), d2KlExitLocal(w)] = ...
                compute_window_d2_pair(wPopActivity, binSize(a), config);
            if ~isfinite(d2KlLocal(w))
                nFailedKl = nFailedKl + 1;
            end

            if mod(w, 25) == 0 || w == numWindows
                fprintf('  window %d/%d (%.1f min elapsed)\n', w, numWindows, toc/60);
            end
        end

        fprintf('Area %s completed in %.1f minutes (%d/%d KL windows failed/NaN)\n', ...
            areas{a}, toc/60, nFailedKl, numWindows);

        tempPopActivity{idx} = popActivityLocal;
        tempD2{idx} = d2Local;
        tempD2Kl{idx} = d2KlLocal;
        tempD2KlErr{idx} = d2KlErrLocal;
        tempD2KlExit{idx} = d2KlExitLocal;
        tempStartS{idx} = startSLocal;
        tempPopActivityWindows{idx} = popActivityWindowsLocal;
        tempPopActivityFull{idx} = popActivityFullLocal;
    end

    for idx = 1:nAreasToProcess
        a = areasToProcess(idx);
        popActivity{a} = tempPopActivity{idx};
        d2{a} = tempD2{idx};
        d2Kl{a} = tempD2Kl{idx};
        d2KlErr{a} = tempD2KlErr{idx};
        d2KlExit{a} = tempD2KlExit{idx};
        startS{a} = tempStartS{idx};
        popActivityWindows{a} = tempPopActivityWindows{idx};
        popActivityFull{a} = tempPopActivityFull{idx};
        behaviorProportion{a} = tempBehaviorProportion{idx};
    end

    results = struct();
    results.sessionType = sessionType;
    results.areas = areas;
    results.d2 = d2;
    results.d2Kl = d2Kl;
    results.d2KlErr = d2KlErr;
    results.d2KlExit = d2KlExit;
    results.startS = startS;
    results.popActivity = popActivity;
    results.popActivityWindows = popActivityWindows;
    results.popActivityFull = popActivityFull;
    results.nNeurons = nNeuronsPerArea;
    results.binSize = binSize;
    results.slidingWindowSize = slidingWindowSize;
    results.d2WindowSize = slidingWindowSize;
    results.behaviorProportion = behaviorProportion;
    results.filenameSuffix = filenameSuffix;

    results.params = struct();
    results.params.slidingWindowSize = config.slidingWindowSize;
    results.params.stepSize = config.stepSize;
    results.params.timeOrigin = session_time_origin(dataStruct);
    results.params.areasToTest = areasToTest;
    if isfield(config, 'brainAreas') && ~isempty(config.brainAreas)
        results.params.brainAreas = config.brainAreas;
    else
        results.params.brainAreas = areas(areasToTest);
    end
    results.params.analyzeD2 = true;
    results.params.pcaFlag = config.pcaFlag;
    results.params.pcaFirstFlag = config.pcaFirstFlag;
    results.params.nDim = config.nDim;
    results.params.sdfFlag = config.sdfFlag;
    results.params.sdfSigmaMs = config.sdfSigmaMs;
    results.params.pOrder = config.pOrder;
    results.params.critType = config.critType;
    results.params.useLog10D2 = config.useLog10D2;
    results.params.klFitMethod = config.klFitMethod;
    results.params.klErrBars = config.klErrBars;
    results.params.klParallel = config.klParallel;
    if strcmp(sessionType, 'spontaneous') && isfield(config, 'behaviorNumeratorIDs')
        results.params.behaviorNumeratorIDs = config.behaviorNumeratorIDs;
        results.params.behaviorDenominatorIDs = config.behaviorDenominatorIDs;
    end

    if config.saveData
        save(resultsPath, 'results');
        fprintf('Saved compare d2 results to %s\n', resultsPath);
    else
        fprintf('Skipping save (config.saveData = false)\n');
    end

    if config.makePlots
        plotArgs = {};
        if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
            plotArgs = [plotArgs, {'sessionName', dataStruct.sessionName}];
        end
        if isfield(dataStruct, 'dataBaseName') && ~isempty(dataStruct.dataBaseName)
            plotArgs = [plotArgs, {'dataBaseName', dataStruct.dataBaseName}];
        end
        plotConfig = setup_plotting(config.saveDir, plotArgs{:});
        criticality_ar_compare_plot(results, plotConfig, config, dataStruct, filenameSuffix);
    end
end

function config = set_compare_config_defaults(config)
% SET_COMPARE_CONFIG_DEFAULTS Defaults for the Euclidean vs KL d2 sandbox

    defaults = struct();
    defaults.pcaFlag = 0;
    defaults.pcaFirstFlag = 1;
    defaults.nDim = 4;
    defaults.useOptimalBinWindowFunction = false;
    defaults.makePlots = true;
    defaults.saveData = true;
    defaults.useLog10D2 = false;
    defaults.minSpikesPerBin = 3;
    defaults.maxSpikesPerBin = 50;
    defaults.minBinsPerWindow = 1000;
    defaults.pOrder = 10;
    defaults.critType = 2;
    defaults.nMinNeurons = 10;
    defaults.brainAreas = {};
    defaults.sdfFlag = false;
    defaults.sdfSigmaMs = 10;
    defaults.klFitMethod = 'MaxLikelihood';
    defaults.klErrBars = true;
    defaults.klParallel = false;

    fields = fieldnames(defaults);
    for i = 1:length(fields)
        if ~isfield(config, fields{i})
            config.(fields{i}) = defaults.(fields{i});
        end
    end
end

function [binSize, slidingWindowSize] = find_compare_bin_window(dataStruct, config, areasToTest, timeRange)
% FIND_COMPARE_BIN_WINDOW Per-area bin/window sizes from spikes or config

    numAreas = length(dataStruct.areas);
    binSize = zeros(1, numAreas);
    slidingWindowSize = zeros(1, numAreas);

    if config.useOptimalBinWindowFunction
        for a = areasToTest
            neuronIDs = dataStruct.idLabel{a};
            thisFiringRate = calculate_firing_rate_from_spikes(...
                dataStruct.spikeTimes, dataStruct.spikeClusters, neuronIDs, timeRange);
            [binSize(a), slidingWindowSize(a)] = ...
                find_optimal_bin_and_window(thisFiringRate, config.minSpikesPerBin, config.minBinsPerWindow);
        end
    else
        if ~isfield(config, 'binSize')
            error('binSize must be provided if useOptimalBinWindowFunction is false');
        end
        if isscalar(config.binSize)
            binSize = repmat(config.binSize, 1, numAreas);
        else
            binSize = config.binSize;
        end
        if isscalar(config.slidingWindowSize)
            slidingWindowSize = repmat(config.slidingWindowSize, 1, numAreas);
        else
            slidingWindowSize = config.slidingWindowSize;
        end
    end

    for a = areasToTest
        fprintf('Area %s: bin size = %.3f s, window size = %.1f s\n', ...
            dataStruct.areas{a}, binSize(a), slidingWindowSize(a));
    end
end

function [d2Old, d2Kl, d2KlErr, d2KlExit] = compute_window_d2_pair(wPopActivity, binSize, config)
% COMPUTE_WINDOW_D2_PAIR Euclidean d2 and KL-rate d2 (+ error) for one window
%
% Variables:
%   wPopActivity - Population spike-count trace for this window
%   binSize      - Bin duration in seconds (deltaT for calc_db)
%   config       - pOrder, critType, klFitMethod, klErrBars, klParallel
%
% Goal:
%   Fit both metrics on the same pop-activity vector. KL error bars follow
%   Sooter et al. S2.5 and require MaxLikelihood fitting.

    d2Old = nan;
    d2Kl = nan;
    d2KlErr = nan;
    d2KlExit = 1;

    popTrace = double(wPopActivity(:));
    if numel(popTrace) < (config.pOrder + 2)
        return;
    end
    popStd = nanstd(popTrace);
    if ~any(isfinite(popTrace)) || ~(popStd > 0)
        return;
    end

    try
        [varphi, ~] = myYuleWalker3(popTrace, config.pOrder);
        d2Old = getFixedPointDistance2(config.pOrder, config.critType, varphi);
    catch
        d2Old = nan;
    end

    dbopt = struct();
    dbopt.fit_method = config.klFitMethod;
    dbopt.with_err_bars = logical(config.klErrBars);
    dbopt.with_QC = false;
    dbopt.with_parallel = logical(config.klParallel);

    try
        % evalc swallows calc_db's explosive-model fprintf spam
        [~, dbVal, sdVal, ~, ~, ~, ~, exitStatus] = evalc( ...
            'calc_db(popTrace, config.pOrder, binSize, config.critType, dbopt)');
        d2Kl = dbVal;
        d2KlErr = sdVal;
        d2KlExit = exitStatus;
    catch
        d2Kl = nan;
        d2KlErr = nan;
        d2KlExit = 1;
    end

    if ~isreal(d2Kl) || ~isfinite(d2Kl)
        d2Kl = nan;
    end
    if ~isreal(d2KlErr) || ~isfinite(d2KlErr) || d2KlErr < 0
        d2KlErr = nan;
    end
end

function add_compare_toolbox_paths(srcRoot)
% ADD_COMPARE_TOOLBOX_PATHS Add Euclidean d2 and prox_crit_toolkit to path
%
% Variables:
%   srcRoot - neuro-behavior/src (already resolved, no '..' segments)
%
% Goal:
%   fileparts() does not collapse '..', so toolbox roots must be walked
%   directory-by-directory. Try Projects/toolboxes (sibling of the repo)
%   then neuro-behavior/toolboxes.

    projectRoot = fileparts(srcRoot);
    projectsRoot = fileparts(projectRoot);
    candidateRoots = {
        fullfile(projectsRoot, 'toolboxes')
        fullfile(projectRoot, 'toolboxes')
        };

    shewPath = '';
    klPath = '';
    for i = 1:numel(candidateRoots)
        shewCandidate = fullfile(candidateRoots{i}, 'criticality_shew');
        klCandidate = fullfile(candidateRoots{i}, 'prox_crit_toolkit', 'src');
        if isempty(shewPath) && exist(shewCandidate, 'dir')
            shewPath = shewCandidate;
        end
        if isempty(klPath) && exist(klCandidate, 'dir') && ...
                exist(fullfile(klCandidate, 'calc_db.m'), 'file')
            klPath = klCandidate;
        end
    end

    if ~isempty(shewPath)
        addpath(shewPath);
    end
    if isempty(klPath)
        error(['prox_crit_toolkit not found. Expected calc_db.m under ', ...
            '%s\\prox_crit_toolkit\\src or %s\\prox_crit_toolkit\\src'], ...
            candidateRoots{1}, candidateRoots{2});
    end
    addpath(klPath);

    if exist('calc_db', 'file') ~= 2
        error('calc_db.m not on the path after adding %s', klPath);
    end
    if exist('myYuleWalker3', 'file') ~= 2
        error('myYuleWalker3.m not on the path. Expected %s', shewPath);
    end
end
