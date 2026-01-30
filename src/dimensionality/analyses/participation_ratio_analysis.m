function results = participation_ratio_analysis(dataStruct, config)
% PARTICIPATION_RATIO_ANALYSIS Sliding window participation ratio from spike data
%
% Variables:
%   dataStruct - Data structure from load_sliding_window_data()
%   config     - Configuration struct with fields:
%     .slidingWindowSize - Window size in seconds
%     .stepSize          - Step size in seconds (required)
%     .useOptimalBinWindowFunction - Find optimal bin/window (default: true)
%     .binSize           - Bin size in seconds (used if useOptimalBinWindowFunction false)
%     .minSpikesPerBin   - For optimal bin (default: 3)
%     .minBinsPerWindow  - For optimal bin finding (default: 1000)
%     .windowSizeNeuronMultiple - Sliding window length in units of (nNeurons * binSize).
%                                Number of time bins per window = this multiple * nNeurons,
%                                so window (s) = multiple * nNeurons * binSize (default: 10).
%     .makePlots         - Create plots (default: true)
%     .saveData          - Save results (default: true)
%     .saveDir           - Save directory (optional, uses dataStruct.saveDir)
%     .nMinNeurons       - Minimum neurons per area (default: 10)
%
% Goal:
%   Compute participation ratio PR = (trace(C))^2 / trace(C^2) in sliding windows,
%   where C is the covariance of the binned population activity [time x neurons].
%
% Returns:
%   results - Structure with participationRatio, startS, popActivity, popActivityWindows, params

    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils'));

    validate_workspace_vars({'spikeTimes', 'spikeClusters', 'areas', 'idMatIdx'}, dataStruct, ...
        'errorMsg', 'Required field', 'source', 'load_sliding_window_data');

    if nargin < 2 || isempty(config) || ~isstruct(config)
        config = struct();
    end
    config = set_participation_ratio_config_defaults(config);

    sessionType = dataStruct.sessionType;
    areas = dataStruct.areas;
    numAreas = length(areas);

    if isfield(dataStruct, 'areasToTest')
        areasToTest = dataStruct.areasToTest;
    else
        areasToTest = 1:numAreas;
    end

    fprintf('\n=== Participation Ratio Analysis ===\n');
    fprintf('Data type: %s, areas: %d\n', sessionType, numAreas);
    fprintf('Window size: %.2f s, step: %.3f s\n', config.slidingWindowSize, config.stepSize);

    if ~isfield(config, 'saveDir') || isempty(config.saveDir)
        config.saveDir = dataStruct.saveDir;
    end

    sessionNameForPath = '';
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        sessionNameForPath = dataStruct.sessionName;
    end

    resultsPath = create_results_path('participation_ratio', sessionType, ...
        sessionNameForPath, config.saveDir, 'createDir', true);

    if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
        timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
    else
        timeRange = [0, max(dataStruct.spikeTimes)];
    end

    % Bin and window sizes
    fprintf('\n--- Bin and window sizes ---\n');
    [binSize, slidingWindowSize] = find_bin_window_participation_ratio(...
        dataStruct, config, areasToTest, timeRange);

    if ~isfield(config, 'stepSize') || isempty(config.stepSize)
        error('stepSize must be provided in config');
    end

    totalTime = timeRange(2) - timeRange(1);
    maxWindowSize = max(slidingWindowSize(areasToTest));
    firstCenterTime = maxWindowSize / 2;
    lastCenterTime = totalTime - maxWindowSize / 2;
    commonCenterTimes = firstCenterTime:config.stepSize:lastCenterTime;

    if isempty(commonCenterTimes)
        error('No valid windows. Check slidingWindowSize and stepSize.');
    end

    numWindows = length(commonCenterTimes);
    fprintf('Common window centers: %d (step %.3f s)\n', numWindows, config.stepSize);

    [participationRatio, startS, popActivity, popActivityWindows, popActivityFull] = ...
        deal(cell(1, numAreas));

    areasToProcess = [];
    for a = areasToTest
        nNeurons = length(dataStruct.idMatIdx{a});
        if nNeurons < config.nMinNeurons || isnan(binSize(a))
            participationRatio{a} = [];
            startS{a} = [];
            popActivity{a} = [];
            popActivityWindows{a} = [];
            popActivityFull{a} = [];
            fprintf('  Skip %s: n=%d or invalid bin\n', areas{a}, nNeurons);
        else
            areasToProcess = [areasToProcess, a];
        end
    end

    if isempty(areasToProcess)
        error('No areas to process.');
    end

    fprintf('\n--- Processing areas ---\n');
    for a = areasToProcess
        fprintf('Area %s...\n', areas{a});
        tic;
        neuronIDs = dataStruct.idLabel{a};
        aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
            neuronIDs, timeRange, binSize(a));
        numTimePoints = size(aDataMat, 1);
        winSamples = round(slidingWindowSize(a) / binSize(a));

        popActivity{a} = mean(aDataMat, 2);
        [startS{a}, participationRatio{a}, popActivityWindows{a}, popActivityFull{a}] = ...
            deal(nan(1, numWindows));

        for w = 1:numWindows
            centerTime = commonCenterTimes(w);
            startS{a}(w) = centerTime;
            [startIdx, endIdx] = calculate_window_indices_from_center(...
                centerTime, slidingWindowSize(a), binSize(a), numTimePoints);

            if startIdx < 1 || endIdx > numTimePoints || startIdx > endIdx
                continue;
            end

            wDataMat = aDataMat(startIdx:endIdx, :);  % [time x neurons]
            wPopActivity = mean(wDataMat, 2);
            popActivityWindows{a}(w) = mean(wPopActivity);
            popActivityFull{a}(w) = popActivity{a}(startIdx + round(winSamples/2) - 1);

            prVal = compute_participation_ratio(wDataMat);
            participationRatio{a}(w) = prVal;
        end

        fprintf('  %s done in %.1f min\n', areas{a}, toc/60);
    end

    % Build results
    results = struct();
    results.sessionType = sessionType;
    results.areas = areas;
    results.participationRatio = participationRatio;
    results.startS = startS;
    results.popActivity = popActivity;
    results.popActivityWindows = popActivityWindows;
    results.popActivityFull = popActivityFull;
    results.binSize = binSize;
    results.slidingWindowSize = slidingWindowSize;
    results.params.slidingWindowSize = config.slidingWindowSize;
    results.params.stepSize = config.stepSize;
    results.params.windowSizeNeuronMultiple = config.windowSizeNeuronMultiple;

    if config.saveData
        save(resultsPath, 'results');
        fprintf('Saved to %s\n', resultsPath);
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
        participation_ratio_plot(results, plotConfig, config, dataStruct);
    end
end

function pr = compute_participation_ratio(X)
% X is [time x neurons]. C = cov(X) is [neurons x neurons].
% PR = (trace(C))^2 / trace(C^2) = (sum(lambda))^2 / sum(lambda.^2).
    if size(X, 1) < 2 || size(X, 2) < 1
        pr = nan;
        return;
    end
    C = cov(X);
    if any(isnan(C(:))) || any(isinf(C(:)))
        pr = nan;
        return;
    end
    trC = trace(C);
    trC2 = trace(C * C);
    if trC2 <= 0 || ~isfinite(trC) || ~isfinite(trC2)
        pr = nan;
        return;
    end
    pr = (trC ^ 2) / trC2;
end

function config = set_participation_ratio_config_defaults(config)
    defaults = struct();
    defaults.slidingWindowSize = 10;
    defaults.stepSize = 0.1;
    defaults.useOptimalBinWindowFunction = true;
    defaults.minSpikesPerBin = 3;
    defaults.minBinsPerWindow = 1000;
    defaults.windowSizeNeuronMultiple = 10;
    defaults.makePlots = true;
    defaults.saveData = true;
    defaults.nMinNeurons = 10;
    fn = fieldnames(defaults);
    for i = 1:length(fn)
        if ~isfield(config, fn{i})
            config.(fn{i}) = defaults.(fn{i});
        end
    end
end

function [binSize, slidingWindowSize] = find_bin_window_participation_ratio(...
    dataStruct, config, areasToTest, timeRange)
    numAreas = length(dataStruct.areas);
    binSize = zeros(1, numAreas);
    slidingWindowSize = zeros(1, numAreas);
    multiple = config.windowSizeNeuronMultiple;

    if config.useOptimalBinWindowFunction
        for a = areasToTest
            neuronIDs = dataStruct.idLabel{a};
            nNeurons = length(dataStruct.idMatIdx{a});
            fr = calculate_firing_rate_from_spikes(...
                dataStruct.spikeTimes, dataStruct.spikeClusters, ...
                neuronIDs, timeRange);
            [binSize(a), ~] = find_optimal_bin_and_window(fr, config.minSpikesPerBin, config.minBinsPerWindow);
            % Window length so that time bins per window = multiple * nNeurons (reliable covariance)
            slidingWindowSize(a) = multiple * nNeurons * binSize(a);
        end
    else
        if isfield(config, 'binSize')
            binSize = repmat(config.binSize, 1, numAreas);
        else
            error('binSize required when useOptimalBinWindowFunction is false');
        end
        for a = areasToTest
            nNeurons = length(dataStruct.idMatIdx{a});
            slidingWindowSize(a) = multiple * nNeurons * binSize(a);
        end
    end

    for a = areasToTest
        nNeurons = length(dataStruct.idMatIdx{a});
        nBins = round(slidingWindowSize(a) / binSize(a));
        fprintf('  %s: bin = %.3f s, win = %.1f s (%d bins, %d neurons, multiple %d)\n', ...
            dataStruct.areas{a}, binSize(a), slidingWindowSize(a), nBins, nNeurons, multiple);
    end
end
