function results = criticality_prg_analysis(dataStruct, config)
% CRITICALITY_PRG_ANALYSIS PRG kurtosis in non-overlapping windows (PCA or ICG)
%
% Implements phenomenological renormalization group (PRG) analyses per brain area:
%   'pca' - Momentum-space eigenvector projection (Cambrainha et al. 2025 / Bradde & Bialek 2017)
%   'icg' - Real-space iterative correlation grouping (Morales et al. 2023 PNAS)
%
% Paper pipeline (this file):
%   1. Bin spike trains and segment into non-overlapping block windows.
%   2. Per window: compute population CV; exclude artefact windows (CV > 5).
%   3. Per window: coarse-grain (PCA or ICG) + kurtosis and D_JS to Gaussian.
%   4. Optional: surrogate null (config.surrogateMethod):
%        'isi' (default) - ISI-shuffle per unit within window, re-binned (Cambrainha Appendix)
%        'circular' - random circshift per neuron on binned activity within window
%                     (same circular-per-neuron null as criticality AR / svm_decoding_compare)
%
% Not implemented here (paper Appendix 4): real-space paired-correlation
% coarse-graining and scaling exponents (alpha, beta, mu, z).
%
% Variables:
%   dataStruct - From load_session_data(); needs spikeTimes, spikeClusters, areas, idLabel
%   config     - See set_prg_config_defaults(); prgMethod ('pca'|'icg');
%                surrogateMethod ('isi'|'circular') when enableSurrogates is true.
%
% Returns:
%   results - kappa (N/16), djs (Jensen-Shannon distance to Gaussian at final cutoff),
%   kappaByCutoff (RG flow), windowStartS, popCv, windowExcluded, ...
%   When makePlots is true, criticality_prg_plot uses the same kappa axis limits on every
%   area tile (y-axis on time series, x-axis and histogram bins on distribution figures),
%   with upper bound results.params.kappaAxisMax (default 20).

    %% --- Paths and input validation ---
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'session_prep', 'utils'));
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'data_prep'));
    addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

    validate_workspace_vars({'sessionType', 'spikeTimes', 'spikeClusters', 'areas', 'idMatIdx'}, dataStruct, ...
        'errorMsg', 'Required field', 'source', 'load_session_data');

    if nargin < 2 || isempty(config) || ~isstruct(config)
        config = struct();
    end
    config = set_prg_config_defaults(config);

    sessionType = dataStruct.sessionType;
    areas = dataStruct.areas;
    numAreas = length(areas);

    % Optional area merges / subset (lab convention; not in Cambrainha paper)
    [areas, dataStruct, numAreas, areasToTest] = setup_prg_areas(dataStruct, config, areas, numAreas);

    fprintf('\n=== Criticality PRG Analysis Setup ===\n');
    fprintf('PRG method: %s\n', config.prgMethod);
    fprintf('Data type: %s\n', sessionType);
    fprintf('Block window: %.1f s (non-overlapping), bin size: %.3f s\n', ...
        config.blockWindowSize, config.binSize);
    fprintf('Kurtosis at N/%d; cutoff divisors: %s\n', ...
        config.finalCutoffDivisor, mat2str(config.cutoffDivisors));
    if config.enableSurrogates
        fprintf('Surrogates: %d x %s\n', config.nSurrogates, config.surrogateMethod);
    end
    if isfield(config, 'useSubsampling') && config.useSubsampling
        fprintf('Subsampling: %d subsets x %d neurons (min neurons x %.2f)\n', ...
            config.nSubsamples, config.nNeuronsSubsample, config.minNeuronsMultiple);
    end

    if isfield(config, 'useSubsampling') && config.useSubsampling
        if ~isfield(config, 'nNeuronsSubsample') || isempty(config.nNeuronsSubsample) || config.nNeuronsSubsample <= 0
            error('When config.useSubsampling is true, config.nNeuronsSubsample must be a positive scalar.');
        end
        if ~isfield(config, 'nSubsamples') || isempty(config.nSubsamples) || config.nSubsamples <= 0
            error('When config.useSubsampling is true, config.nSubsamples must be a positive scalar.');
        end
        if ~isfield(config, 'minNeuronsMultiple') || isempty(config.minNeuronsMultiple)
            config.minNeuronsMultiple = 1.0;
        end
        config.nMinNeurons = round(config.nNeuronsSubsample * config.minNeuronsMultiple);
    end

    if ~isfield(config, 'saveDir') || isempty(config.saveDir)
        config.saveDir = dataStruct.saveDir;
    end

    sessionNameForPath = '';
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        sessionNameForPath = dataStruct.sessionName;
    end

    resultsPath = create_results_path('criticality_prg', sessionType, ...
        sessionNameForPath, config.saveDir);

    %% --- Step 1: Define recording interval and non-overlapping 30 s windows ---
    % Cambrainha Appendix (Methods): "binned into 50-ms intervals, and segmented
    % into 30-s windows for analysis." Windows are contiguous and non-overlapping
    % (unlike sliding-window d2 analyses in this codebase).
    if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
        timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
    else
        timeRange = [0, max(dataStruct.spikeTimes)];
    end

    totalDuration = timeRange(2) - timeRange(1);
    numWindows = floor(totalDuration / config.blockWindowSize);
    if numWindows < 1
        error('No complete %.1f s windows in [%.2f, %.2f] s.', ...
            config.blockWindowSize, timeRange(1), timeRange(2));
    end

    % Window w covers [windowStartTimes(w), windowStartTimes(w) + blockWindowSize)
    windowStartTimes = timeRange(1) + (0:numWindows - 1) * config.blockWindowSize;
    fprintf('Non-overlapping windows: %d (%.2f s to %.2f s)\n', ...
        numWindows, windowStartTimes(1), windowStartTimes(end) + config.blockWindowSize);

    % Preallocate per-area result cells (one vector/matrix per area index)
    [kappa, kappaByCutoff, windowStartS, popCv, windowExcluded, nNeuronsPerWindow, ...
        kappaSurrogate, djs, djsSurrogate, nCutoffListRef] = deal(cell(1, numAreas));

    areasToProcess = filter_prg_areas(dataStruct, areas, areasToTest, config);

    %% --- Step 2–4: Per brain area, per 30 s window ---
    % Paper analyzes V1 (or here: each area separately). For each window we treat
    % binned activity phi_i(t) as N parallel time series (units/neurons).
    fprintf('\n=== Processing Areas ===\n');
    for a = areasToProcess
        fprintf('\nProcessing area %s (%s)...\n', areas{a}, sessionType);
        tic;

        neuronIds = dataStruct.idLabel{a};
        kappa{a} = nan(1, numWindows);
        djs{a} = nan(1, numWindows);
        popCv{a} = nan(1, numWindows);
        windowExcluded{a} = false(1, numWindows);
        nNeuronsPerWindow{a} = repmat(numel(neuronIds), 1, numWindows);
        windowStartS{a} = windowStartTimes;
        kappaByCutoff{a} = nan(numWindows, numel(config.cutoffDivisors));
        nCutoffListRef{a} = [];

        useSubsamplingArea = isfield(config, 'useSubsampling') && config.useSubsampling;
        nSubsamplesArea = 1;
        if useSubsamplingArea
            nSubsamplesArea = config.nSubsamples;
        end

        if config.enableSurrogates
            kappaSurrogate{a} = nan(numWindows, config.nSurrogates * nSubsamplesArea);
            djsSurrogate{a} = nan(numWindows, config.nSurrogates * nSubsamplesArea);
        else
            kappaSurrogate{a} = [];
            djsSurrogate{a} = [];
        end

        numNeuronsArea = numel(neuronIds);
        nNeuronsSubsampleArea = numNeuronsArea;
        neuronIdxSubsamples = {};
        if useSubsamplingArea
            nNeuronsSubsampleArea = min(config.nNeuronsSubsample, numNeuronsArea);
            neuronIdxSubsamples = cell(1, nSubsamplesArea);
            for s = 1:nSubsamplesArea
                if nNeuronsSubsampleArea == numNeuronsArea
                    neuronIdxSubsamples{s} = 1:numNeuronsArea;
                else
                    neuronIdxSubsamples{s} = randperm(numNeuronsArea, nNeuronsSubsampleArea);
                end
            end
        end

        for w = 1:numWindows
            winStart = windowStartTimes(w);
            winEnd = winStart + config.blockWindowSize;
            winRange = [winStart, winEnd];

            %% Step 2a: Bin spikes at 50 ms within this window (paper: 50-ms bins)
            % dataMat(t, i) = spike count of unit i in bin t; N = numel(neuronIds).
            dataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
                neuronIds, winRange, config.binSize);

            %% Step 2b: Artefact exclusion via population coefficient of variation ---
            prgOutFull = compute_prg_window_metrics(dataMat, config);
            popCv{a}(w) = prgOutFull.popCv;
            if isempty(nCutoffListRef{a})
                nCutoffListRef{a} = prgOutFull.nCutoffList;
                kappaByCutoff{a} = nan(numWindows, numel(prgOutFull.nCutoffList));
            end
            if prgOutFull.popCv > config.cvThreshold
                windowExcluded{a}(w) = true;
                continue;
            end

            if useSubsamplingArea
                kappaSub = nan(1, nSubsamplesArea);
                djsSub = nan(1, nSubsamplesArea);
                kappaByCutoffSub = nan(nSubsamplesArea, size(kappaByCutoff{a}, 2));
                for s = 1:nSubsamplesArea
                    dataSub = dataMat(:, neuronIdxSubsamples{s});
                    prgOut = compute_prg_window_metrics(dataSub, config);
                    kappaSub(s) = prgOut.kappaFinal;
                    djsSub(s) = prgOut.djsFinal;
                    nCut = min(numel(prgOut.kappaByCutoff), size(kappaByCutoff{a}, 2));
                    kappaByCutoffSub(s, 1:nCut) = prgOut.kappaByCutoff(1:nCut);

                    if config.enableSurrogates
                        for surrIdx = 1:config.nSurrogates
                            surrMat = build_prg_surrogate_data_mat(dataSub, dataStruct, ...
                                neuronIds(neuronIdxSubsamples{s}), winRange, config);
                            surrOut = compute_prg_window_metrics(surrMat, config);
                            colIdx = (s - 1) * config.nSurrogates + surrIdx;
                            kappaSurrogate{a}(w, colIdx) = surrOut.kappaFinal;
                            djsSurrogate{a}(w, colIdx) = surrOut.djsFinal;
                        end
                    end
                end
                kappa{a}(w) = nanmean(kappaSub);
                djs{a}(w) = nanmean(djsSub);
                kappaByCutoff{a}(w, :) = nanmean(kappaByCutoffSub, 1);
            else
                %% Step 3: PRG coarse-graining + kurtosis / D_JS (PCA or ICG)
                kappa{a}(w) = prgOutFull.kappaFinal;
                djs{a}(w) = prgOutFull.djsFinal;
                nCut = min(numel(prgOutFull.kappaByCutoff), size(kappaByCutoff{a}, 2));
                kappaByCutoff{a}(w, 1:nCut) = prgOutFull.kappaByCutoff(1:nCut);

                if config.enableSurrogates
                    for s = 1:config.nSurrogates
                        surrMat = build_prg_surrogate_data_mat(dataMat, dataStruct, ...
                            neuronIds, winRange, config);
                        surrOut = compute_prg_window_metrics(surrMat, config);
                        kappaSurrogate{a}(w, s) = surrOut.kappaFinal;
                        djsSurrogate{a}(w, s) = surrOut.djsFinal;
                    end
                end
            end
        end

        nValid = sum(isfinite(kappa{a}) & ~windowExcluded{a});
        fprintf('Area %s: %d / %d windows with valid kappa (%.1f min)\n', ...
            areas{a}, nValid, numWindows, toc / 60);
    end

    % Areas skipped in filter_prg_areas get empty outputs
    for a = setdiff(1:numAreas, areasToProcess)
        kappa{a} = [];
        kappaByCutoff{a} = [];
        djs{a} = [];
        windowStartS{a} = [];
        popCv{a} = [];
        windowExcluded{a} = [];
        nNeuronsPerWindow{a} = [];
        kappaSurrogate{a} = [];
        djsSurrogate{a} = [];
        nCutoffListRef{a} = [];
    end

    results = build_prg_results_structure(dataStruct, config, areas, ...
        kappa, kappaByCutoff, windowStartS, popCv, windowExcluded, ...
        nNeuronsPerWindow, kappaSurrogate, djs, djsSurrogate, nCutoffListRef, windowStartTimes);

    if config.saveData
        save(resultsPath, 'results');
        fprintf('Saved PRG results to %s\n', resultsPath);
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
        criticality_prg_plot(results, plotConfig, config, dataStruct);
    end
end

function config = set_prg_config_defaults(config)
% SET_PRG_CONFIG_DEFAULTS Paper-aligned defaults (Cambrainha et al. 2025 Appendix).

    defaults = struct();
    defaults.blockWindowSize = 30;       % 30 s windows (Methods)
    defaults.binSize = 0.05;             % 50 ms bins (Methods)
    defaults.cvThreshold = 5;            % CV exclusion (Appendix Eq. A1)
    defaults.cutoffDivisors = [1, 2, 4, 8, 16];  % N_c = N, N/2, ... N/16 (Fig. 1d)
    defaults.finalCutoffDivisor = 16;    % Report kappa at N/16 (Appendix 3)
    defaults.enableSurrogates = false;
    defaults.surrogateMethod = 'isi';    % 'isi' (paper) or 'circular' (per-neuron circshift on binned data)
    defaults.nSurrogates = 1;
    defaults.makePlots = true;
    defaults.saveData = true;
    defaults.includeM2356 = false;
    defaults.nMinNeurons = 10;           % Paper used >128 units; lab minimum is configurable
    defaults.useSubsampling = false;
    defaults.nSubsamples = 10;
    defaults.nNeuronsSubsample = 10;
    defaults.minNeuronsMultiple = 1.0;
    defaults.brainAreas = {};
    defaults.kappaAxisMax = 20;          % Cap kurtosis axis (y time series, x histograms); use Inf for no cap
    defaults.prgMethod = 'pca';        % 'pca' (momentum-space) or 'icg' (real-space ICG, Morales 2023)

    fields = fieldnames(defaults);
    for i = 1:numel(fields)
        if ~isfield(config, fields{i})
            config.(fields{i}) = defaults.(fields{i});
        end
    end

    config.surrogateMethod = lower(strtrim(config.surrogateMethod));
    validSurrogateMethods = {'isi', 'circular'};
    if ~ismember(config.surrogateMethod, validSurrogateMethods)
        error('config.surrogateMethod must be ''isi'' or ''circular'', got "%s".', config.surrogateMethod);
    end

    config.prgMethod = lower(strtrim(config.prgMethod));
    validPrgMethods = {'pca', 'icg'};
    if ~ismember(config.prgMethod, validPrgMethods)
        error('config.prgMethod must be ''pca'' or ''icg'', got "%s".', config.prgMethod);
    end
end

function surrMat = build_prg_surrogate_data_mat(dataMat, dataStruct, neuronIds, winRange, config)
% BUILD_PRG_SURROGATE_DATA_MAT Binned surrogate activity for one window
%
% Variables:
%   dataMat - [timeBins x neurons] real binned activity for the window
%   dataStruct - Session data (spikeTimes, spikeClusters for ISI method)
%   neuronIds - Neuron IDs in this area
%   winRange - [startTime, endTime] in seconds
%   config - PRG config with surrogateMethod ('isi' or 'circular')
%
% Returns:
%   surrMat - Binned surrogate spike counts, same size as dataMat

    switch config.surrogateMethod
        case 'isi'
            [surrTimes, surrClusters] = shuffle_spike_isi_within_window( ...
                dataStruct.spikeTimes, dataStruct.spikeClusters, neuronIds, winRange);
            surrMat = bin_spikes(surrTimes, surrClusters, neuronIds, winRange, config.binSize);
        case 'circular'
            surrMat = apply_circular_permutation_per_neuron(dataMat);
        otherwise
            error('Unknown config.surrogateMethod: %s', config.surrogateMethod);
    end
end

function permutedMat = apply_circular_permutation_per_neuron(dataMat)
% APPLY_CIRCULAR_PERMUTATION_PER_NEURON Circular permutation per neuron (time axis)
%
% Variables:
%   dataMat - [timeBins x neurons] binned activity for one PRG window
%
% Goal:
%   Destroy temporal correlations while preserving per-neuron rate structure within
%   the window (circular-shift null used in AR decoding and d2 analyses).
%
% Returns:
%   permutedMat - Same size; each column circshifted by a random amount in [1, floor(T/2)]

    numTimeBins = size(dataMat, 1);
    maxShift = floor(numTimeBins / 2);
    if maxShift < 1
        permutedMat = dataMat;
        return;
    end

    permutedMat = zeros(size(dataMat));
    numNeurons = size(dataMat, 2);
    for n = 1:numNeurons
        shiftAmount = randi([1, maxShift]);
        permutedMat(:, n) = circshift(dataMat(:, n), shiftAmount);
    end
end

function [areas, dataStruct, numAreas, areasToTest] = setup_prg_areas(dataStruct, config, areas, numAreas)
% SETUP_PRG_AREAS M2356 merge, areasToTest, and brainAreas filtering (lab-specific).

    if config.includeM2356
        idxM23 = find(strcmp(areas, 'M23'));
        idxM56 = find(strcmp(areas, 'M56'));
        if ~isempty(idxM23) && ~isempty(idxM56) && ~any(strcmp(areas, 'M2356'))
            areas{end + 1} = 'M2356';
            dataStruct.areas = areas;
            dataStruct.idMatIdx{end + 1} = [dataStruct.idMatIdx{idxM23}(:); dataStruct.idMatIdx{idxM56}(:)];
            if isfield(dataStruct, 'idLabel')
                dataStruct.idLabel{end + 1} = [dataStruct.idLabel{idxM23}(:); dataStruct.idLabel{idxM56}(:)];
            end
            numAreas = numel(areas);
        end
    end

    if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
        areasToTest = dataStruct.areasToTest;
    else
        areasToTest = 1:numAreas;
    end

    if config.includeM2356 && any(strcmp(areas, 'M2356'))
        m2356Idx = find(strcmp(areas, 'M2356'));
        if ~ismember(m2356Idx, areasToTest)
            areasToTest = [areasToTest, m2356Idx];
        end
    end

    if isfield(config, 'brainAreas') && ~isempty(config.brainAreas)
        if ischar(config.brainAreas)
            desiredAreas = {config.brainAreas};
        else
            desiredAreas = config.brainAreas;
        end
        if config.includeM2356
            hasM23 = any(strcmp(desiredAreas, 'M23'));
            hasM56 = any(strcmp(desiredAreas, 'M56'));
            if hasM23 && hasM56 && any(strcmp(areas, 'M2356'))
                desiredAreas = [desiredAreas, {'M2356'}];
            end
        end
        selectedIdx = [];
        for iArea = 1:numel(desiredAreas)
            idx = find(strcmp(areas, desiredAreas{iArea}));
            if isempty(idx)
                fprintf('Warning: requested brain area "%s" not found. Skipping.\n', desiredAreas{iArea});
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
end

function areasToProcess = filter_prg_areas(dataStruct, areas, areasToTest, config)
% FILTER_PRG_AREAS Require minimum neuron count per area (config.nMinNeurons).

    areasToProcess = [];
    fprintf('\n=== Filtering Areas to Process ===\n');
    for a = areasToTest
        nNeurons = numel(dataStruct.idMatIdx{a});
        if nNeurons < config.nMinNeurons
            fprintf('  Will skip area %s: only %d neurons (min %d)\n', ...
                areas{a}, nNeurons, config.nMinNeurons);
        else
            areasToProcess = [areasToProcess, a]; %#ok<AGROW>
        end
    end
    if isempty(areasToProcess)
        error('No valid areas to process.');
    end
    fprintf('  Will process %d area(s): %s\n', numel(areasToProcess), strjoin(areas(areasToProcess), ', '));
end

function results = build_prg_results_structure(dataStruct, config, areas, ...
    kappa, kappaByCutoff, windowStartS, popCv, windowExcluded, ...
    nNeuronsPerWindow, kappaSurrogate, djs, djsSurrogate, nCutoffListRef, windowStartTimes)
% BUILD_PRG_RESULTS_STRUCTURE Package outputs for saving and downstream plots/stats.
%
% Key fields:
%   kappa{area}         - N/16 kurtosis per window (paper's primary window statistic)
%   djs{area}           - Jensen-Shannon distance to N(0,1) at final cutoff (Cambrainha 2026)
%   kappaByCutoff{area} - kappa at each N_c (RG flow across momentum shells)
%   kappaSurrogate{area}- optional null kappa (ISI or circular surrogate)
%   djsSurrogate{area}  - optional null D_JS for the same windows
%   windowExcluded{area}- logical; true if CV > cvThreshold
%   nCutoffList{area}   - actual N_c values used (may differ when N is small)
%   params.kappaAxisMax - upper cap on kurtosis axes in plots (mirrors config)

    results = struct();
    results.sessionType = dataStruct.sessionType;
    results.areas = areas;
    results.kappa = kappa;
    results.djs = djs;
    results.kappaByCutoff = kappaByCutoff;
    results.windowStartS = windowStartS;
    results.popCv = popCv;
    results.windowExcluded = windowExcluded;
    results.nNeuronsPerWindow = nNeuronsPerWindow;
    results.nCutoffList = nCutoffListRef;
    results.kappaSurrogate = kappaSurrogate;
    results.djsSurrogate = djsSurrogate;
    results.params.blockWindowSize = config.blockWindowSize;
    results.params.binSize = config.binSize;
    results.params.cvThreshold = config.cvThreshold;
    results.params.cutoffDivisors = config.cutoffDivisors;
    results.params.finalCutoffDivisor = config.finalCutoffDivisor;
    results.params.enableSurrogates = config.enableSurrogates;
    results.params.surrogateMethod = config.surrogateMethod;
    results.params.nSurrogates = config.nSurrogates;
    results.params.nMinNeurons = config.nMinNeurons;
    if isfield(config, 'useSubsampling')
        results.params.useSubsampling = config.useSubsampling;
        results.useSubsampling = config.useSubsampling;
    end
    if isfield(config, 'nSubsamples')
        results.params.nSubsamples = config.nSubsamples;
    end
    if isfield(config, 'nNeuronsSubsample')
        results.params.nNeuronsSubsample = config.nNeuronsSubsample;
    end
    results.params.windowStartTimes = windowStartTimes;
    results.params.kappaAxisMax = config.kappaAxisMax;
    results.params.prgMethod = config.prgMethod;
    results.analysisType = 'criticality_prg';
end
