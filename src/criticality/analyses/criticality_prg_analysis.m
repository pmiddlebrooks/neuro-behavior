function results = criticality_prg_analysis(dataStruct, config)
% CRITICALITY_PRG_ANALYSIS Momentum-space PRG kurtosis in non-overlapping windows
%
% Implements the momentum-space phenomenological renormalization group (PRG)
% kurtosis pipeline from Cambrainha et al. 2025 PRX Life (building on Bradde &
% Bialek 2017 J. Stat. Phys.), per brain area.
%
% Paper pipeline (this file):
%   1. Bin spike trains (50 ms) and segment into non-overlapping 30 s windows.
%   2. Per window: compute population CV; exclude artefact windows (CV > 5).
%   3. Per window: momentum-space coarse-graining + kurtosis (compute_prg_momentum_kurtosis).
%   4. Optional: ISI-shuffled surrogate data for the same steps (Appendix, Surrogate data).
%
% Not implemented here (paper Appendix 4): real-space paired-correlation
% coarse-graining and scaling exponents (alpha, beta, mu, z).
%
% Variables:
%   dataStruct - From load_session_data(); needs spikeTimes, spikeClusters, areas, idLabel
%   config     - See set_prg_config_defaults(); optional; kappaAxisMax caps PRG plot kurtosis axes (default 20).
%
% Returns:
%   results - kappa (N/16), kappaByCutoff (RG flow), windowStartS, popCv, windowExcluded, ...
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
    fprintf('Data type: %s\n', sessionType);
    fprintf('Block window: %.1f s (non-overlapping), bin size: %.3f s\n', ...
        config.blockWindowSize, config.binSize);
    fprintf('Kurtosis at N/%d; cutoff divisors: %s\n', ...
        config.finalCutoffDivisor, mat2str(config.cutoffDivisors));

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
        kappaSurrogate, nCutoffListRef] = deal(cell(1, numAreas));

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
        popCv{a} = nan(1, numWindows);
        windowExcluded{a} = false(1, numWindows);
        nNeuronsPerWindow{a} = repmat(numel(neuronIds), 1, numWindows);
        windowStartS{a} = windowStartTimes;
        kappaByCutoff{a} = nan(numWindows, numel(config.cutoffDivisors));
        nCutoffListRef{a} = [];

        if config.enableSurrogates
            kappaSurrogate{a} = nan(numWindows, config.nSurrogates);
        else
            kappaSurrogate{a} = [];
        end

        for w = 1:numWindows
            winStart = windowStartTimes(w);
            winEnd = winStart + config.blockWindowSize;
            winRange = [winStart, winEnd];

            %% Step 2a: Bin spikes at 50 ms within this window (paper: 50-ms bins)
            % dataMat(t, i) = spike count of unit i in bin t; N = numel(neuronIds).
            dataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
                neuronIds, winRange, config.binSize);

            %% Step 3: Momentum-space PRG + kurtosis (Appendix 3; Bradde & Bialek 2017)
            % compute_prg_momentum_kurtosis implements, for each N_cutoff in
            % {N, N/2, N/4, N/8, N/16}:
            %   (i)  C_ij = <phi_i phi_j> - <phi_i><phi_j>           [Eq. A2]
            %   (ii) Eigen-decompose C; rank eigenvalues lambda_1 >= ...
            %   (iii) Projector P_ij(N_c) = sum_{mu=1}^{N_c} u_mu_i u_mu_j  [Eq. A4]
            %   (iv) Coarse-grained psi_i = Z_i * sum_j P_ij (phi_j - <phi_j>) [Eq. A5]
            %        with Z_i chosen so var(psi_i) = 1 (Fig. 1d normalization)
            %   (v)  kappa = <psi^4> / <psi^2>^2 pooled over units and time bins
            % Final reported kappa is at N_cutoff = N/16 (paper Fig. 2–3, Appendix 3).
            % Gaussian fixed point has kappa = 3; noncritical windows flow toward
            % Gaussian as N_c decreases; critical-like windows retain elevated kappa.
            prgOut = compute_prg_momentum_kurtosis(dataMat, ...
                'cutoffDivisors', config.cutoffDivisors, ...
                'finalCutoffDivisor', config.finalCutoffDivisor);

            % kappaByCutoff stores the RG "flow" of kappa across cutoffs (one row per window)
            popCv{a}(w) = prgOut.popCv;
            if isempty(nCutoffListRef{a})
                nCutoffListRef{a} = prgOut.nCutoffList;
                kappaByCutoff{a} = nan(numWindows, numel(prgOut.nCutoffList));
            end

            %% Step 2b: Artefact exclusion via population coefficient of variation ---
            % Appendix Eq. (A1): CV = sigma/mu of population activity across the window.
            % Windows with CV > 5 are excluded (likely artefacts at this window length).
            if prgOut.popCv > config.cvThreshold
                windowExcluded{a}(w) = true;
                continue;
            end

            % Primary metric: kappa at finest coarse-graining step (N/16)
            kappa{a}(w) = prgOut.kappaFinal;
            nCut = min(numel(prgOut.kappaByCutoff), size(kappaByCutoff{a}, 2));
            kappaByCutoff{a}(w, 1:nCut) = prgOut.kappaByCutoff(1:nCut);

            %% Step 4 (optional): Surrogate / null comparison (Appendix, Surrogate data) ---
            % Paper: shuffle inter-spike intervals independently per unit within each
            % window, breaking cross-unit correlations. Surrogate kappa should stay
            % nearer the Gaussian fixed point (lower kappa than real data on average).
            if config.enableSurrogates
                for s = 1:config.nSurrogates
                    [surrTimes, surrClusters] = shuffle_spike_isi_within_window( ...
                        dataStruct.spikeTimes, dataStruct.spikeClusters, neuronIds, winRange);
                    surrMat = bin_spikes(surrTimes, surrClusters, neuronIds, winRange, config.binSize);
                    surrOut = compute_prg_momentum_kurtosis(surrMat, ...
                        'cutoffDivisors', config.cutoffDivisors, ...
                        'finalCutoffDivisor', config.finalCutoffDivisor);
                    kappaSurrogate{a}(w, s) = surrOut.kappaFinal;
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
        windowStartS{a} = [];
        popCv{a} = [];
        windowExcluded{a} = [];
        nNeuronsPerWindow{a} = [];
        kappaSurrogate{a} = [];
        nCutoffListRef{a} = [];
    end

    results = build_prg_results_structure(dataStruct, config, areas, ...
        kappa, kappaByCutoff, windowStartS, popCv, windowExcluded, ...
        nNeuronsPerWindow, kappaSurrogate, nCutoffListRef, windowStartTimes);

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
    defaults.enableSurrogates = false;   % ISI-shuffle null (Appendix, Surrogate data)
    defaults.nSurrogates = 1;
    defaults.makePlots = true;
    defaults.saveData = true;
    defaults.includeM2356 = false;
    defaults.nMinNeurons = 10;           % Paper used >128 units; lab minimum is configurable
    defaults.brainAreas = {};
    defaults.kappaAxisMax = 20;          % Cap kurtosis axis (y time series, x histograms); use Inf for no cap

    fields = fieldnames(defaults);
    for i = 1:numel(fields)
        if ~isfield(config, fields{i})
            config.(fields{i}) = defaults.(fields{i});
        end
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
    nNeuronsPerWindow, kappaSurrogate, nCutoffListRef, windowStartTimes)
% BUILD_PRG_RESULTS_STRUCTURE Package outputs for saving and downstream plots/stats.
%
% Key fields:
%   kappa{area}         - N/16 kurtosis per window (paper's primary window statistic)
%   kappaByCutoff{area} - kappa at each N_c (RG flow across momentum shells)
%   kappaSurrogate{area}- optional null kappa (ISI-shuffled)
%   windowExcluded{area}- logical; true if CV > cvThreshold
%   nCutoffList{area}   - actual N_c values used (may differ when N is small)
%   params.kappaAxisMax - upper cap on kurtosis axes in plots (mirrors config)

    results = struct();
    results.sessionType = dataStruct.sessionType;
    results.areas = areas;
    results.kappa = kappa;
    results.kappaByCutoff = kappaByCutoff;
    results.windowStartS = windowStartS;
    results.popCv = popCv;
    results.windowExcluded = windowExcluded;
    results.nNeuronsPerWindow = nNeuronsPerWindow;
    results.nCutoffList = nCutoffListRef;
    results.kappaSurrogate = kappaSurrogate;
    results.params.blockWindowSize = config.blockWindowSize;
    results.params.binSize = config.binSize;
    results.params.cvThreshold = config.cvThreshold;
    results.params.cutoffDivisors = config.cutoffDivisors;
    results.params.finalCutoffDivisor = config.finalCutoffDivisor;
    results.params.enableSurrogates = config.enableSurrogates;
    results.params.nSurrogates = config.nSurrogates;
    results.params.nMinNeurons = config.nMinNeurons;
    results.params.windowStartTimes = windowStartTimes;
    results.params.kappaAxisMax = config.kappaAxisMax;
    results.analysisType = 'criticality_prg';
end
