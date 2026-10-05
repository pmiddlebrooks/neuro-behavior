%%
% Session LFP Criticality (Manuscript)
%
% For one session and one brain area, cleans the LFP (artifact removal and a
% zero-mean analysis window), lowpasses at 100 Hz, and runs:
%   d2        - non-overlapping windows on mean-binned raw LFP and on each
%               band's binned power (criticality_lfp_analysis)
%   avalanches - Klaus / Plenz procedure at a 2.5 SD cutoff
%                raw LFP: negative peaks <= -nSd * SD, binned at the mean
%                inter-event interval
%                binned power: bins >= mean + nSd * SD
%
% d2 distributions for raw and each band are overlaid on one axes, each in
% its own color. Avalanche size and duration CCDFs use a separate axes for
% raw LFP and for each binned power band (same figure).
%
% Variables (configure in this section; session identity comes from the
% workspace, as in session_d2_distributions.m):
%   sessionType      - 'spontaneous', 'interval', 'reach', 'semicircle', 'schall'
%   sessionName      - Session identifier
%   subjectName      - Required for spontaneous/interval; '' for reach
%   collectStart     - Window start (seconds from session onset)
%   collectEnd       - Window end (seconds); [] = through the end of the loaded LFP
%   brainArea        - Area name (e.g. 'M56'). A combined name such as
%                      'M23M56' averages the source-area LFP traces.
%   brainAreaCombinations - Merged areas: struct('name','M23M56','areas',{{'M23','M56'}})
%   d2Window         - Non-overlapping d2 window (seconds) when d2Step = d2Window
%   d2Step           - Step between window centers (seconds)
%   rawD2BinSize     - Mean-bin width (s) for raw-LFP d2. 0.005 s matches a
%                      100 Hz lowpass (200 Hz bins).
%   useLog10D2       - If true, plot log10(d2)
%   pOrder, critType - AR order and criticality type for getFixedPointDistance2
%   lfpLowpassHz     - Lowpass cutoff passed to clean_lfp_artifacts (default 100)
%   nSdCutoff        - Avalanche cutoff in SD (default 2.5)
%   bands            - Band name and [fLow fHigh] rows for lfp_bin_bandpower
%   powerLawFitMethod - 'clauset', 'plfit2023', or 'hybrid'
%   compareTailModels - If true, Vuong/AIC tests vs exponential, lognormal,
%                       and truncated power law (Klaus-style model comparison)
%   runClausetPlpva  - If true and method is 'clauset', run plpva (slow)
%   gofThreshold     - Used for 'plfit2023' and 'hybrid'
%   saveFigure       - Export PNG/EPS to dropPath/criticality_manuscript
%   plotConfig       - Axis fonts/line widths (see fill_manuscript_plot_config)
%
% Goal:
%   Compare raw-LFP and band-power criticality (d2 and avalanches) in one area.

%% Configuration
sessionType = 'spontaneous';
subjectName = 'ag25290';
sessionName = '112321';

collectStart = 0;
collectEnd = [];

brainArea = 'M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();

d2Window = 45;   % seconds
d2Step = 45;     % seconds; equal to d2Window => non-overlapping
rawD2BinSize = 0.005;
useLog10D2 = false;
pOrder = 10;
critType = 2;

lfpLowpassHz = 100;
nSdCutoff = 2.5;

bands = {'alpha', [8 13]; ...
    'beta', [13 30]; ...
    'lowGamma', [30 50]; ...
    'highGamma', [50 80]};

powerLawFitMethod = 'clauset';
compareTailModels = true;
runClausetPlpva = false;
gofThreshold = 0.8;
saveFigure = false;

plotConfig = fill_manuscript_plot_config();
plotConfig.drawPowerLawFit = true;
plotConfig.observedMarkerSize = 5;
plotConfig.fitLineWidth = 2;
plotConfig.observedMarkerFaceAlpha = 0.45;

%% Paths and session load
if ~exist('sessionType', 'var') || isempty(sessionType) || ~exist('sessionName', 'var') || isempty(sessionName)
    error('session_lfp_criticality:MissingSession', ...
        'Set sessionType and sessionName in the workspace before running.');
end
if isempty(brainArea)
    error('session_lfp_criticality:MissingArea', ...
        'Set brainArea to the LFP area to analyze (e.g. ''M56'' or ''M23M56'').');
end

setup_criticality_manuscript_paths('session_lfp_criticality');
paths = get_paths();

subjectNameForLoad = '';
if exist('subjectName', 'var') && ~isempty(subjectName)
    subjectNameForLoad = subjectName;
end

opts = neuro_behavior_options();
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.firingRateCheckTime = [];

lfpCleanParams = struct();
lfpCleanParams.spikeThresh = 4;
lfpCleanParams.spikeWinSize = 50;
lfpCleanParams.notchFreqs = [60 120 180];
lfpCleanParams.lowpassFreq = lfpLowpassHz;
lfpCleanParams.useHampel = true;
lfpCleanParams.hampelK = 5;
lfpCleanParams.hampelNsigma = 3;
lfpCleanParams.detrendOrder = 'linear';

fprintf('\n=== Session LFP Criticality ===\n');
fprintf('Session [%s]: %s\n', sessionType, sessionName);
fprintf('Area: %s\n', brainArea);
fprintf('LFP clean: artifact removal, zero-mean window, lowpass %.0f Hz\n', lfpLowpassHz);
fprintf('Avalanches: Klaus nLFP / binned-power cutoff = %.1f SD\n', nSdCutoff);
fprintf('d2 windows: %.1f s, step %.1f s; raw bin %.0f ms\n', ...
    d2Window, d2Step, rawD2BinSize * 1000);

loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
loadArgs = [loadArgs, {'lfpCleanParams', lfpCleanParams, 'bands', bands}];
dataStruct = load_session_data(sessionType, 'lfp', loadArgs{:});

% Loaders apply session-metadata floors to opts.collectStart / collectEnd.
% The LFP matrix from load_data already starts at that collectStart; reach
% files still start at t = 0 and are trimmed in prepare_area_lfp_for_criticality.
if isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectStart') ...
        && ~isempty(dataStruct.opts.collectStart)
    collectStart = dataStruct.opts.collectStart;
end
if isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectEnd') ...
        && ~isempty(dataStruct.opts.collectEnd)
    collectEnd = dataStruct.opts.collectEnd;
end

[dataStruct, areaLabel, collectStart, collectEnd] = prepare_area_lfp_for_criticality( ...
    dataStruct, brainArea, brainAreaCombinations, collectStart, collectEnd, ...
    bands, lfpCleanParams, rawD2BinSize);

fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', ...
    collectStart, collectEnd, (collectEnd - collectStart) / 60);
fprintf('Analysis trace: %s, %d samples at %.0f Hz, mean = %.3g\n', ...
    areaLabel, size(dataStruct.lfpPerArea, 1), dataStruct.opts.fsLfp, ...
    mean(dataStruct.lfpPerArea(:, 1)));

%% d2 on raw LFP and binned power
d2Config = struct();
d2Config.slidingWindowSize = d2Window;
d2Config.stepSize = d2Step;
d2Config.analyzeD2 = true;
d2Config.analyzeDFA = false;
d2Config.enablePermutations = false;
d2Config.makePlots = false;
d2Config.saveResults = false;
d2Config.plotBinnedEnvelopes = false;
d2Config.plotRawLfp = false;
d2Config.binnedSignalField = 'binnedPower';
d2Config.pOrder = pOrder;
d2Config.critType = critType;
d2Config.minSegmentLength = 50;

d2Results = criticality_lfp_analysis(dataStruct, d2Config);
d2Plot = collect_lfp_d2_series(d2Results, useLog10D2);
print_lfp_d2_summary(d2Plot, useLog10D2);

figD2 = plot_lfp_d2_overlap(d2Plot, sessionType, sessionName, areaLabel, ...
    d2Window, collectStart, collectEnd, useLog10D2, plotConfig);

%% Avalanches on raw LFP and each binned power band
[clausetPlfitPath, plfit2023Path] = resolve_power_law_paths();
fitConfig = struct();
fitConfig.powerLawFitMethod = powerLawFitMethod;
fitConfig.clausetPlfitPath = clausetPlfitPath;
fitConfig.plfit2023Path = plfit2023Path;
fitConfig.gofThreshold = gofThreshold;
fitConfig.runClausetPlpva = runClausetPlpva;
fitConfig.compareTailModels = compareTailModels;

avPlot = run_lfp_avalanches(dataStruct, nSdCutoff, fitConfig);
print_lfp_avalanche_summary(avPlot);

figAv = plot_lfp_avalanche_distributions(avPlot, sessionName, areaLabel, ...
    nSdCutoff, lfpLowpassHz, plotConfig);

if saveFigure
    saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end
    collectTag = format_collect_tag(collectStart, collectEnd);
    areaTag = matlab.lang.makeValidName(areaLabel);
    d2Base = sprintf('session_lfp_d2_%s_%s_win%.0fs_%s', ...
        sessionName, areaTag, d2Window, collectTag);
    if useLog10D2
        d2Base = [d2Base, '_log10'];
    end
    exportgraphics(figD2, fullfile(saveDir, [d2Base, '.png']), 'Resolution', 300);
    exportgraphics(figD2, fullfile(saveDir, [d2Base, '.eps']), 'ContentType', 'vector');
    fprintf('\nSaved figure: %s\n', fullfile(saveDir, d2Base));

    avBase = sprintf('session_lfp_avalanches_%s_%s_%.1fsd_lp%.0f_%s', ...
        sessionName, areaTag, nSdCutoff, lfpLowpassHz, collectTag);
    exportgraphics(figAv, fullfile(saveDir, [avBase, '.png']), 'Resolution', 300);
    exportgraphics(figAv, fullfile(saveDir, [avBase, '.eps']), 'ContentType', 'vector');
    fprintf('Saved figure: %s\n', fullfile(saveDir, avBase));
end

fprintf('\n=== Done ===\n');

%% Local functions

function [dataStruct, areaLabel, collectStart, collectEnd] = prepare_area_lfp_for_criticality( ...
    dataStruct, brainArea, combinations, collectStart, collectEnd, bands, lfpCleanParams, rawD2BinSize)
% PREPARE_AREA_LFP_FOR_CRITICALITY - One area, zero-mean, artifact-cleaned, re-binned
%
% Variables:
%   dataStruct     - LFP session from load_session_data (already passed through
%                    clean_lfp_artifacts at lfpCleanParams.lowpassFreq)
%   brainArea      - Requested area or combined-area name
%   combinations   - Cell of struct('name', ..., 'areas', {{...}})
%   collectStart   - Requested start (s)
%   collectEnd     - Requested end (s); [] = end of the trace after trim rules
%   bands          - Frequency bands for lfp_bin_bandpower
%   lfpCleanParams - Cleaning parameters (lowpass already applied at load)
%   rawD2BinSize   - Bin width (s) stored as dataStruct.lfpBinSize for raw d2
%
% Goal:
%   Restrict LFP to the requested area (average channels for a combined area),
%   keep the collect window, subtract the mean so the Klaus threshold is about
%   zero, and recompute binned band power on that trace.

if ~isfield(dataStruct, 'lfpPerArea') || isempty(dataStruct.lfpPerArea)
    error('session_lfp_criticality:NoLfp', 'Loaded session has no lfpPerArea.');
end
if ~isfield(dataStruct, 'opts') || ~isfield(dataStruct.opts, 'fsLfp') || isempty(dataStruct.opts.fsLfp)
    error('session_lfp_criticality:NoFs', 'opts.fsLfp is missing.');
end

fs = dataStruct.opts.fsLfp;
[trace, areaLabel] = select_lfp_area_trace(dataStruct.lfpPerArea, dataStruct.areas, ...
    brainArea, combinations);
[trace, collectStart, collectEnd] = trim_lfp_collect_window(trace, fs, ...
    dataStruct.sessionType, collectStart, collectEnd);
trace = zero_mean_lfp(trace);

dataStruct.lfpPerArea = trace;
dataStruct.areas = {areaLabel};
dataStruct.areasToTest = 1;
dataStruct.opts.collectStart = collectStart;
dataStruct.opts.collectEnd = collectEnd;
dataStruct.opts.fsLfp = fs;
if ~isfield(dataStruct, 'dataType') || isempty(dataStruct.dataType)
    dataStruct.dataType = dataStruct.sessionType;
end
dataStruct.bands = bands;
dataStruct = compute_lfp_binned_envelopes(dataStruct, dataStruct.opts, lfpCleanParams, bands);
dataStruct.lfpBinSize = rawD2BinSize;
end

function [trace, areaLabel] = select_lfp_area_trace(lfpPerArea, areaNames, brainArea, combinations)
% SELECT_LFP_AREA_TRACE - One column, or the mean of a combined area
%
% Variables:
%   lfpPerArea   - [samples x areas] LFP matrix
%   areaNames    - Cell of area names, one per column
%   brainArea    - Requested name
%   combinations - Combined-area definitions
%
% Goal:
%   Return the LFP used for this analysis. Combined names (for example M23M56)
%   average the source columns.

brainArea = char(brainArea);
combo = lookup_lfp_area_combination(brainArea, combinations);
if ~isempty(combo)
    sourceIdx = zeros(1, numel(combo.areas));
    for iArea = 1:numel(combo.areas)
        sourceIdx(iArea) = find(strcmp(areaNames, combo.areas{iArea}), 1);
        if isempty(sourceIdx(iArea))
            error('session_lfp_criticality:MissingSourceArea', ...
                'Combined area %s needs %s, which is not in this session.', ...
                combo.name, combo.areas{iArea});
        end
    end
    trace = mean(lfpPerArea(:, sourceIdx), 2);
    areaLabel = combo.name;
    fprintf('  LFP area %s = mean of %s\n', areaLabel, strjoin(combo.areas, '+'));
    return;
end

areaIdx = find(strcmp(areaNames, brainArea), 1);
if isempty(areaIdx)
    error('session_lfp_criticality:MissingArea', ...
        'Brain area "%s" is not in this session (%s).', ...
        brainArea, strjoin(areaNames, ', '));
end
trace = lfpPerArea(:, areaIdx);
areaLabel = areaNames{areaIdx};
fprintf('  Restricting LFP to area: %s\n', areaLabel);
end

function combo = lookup_lfp_area_combination(brainArea, combinations)
% LOOKUP_LFP_AREA_COMBINATION - Match brainArea to a merged-area definition

combo = [];
if isempty(combinations)
    return;
end
for iCombo = 1:numel(combinations)
    entry = combinations{iCombo};
    if isstruct(entry) && isfield(entry, 'name') && strcmpi(entry.name, brainArea)
        combo = entry;
        combo.areas = cellstr(combo.areas);
        return;
    end
end
end

function [trace, collectStart, collectEnd] = trim_lfp_collect_window( ...
    trace, fs, sessionType, collectStart, collectEnd)
% TRIM_LFP_COLLECT_WINDOW - Keep samples inside the requested collect window
%
% Variables:
%   trace        - LFP column
%   fs           - Sampling rate (Hz)
%   sessionType  - Reach LFP is loaded from t = 0; other loaders already
%                  slice load_data from collectStart, so sample 1 is collectStart
%   collectStart - Requested start (s)
%   collectEnd   - Requested end (s); [] = last sample
%
% Goal:
%   Return the analysis segment and the absolute times of its first and last
%   samples. collectEnd is filled in when the caller left it empty.

if nargin < 4 || isempty(collectStart)
    collectStart = 0;
end
nSamples = size(trace, 1);
if strcmpi(sessionType, 'reach')
    signalStart = 0;
else
    signalStart = collectStart;
end

i0 = 1 + round((collectStart - signalStart) * fs);
if isempty(collectEnd)
    i1 = nSamples;
else
    i1 = round((collectEnd - signalStart) * fs);
end
i0 = max(1, min(nSamples, i0));
i1 = max(i0, min(nSamples, i1));
trace = trace(i0:i1, :);
collectStart = signalStart + (i0 - 1) / fs;
collectEnd = signalStart + i1 / fs;
end

function trace = zero_mean_lfp(trace)
% ZERO_MEAN_LFP - Drop non-finite samples' contribution and subtract the mean
%
% Variables:
%   trace - LFP column already cleaned at load (detrend, artifacts, lowpass)
%
% Goal:
%   The analysis window is zero-mean so the 2.5 SD avalanche cutoff is
%   symmetric about zero. Artifact removal itself is done once, at load.

trace = trace(:, 1);
bad = ~isfinite(trace);
if any(bad)
    trace(bad) = 0;
end
trace = trace - mean(trace);
end

function d2Plot = collect_lfp_d2_series(d2Results, useLog10D2)
% COLLECT_LFP_D2_SERIES - Raw and per-band window-wise d2 for one area
%
% Variables:
%   d2Results  - Output of criticality_lfp_analysis (one area)
%   useLog10D2 - If true, store log10(d2) and drop non-positive values
%
% Goal:
%   Build the series that are overlaid on the d2 distribution figure.
%   Order is raw LFP, then each band.

d2Plot = struct();
d2Plot.names = {};
d2Plot.values = {};
d2Plot.binSizeSec = [];
areaIdx = 1;

if isfield(d2Results, 'd2Lfp') && ~isempty(d2Results.d2Lfp) ...
        && numel(d2Results.d2Lfp) >= areaIdx && ~isempty(d2Results.d2Lfp{areaIdx})
    rawVals = d2Results.d2Lfp{areaIdx}{1}(:);
    rawBin = nan;
    if isfield(d2Results, 'lfpBinSize') && ~isempty(d2Results.lfpBinSize)
        rawBin = d2Results.lfpBinSize(1);
    end
    d2Plot = append_d2_series(d2Plot, 'Raw LFP', rawVals, rawBin, useLog10D2);
end

if isfield(d2Results, 'd2') && numel(d2Results.d2) >= areaIdx && ~isempty(d2Results.d2{areaIdx})
    numBands = numel(d2Results.d2{areaIdx});
    for b = 1:numBands
        bandName = sprintf('Band %d', b);
        if isfield(d2Results, 'bands') && size(d2Results.bands, 1) >= b
            bandName = char(d2Results.bands{b, 1});
        end
        bandBin = nan;
        if isfield(d2Results, 'bandBinSizes') && numel(d2Results.bandBinSizes) >= b
            bandBin = d2Results.bandBinSizes(b);
        end
        bandVals = d2Results.d2{areaIdx}{b};
        if isempty(bandVals)
            bandVals = nan;
        end
        d2Plot = append_d2_series(d2Plot, bandName, bandVals(:), bandBin, useLog10D2);
    end
end
end

function d2Plot = append_d2_series(d2Plot, name, values, binSizeSec, useLog10D2)
% APPEND_D2_SERIES - Keep finite d2 values for one signal

values = values(:);
if useLog10D2
    values = log10_safe_numeric(values);
end
values = values(isfinite(values));
d2Plot.names{end + 1} = name;
d2Plot.values{end + 1} = values;
d2Plot.binSizeSec(end + 1) = binSizeSec;
end

function print_lfp_d2_summary(d2Plot, useLog10D2)
% PRINT_LFP_D2_SUMMARY - Window count and mean d2 for each signal

fprintf('\n=== LFP d2 summary ===\n');
valueName = 'd2';
if useLog10D2
    valueName = 'log10(d2)';
end
for iSig = 1:numel(d2Plot.names)
    vals = d2Plot.values{iSig};
    if isempty(vals)
        fprintf('  %s: no finite %s\n', d2Plot.names{iSig}, valueName);
    else
        fprintf('  %s: %d windows, mean %s = %.4f (bin %.0f ms)\n', ...
            d2Plot.names{iSig}, numel(vals), valueName, mean(vals), ...
            d2Plot.binSizeSec(iSig) * 1000);
    end
end
end

function fig = plot_lfp_d2_overlap(d2Plot, sessionType, sessionName, areaLabel, ...
    d2Window, collectStart, collectEnd, useLog10D2, plotConfig)
% PLOT_LFP_D2_OVERLAP - One axes, raw and band d2 densities in different colors
%
% Variables:
%   d2Plot     - From collect_lfp_d2_series
%   plotConfig - Manuscript fonts and histogram alpha
%
% Goal:
%   Overlay probability densities of window-wise d2. Raw LFP and each binned
%   power band share bin edges and a legend.

if nargin < 9 || isempty(plotConfig)
    plotConfig = fill_manuscript_plot_config();
end

allVals = [];
for iSig = 1:numel(d2Plot.values)
    allVals = [allVals; d2Plot.values{iSig}(:)]; %#ok<AGROW>
end
allVals = allVals(isfinite(allVals));
if isempty(allVals)
    error('session_lfp_criticality:NoD2', ...
        'No finite d2 values to plot. Check window length and LFP length.');
end

[binEdges, xMin, xMax] = build_shared_histogram_bin_edges(allVals, 28);
if useLog10D2
    xLabelText = 'log_{10}(d2)';
    labelInterpreter = 'tex';
else
    xLabelText = 'd2';
    labelInterpreter = 'none';
end

colors = lfp_signal_colors(numel(d2Plot.names));
fig = figure('Color', 'w', 'Position', [120 140 760 480], ...
    'Name', sprintf('LFP d2 | %s | %s', sessionName, areaLabel));
ax = axes(fig);
hold(ax, 'on');
for iSig = 1:numel(d2Plot.names)
    vals = d2Plot.values{iSig};
    if numel(vals) < 1
        continue;
    end
    binMs = d2Plot.binSizeSec(iSig) * 1000;
    if isfinite(binMs)
        displayName = sprintf('%s, %.0f ms bin (n=%d)', d2Plot.names{iSig}, binMs, numel(vals));
    else
        displayName = sprintf('%s (n=%d)', d2Plot.names{iSig}, numel(vals));
    end
    histogram(ax, vals, binEdges, 'Normalization', 'pdf', ...
        'FaceColor', colors(iSig, :), 'FaceAlpha', plotConfig.histogramFaceAlpha, ...
        'EdgeColor', 'none', 'DisplayName', displayName);
end
xlim(ax, [xMin, xMax]);
apply_manuscript_axes_style(ax, plotConfig, xLabelText, 'Probability density', '', ...
    labelInterpreter);
legend(ax, 'Location', 'northeast', 'FontSize', plotConfig.legendFontSize, ...
    'Interpreter', 'none');
grid(ax, 'on');
hold(ax, 'off');

sgtitle(fig, sprintf('LFP d2 | %s | %s | %s | %.0f s windows%s', ...
    areaLabel, sessionType, sessionName, d2Window, format_title_window(collectStart, collectEnd)), ...
    'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

function avPlot = run_lfp_avalanches(dataStruct, nSdCutoff, fitConfig)
% RUN_LFP_AVALANCHES - Klaus avalanches for raw LFP and each binned power band
%
% Variables:
%   dataStruct - Prepared single-area LFP struct (lfpPerArea, binnedPower, bands)
%   nSdCutoff  - SD cutoff (2.5)
%   fitConfig  - Fields for fit_avalanche_power_law
%
% Goal:
%   Detect avalanches, fit size (threshold units) and duration (bins), and
%   keep Δt so duration CCDFs can be drawn in milliseconds.

avPlot = struct();
avPlot.names = {};
avPlot.results = {};
fs = dataStruct.opts.fsLfp;
rawAv = detect_klaus_avalanches(dataStruct.lfpPerArea(:, 1), 1 / fs, nSdCutoff, 'nlfp');
rawAv = fit_lfp_avalanche_laws(rawAv, fitConfig);
rawAv.name = 'Raw LFP';
avPlot = append_av_result(avPlot, rawAv);

numBands = size(dataStruct.bands, 1);
for b = 1:numBands
    bandName = char(dataStruct.bands{b, 1});
    bandSignal = dataStruct.binnedPower{1}{b};
    bandAv = detect_klaus_avalanches(bandSignal, dataStruct.bandBinSizes(b), ...
        nSdCutoff, 'binnedPower');
    bandAv = fit_lfp_avalanche_laws(bandAv, fitConfig);
    bandAv.name = bandName;
    avPlot = append_av_result(avPlot, bandAv);
end
end

function avPlot = append_av_result(avPlot, av)
% APPEND_AV_RESULT - Store one signal's avalanche struct

avPlot.names{end + 1} = av.name;
avPlot.results{end + 1} = av;
end

function av = fit_lfp_avalanche_laws(av, fitConfig)
% FIT_LFP_AVALANCHE_LAWS - Power-law fits for size and duration (in bins)
%
% Variables:
%   av        - detect_klaus_avalanches output
%   fitConfig - powerLawFitMethod and toolbox paths
%
% Goal:
%   Fit size in threshold units and duration in bins. Duration plot limits
%   are scaled to milliseconds with deltaT. Exponent is unchanged by that scale.

av.sizeFit = empty_power_law_fit();
av.durFit = empty_power_law_fit();
av.scalingRelation = nan;
if numel(av.sizes) >= 2
    av.sizeFit = fit_avalanche_power_law(av.sizes, fitConfig);
end
if numel(av.durations) >= 2
    av.durFit = fit_avalanche_power_law(av.durations, fitConfig);
end
tau = av.sizeFit.exponent;
alpha = av.durFit.exponent;
av.scalingRelation = compute_avalanche_scaling_relation(tau, alpha);
end

function fitResult = empty_power_law_fit()
% EMPTY_POWER_LAW_FIT - Placeholder when there are too few avalanches

fitResult = struct('exponent', nan, 'fitMin', nan, 'fitMax', nan, ...
    'decades', nan, 'method', '', 'tailComparison', struct());
end

function print_lfp_avalanche_summary(avPlot)
% PRINT_LFP_AVALANCHE_SUMMARY - Event counts, exponents, and tail-model tests

fprintf('\n=== LFP avalanche summary (Klaus cutoff) ===\n');
for iSig = 1:numel(avPlot.results)
    av = avPlot.results{iSig};
    fprintf('\n%s (%s)\n', av.name, av.mode);
    fprintf('  events=%d, avalanches=%d, dt=%.4g s, active fraction=%.3f\n', ...
        av.nEvents, av.nAvalanches, av.deltaT, av.fractionActive);
    fprintf('  Size:  tau = %.3f, x in [%.3g, %.3g]\n', ...
        av.sizeFit.exponent, av.sizeFit.fitMin, av.sizeFit.fitMax);
    durScale = av.deltaT;
    if ~(isfinite(durScale) && durScale > 0)
        durScale = nan;
    end
    fprintf('  Dur:   alpha = %.3f, x in [%.3g, %.3g] s\n', ...
        av.durFit.exponent, av.durFit.fitMin * durScale, av.durFit.fitMax * durScale);
    fprintf('  Scaling (alpha-1)/(tau-1) = %.3f\n', av.scalingRelation);
    if isfield(av.sizeFit, 'tailComparison')
        print_avalanche_tail_comparison(av.sizeFit.tailComparison, 'Size');
    end
    if isfield(av.durFit, 'tailComparison')
        print_avalanche_tail_comparison(av.durFit.tailComparison, 'Duration');
    end
end
end

function fig = plot_lfp_avalanche_distributions(avPlot, sessionName, areaLabel, ...
    nSdCutoff, lfpLowpassHz, plotConfig)
% PLOT_LFP_AVALANCHE_DISTRIBUTIONS - Size and duration CCDFs, one column per signal
%
% Variables:
%   avPlot     - From run_lfp_avalanches
%   nSdCutoff  - SD cutoff printed in the title
%   plotConfig - drawPowerLawFit, marker size, fit line width
%
% Goal:
%   Same figure, separate axes for raw LFP and each binned power band.
%   Top row is avalanche size; bottom row is duration in milliseconds.

if nargin < 6 || isempty(plotConfig)
    plotConfig = fill_manuscript_plot_config();
end
plotConfig = fill_lfp_avalanche_plot_config(plotConfig);

nSignals = numel(avPlot.results);
colors = lfp_signal_colors(nSignals);
fig = figure('Color', 'w', ...
    'Position', [80 80 max(900, 280 * nSignals) 640], ...
    'Name', sprintf('LFP avalanches | %s | %s', sessionName, areaLabel));
tileLayout = tiledlayout(fig, 2, nSignals, 'TileSpacing', 'compact', 'Padding', 'compact');

for iSig = 1:nSignals
    av = avPlot.results{iSig};
    lineColor = colors(iSig, :);

    axSize = nexttile(tileLayout, iSig);
    plot_lfp_ccdf(axSize, av.sizes, lineColor, plotConfig);
    overlay_lfp_power_law(axSize, av.sizes, av.sizeFit, 1, lineColor, plotConfig);
    sizeTitle = sprintf('%s | n=%d | tau=%.2f', av.name, av.nAvalanches, av.sizeFit.exponent);
    apply_manuscript_axes_style(axSize, plotConfig, 'Size (threshold units)', 'CCDF', ...
        sizeTitle, 'none');
    use_log_scales_if_data(axSize, av.sizes);
    grid(axSize, 'on');

    axDur = nexttile(tileLayout, nSignals + iSig);
    durMs = av.durationsSec * 1000;
    msPerBin = av.deltaT * 1000;
    plot_lfp_ccdf(axDur, durMs, lineColor, plotConfig);
    overlay_lfp_power_law(axDur, durMs, av.durFit, msPerBin, lineColor, plotConfig);
    durTitle = sprintf('alpha=%.2f | dt=%.3g ms', av.durFit.exponent, msPerBin);
    apply_manuscript_axes_style(axDur, plotConfig, 'Duration (ms)', 'CCDF', ...
        durTitle, 'none');
    use_log_scales_if_data(axDur, durMs);
    grid(axDur, 'on');
end

sgtitle(tileLayout, sprintf(['LFP avalanches | %s | %s | %.1f SD | %.0f Hz lowpass | ', ...
    'Klaus nLFP (raw) and binned power'], sessionName, areaLabel, nSdCutoff, lfpLowpassHz), ...
    'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

function use_log_scales_if_data(ax, values)
% USE_LOG_SCALES_IF_DATA - Log-log axes only when the CCDF has positive values
%
% Variables:
%   ax     - Axes that already hold the CCDF
%   values - Sizes or durations drawn on ax
%
% Goal:
%   Avoid log-scale warnings when a signal produced too few avalanches.

values = values(isfinite(values) & values > 0);
if numel(values) >= 2
    set(ax, 'XScale', 'log', 'YScale', 'log');
end
end

function plot_lfp_ccdf(ax, values, lineColor, plotConfig)
% PLOT_LFP_CCDF - Empirical complementary CDF on the current axes
%
% Variables:
%   ax         - Target axes
%   values     - Positive avalanche sizes or durations
%   lineColor  - RGB
%   plotConfig - observedMarkerSize, observedMarkerFaceAlpha
%
% Goal:
%   One marker per unique value, y = P(X >= x).

hold(ax, 'on');
values = values(:);
values = values(isfinite(values) & values > 0);
if numel(values) < 2
    text(ax, 0.5, 0.5, 'too few avalanches', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'Color', [0.45 0.45 0.45]);
    hold(ax, 'off');
    return;
end
values = sort(values);
n = numel(values);
[uniqueVals, firstIdx] = unique(values, 'stable');
ccdf = (n - firstIdx + 1) / n;
scatter(ax, uniqueVals, ccdf, plotConfig.observedMarkerSize ^ 2, lineColor, 'filled', ...
    'MarkerFaceAlpha', plotConfig.observedMarkerFaceAlpha, 'HandleVisibility', 'off');
hold(ax, 'off');
end

function overlay_lfp_power_law(ax, plotValues, fitResult, unitScale, lineColor, plotConfig)
% OVERLAY_LFP_POWER_LAW - CCDF power-law segment on [xmin, xmax]
%
% Variables:
%   ax          - Target axes
%   plotValues  - Values in the same units as the scatter (size, or duration in ms)
%   fitResult   - exponent, fitMin, fitMax from fit_avalanche_power_law
%   unitScale   - Multiplies fitMin/fitMax into plot units (1 for size, ms/bin for duration)
%   lineColor   - RGB
%   plotConfig  - drawPowerLawFit, fitLineWidth
%
% Goal:
%   Draw P(X>=x) ~ x^(-(exponent-1)) anchored at the empirical CCDF at xmin.

if ~isfield(plotConfig, 'drawPowerLawFit') || ~plotConfig.drawPowerLawFit
    return;
end
if ~isstruct(fitResult) || ~isfield(fitResult, 'exponent')
    return;
end
exponent = fitResult.exponent;
if ~(isfinite(exponent) && exponent > 1 && isfinite(unitScale) && unitScale > 0)
    return;
end
fitMin = fitResult.fitMin * unitScale;
fitMax = fitResult.fitMax * unitScale;
if ~(isfinite(fitMin) && isfinite(fitMax) && fitMax > fitMin && fitMin > 0)
    return;
end

plotValues = plotValues(:);
plotValues = plotValues(isfinite(plotValues) & plotValues > 0);
if numel(plotValues) < 2
    return;
end
yAtMin = mean(plotValues >= fitMin);
if ~(isfinite(yAtMin) && yAtMin > 0)
    return;
end

xFit = logspace(log10(fitMin), log10(fitMax), 80);
yFit = (xFit / fitMin) .^ (-(exponent - 1)) * yAtMin;
hold(ax, 'on');
plot(ax, xFit, yFit, '-', 'Color', lineColor, 'LineWidth', plotConfig.fitLineWidth, ...
    'HandleVisibility', 'off');
hold(ax, 'off');
end

function plotConfig = fill_lfp_avalanche_plot_config(plotConfig)
% FILL_LFP_AVALANCHE_PLOT_CONFIG - Marker and fit defaults for avalanche CCDFs

if ~isfield(plotConfig, 'drawPowerLawFit') || isempty(plotConfig.drawPowerLawFit)
    plotConfig.drawPowerLawFit = true;
end
if ~isfield(plotConfig, 'observedMarkerSize') || isempty(plotConfig.observedMarkerSize)
    plotConfig.observedMarkerSize = 5;
end
if ~isfield(plotConfig, 'fitLineWidth') || isempty(plotConfig.fitLineWidth)
    plotConfig.fitLineWidth = 2;
end
if ~isfield(plotConfig, 'observedMarkerFaceAlpha') || isempty(plotConfig.observedMarkerFaceAlpha)
    plotConfig.observedMarkerFaceAlpha = 0.45;
end
end

function colors = lfp_signal_colors(nSignals)
% LFP_SIGNAL_COLORS - Distinct RGB rows; row 1 is raw LFP
%
% Variables:
%   nSignals - Number of traces (raw + bands)
%
% Goal:
%   Same color order on the d2 overlay and the avalanche columns.

palette = [ ...
    0.12 0.12 0.12; ...
    0.20 0.45 0.82; ...
    0.90 0.49 0.13; ...
    0.16 0.62 0.32; ...
    0.55 0.25 0.72; ...
    0.84 0.22 0.35];
if nSignals <= size(palette, 1)
    colors = palette(1:nSignals, :);
    return;
end
colors = lines(nSignals);
colors(1, :) = palette(1, :);
end

function y = log10_safe_numeric(x)
% LOG10_SAFE_NUMERIC - log10 with NaN for non-positive values

validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end

function tag = format_collect_tag(collectStart, collectEnd)
% FORMAT_COLLECT_TAG - Filename piece for the collect window

if isempty(collectEnd)
    tag = sprintf('%.0f-full', collectStart);
else
    tag = sprintf('%.0f-%.0f', collectStart, collectEnd);
end
end

function suffix = format_title_window(collectStart, collectEnd)
% FORMAT_TITLE_WINDOW - Collect-window suffix for figure titles

if isempty(collectStart)
    collectStart = 0;
end
if isempty(collectEnd)
    suffix = sprintf(' [%.0f s-end]', collectStart);
else
    suffix = sprintf(' [%.0f-%.0f s]', collectStart, collectEnd);
end
end
