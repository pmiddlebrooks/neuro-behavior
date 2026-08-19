%%
% Session d2 Distributions (Manuscript)
%
% For one session, runs the same AR/d2 pipeline as criticality_ar_across_tasks.m
% (non-overlapping windows) and plots overlapping probability densities of
% window-wise d2 for real vs shuffled data.
%
% Variables (configure in this section):
%   sessionType      - 'spontaneous', 'interval', 'reach', 'semicircle', 'schall'
%   sessionName      - Session identifier
%   subjectName      - Required for spontaneous/interval/semicircle; '' for reach
%   dataSource       - 'spikes' or 'lfp'
%   collectStart     - Window start (seconds from session onset)
%   collectEnd       - Window end (seconds); [] = session end. If set past the
%                      recording, plots stop at the session end.
%   d2Window         - Non-overlapping window length (seconds)
%   d2WindowAlign    - Timeline timestamp: 'center' (default) or 'leadingEdge'
%                      leadingEdge places d2 at the window end (trailing d2Window)
%   brainArea              - Single or merged area (e.g. 'M56', 'M23M56'); '' uses all valid areas
%   brainAreaCombinations  - Merged areas: struct('name', 'M23M56', 'areas', {{'M23','M56'}})
%   useLog10D2       - If true, plot log10(d2) and log10(shuffled d2)
%   useSubsampling   - If true, d2 per window = mean across neuron subsamples
%   nSubsamples, nNeuronsSubsample, minNeuronsMultiple - subsampling (run_criticality_ar.m)
%   nPermutations    - Number of circular permutations per window for shuffled d2
%   plotD2PopActivity - If true, scatter d2 vs mean pop activity (+ shuffled)
%   plotD2Timeline   - If true, plot mean pop per d2 window, d2, and ethogram vs time
%                      Semicircle ethogram: TaskMatrix outcome lines + leave-home/poke/end fills
%                      Interval (no bhv labels yet): correct/error beam-break schematics;
%                      engagement fills when splitByEngagement
%   useRelativeTime  - If true, timeline x-axis is relative to collectStart (default false)
%   binSize          - Spike bin width (s) for d2 analysis (and window popActivity)
%   saveFigure       - Export PNG/EPS to dropPath/criticality_manuscript
%   plotConfig       - Axis fonts/line widths (see fill_manuscript_plot_config)
%   splitExcitatoryInhibitory - If true, run combined (E+I), excitatory, and inhibitory;
%                               also plots mean +/- SEM summary across windows;
%                               d2 vs pop activity on one figure (shared y-axis)
%   widthCutoff        - Peak-to-trough width threshold in ms (narrow <= cutoff = I)
%                        Waveforms: spontaneous/interval waveforms.mat; reach
%                        reach_task/data/WaveformDATA/*_Neural_WFs.mat
%   splitByEngagement  - If true (reach/interval/semicircle), also run engaged vs
%                        non-engaged d2 distributions via the engagement modules
%   engagementBufferBefore - Seconds before each reach/beam-break = engaged (default 1)
%   engagementBufferAfter  - Seconds after each reach/beam-break = engaged (default 1)
%   engagementBuffer       - Legacy symmetric alias; if before/after unset, sets both
%   minNonEngagedWindow - Min gap without events (s) for non-engaged segments
%   minTimeNonEngaged  - When splitByEngagement, min total non-engaged time (s)
%                        to plot; 0 = no filter. Below this, non-engaged d2 is
%                        blanked (computed but omitted from plots).
%   absorbSingleEvents - Merge isolated single events into non-engaged gaps
%
% Goal:
%   Visualize real d2 vs shuffled d2 distributions for one session across
%   windows, where shuffled values are the mean across permutations per window.

%% Configuration
% Prefer session identity already set in the workspace (e.g. scratch.m batch).
% Otherwise default to a semicircle example session.
if ~exist('sessionType', 'var') || isempty(sessionType)
    % sessionType = 'interval';
    % subjectName = 'ey9166';
    % sessionName = 'ey9166_2026_04_03';

    sessionType = 'semicircle';
    subjectName = 'AS1';
    sessionName = 'AS1_0618_WellLearned';
    % sessionName = 'AS1_0623_TransitionAfterCompletedTrial_80';
    % sessionName = 'AS1_0624_PoorlyLearned';
end
dataSource = 'spikes';
collectStart = 0;
collectEnd = [];
collectEnd = 120*60;
d2Window = 60;  % seconds; non-overlapping windows
d2WindowAlign = 'center';  % 'center' | 'leadingEdge' (window is the trailing d2Window)
brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
useLog10D2 = true;
useSubsampling = true;
nSubsamples = 25;
nNeuronsSubsample = 70;
minNeuronsMultiple = 1.1;
nPermutations = 2;  % circular shuffles per window for shuffled d2 distribution
plotD2PopActivity = true;
plotD2Timeline = true;  % mean pop per d2 window | d2 vs time | ethogram
useRelativeTime = false;  % false: absolute session time (default); true: t=0 at collectStart
binSize = 0.025;  % s; spike binning for d2 (and window mean popActivity)
saveFigure = false;

plotConfig = fill_manuscript_plot_config();
splitExcitatoryInhibitory = false;
widthCutoff = 0.35;  % ms; peak-to-trough width (narrow <= cutoff = inhibitory)

% Optional engaged vs non-engaged (reach / interval / semicircle only)
splitByEngagement = false;
engagementBufferBefore = 3;  % s before each reach/beam-break = engaged
engagementBufferAfter = 1;   % s after each reach/beam-break = engaged
minNonEngagedWindow = 30;    % min gap (s) for non-engaged segments
absorbSingleEvents = true;   % merge isolated single events into non-engaged gaps
minTimeNonEngaged = 180;     % min total non-engaged time (s) to plot; 0 = no filter
% Below minTimeNonEngaged, non-engaged d2 stays computed but is blanked in plots

opts = neuro_behavior_options();
opts.firingRateCheckTime = 5 * 60;
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.25;
opts.maxFiringRate = 200;

analysisConfig = struct();
analysisConfig.slidingWindowSize = d2Window;
analysisConfig.stepSize = 2;
analysisConfig.binSize = binSize;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.analyzeD2 = true;
analysisConfig.analyzeMrBr = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 4;
analysisConfig.enablePermutations = nPermutations > 0;
analysisConfig.nShuffles = nPermutations;
analysisConfig.normalizeD2 = true;
analysisConfig.useLog10D2 = useLog10D2;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.pOrder = 10;
analysisConfig.critType = 2;
analysisConfig.minSpikesPerBin = 2.5;
analysisConfig.minBinsPerWindow = 1000;
analysisConfig.maxSpikesPerBin = 100;
analysisConfig.nMinNeurons = 20;
analysisConfig.useSubsampling = useSubsampling;
analysisConfig.nSubsamples = nSubsamples;
analysisConfig.nNeuronsSubsample = nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = minNeuronsMultiple;

% Paths
setup_criticality_manuscript_paths('session_d2_distributions');
paths = get_paths();

d2WindowAlign = normalize_d2_window_align(d2WindowAlign);

fprintf('\n=== Session d2 Distributions ===\n');
fprintf('Session [%s]: %s\n', sessionType, sessionName);
fprintf('d2 windows: %.1f s (%s); binSize: %.3f s; nPermutations: %d\n', ...
    d2Window, d2WindowAlign, binSize, nPermutations);
fprintf('useLog10D2: %d\n', useLog10D2);
fprintf('splitByEngagement: %d\n', splitByEngagement);
if splitByEngagement
    fprintf('minTimeNonEngaged: %.1f s (blank non-engaged below this; 0 = off)\n', ...
        minTimeNonEngaged);
end
if useSubsampling
    fprintf('Subsampling: %d subsets x %d neurons (min neurons x %.2f)\n', ...
        nSubsamples, nNeuronsSubsample, minNeuronsMultiple);
else
    fprintf('Subsampling: off\n');
end
if splitExcitatoryInhibitory
    fprintf('E/I split: on (widthCutoff = %.3f ms)\n', widthCutoff);
end

% Load session and run d2 analysis
subjectNameForLoad = '';
if exist('subjectName', 'var') && ~isempty(subjectName)
    subjectNameForLoad = subjectName;
end
loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

% Use the loaded collect window (includes session-metadata floors such as collectStartMin)
loadedStart = session_d2_scalar_time(session_time_origin(dataStruct), 0, @min);
collectStart = session_d2_scalar_time([collectStart(:); loadedStart], 0, @max);
opts.collectStart = collectStart;

% Requested collectEnd may be longer than this recording; do not plot past session end
if ~isempty(collectEnd)
    sessionEndAbs = session_d2_loaded_session_end(dataStruct);
    if isscalar(sessionEndAbs) && isfinite(sessionEndAbs)
        collectEnd = clamp_collect_end_to_session(collectEnd, sessionEndAbs, collectStart);
        opts.collectEnd = collectEnd;
    end
end

if isempty(collectEnd)
    fprintf('Collect window: [%.1f, full session] s\n', collectStart);
else
    fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', ...
        collectStart, collectEnd, (collectEnd - collectStart) / 60);
end

[dataStruct, areaOk] = apply_manuscript_brain_area_selection(dataStruct, brainArea, brainAreaCombinations, false);
if ~areaOk
    error('Brain area "%s" not available in this session.', brainArea);
end

if splitExcitatoryInhibitory
    eiCheck = check_session_ei_neuron_counts(dataStruct, paths, widthCutoff, brainArea, ...
        brainAreaCombinations, analysisConfig.nMinNeurons);
    if ~eiCheck.isOk
        return;
    end
end

cellTypesToRun = get_session_cell_types_to_run(splitExcitatoryInhibitory);
if splitExcitatoryInhibitory
    eiSummary = init_session_ei_summary({'d2'}, {get_d2_axis_label(useLog10D2)});
    eiPopActivityResults = cell(1, numel(cellTypesToRun));
end

for iCellRun = 1:numel(cellTypesToRun)
    cellType = cellTypesToRun{iCellRun};
    dataStructRun = prepare_session_data_for_cell_type(dataStruct, paths, cellType, widthCutoff, splitExcitatoryInhibitory);

    [dataStructRun, ~] = apply_manuscript_brain_area_selection(dataStructRun, brainArea, brainAreaCombinations);

    results = criticality_ar_analysis(dataStructRun, analysisConfig);

    if ~isempty(brainArea)
        results = filter_ar_results_to_brain_area(results, brainArea);
        if isempty(results.areas)
            error('No d2 results for brain area "%s" (%s).', brainArea, cell_type_label(cellType));
        end
    end

    print_session_d2_summary(results, useLog10D2);

    if splitExcitatoryInhibitory
        eiSummary = set_session_ei_summary_population(eiSummary, cellType, ...
            extract_d2_summary_metric_values(results, useLog10D2));
    end

    % Build distributions and plot
    plotData = build_d2_distribution_data(results, useLog10D2);
    if isempty(plotData.areas)
        error(['No valid d2 distribution data found (%s). Check d2 values and shuffled ' ...
            'permutation outputs for this session.'], cell_type_label(cellType));
    end

    fig = plot_d2_distributions(plotData, sessionType, sessionName, d2Window, collectStart, collectEnd, useLog10D2, plotConfig);
    if splitExcitatoryInhibitory
        sgtitle(fig, sprintf('%s | %s | width cutoff %.3f ms', ...
            sessionName, cell_type_label(cellType), widthCutoff), 'Interpreter', 'none');
    end

    if saveFigure
        saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
        if ~exist(saveDir, 'dir')
            mkdir(saveDir);
        end
        areaTag = format_areas_label(plotData.areas);
        plotBase = sprintf('session_d2_distributions_%s_%s_win%.0fs_%.0f-%.0fs%s', ...
            sessionName, areaTag, d2Window, collectStart, collectEnd, cell_type_file_tag(cellType));
        if useLog10D2
            plotBase = [plotBase, '_log10'];
        end
        exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
        exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
        fprintf('\nSaved figure: %s\n', fullfile(saveDir, plotBase));
    end

    % d2 vs mean population activity (real and shuffled mean per window)
    if plotD2PopActivity
        if splitExcitatoryInhibitory
            eiPopActivityResults{iCellRun} = struct('cellType', cellType, 'results', results);
            print_d2_popactivity_correlations(results, useLog10D2, cell_type_label(cellType));
        else
            figPop = plot_d2_vs_popactivity(results, useLog10D2, d2Window, plotConfig);
            print_d2_popactivity_correlations(results, useLog10D2);
            if saveFigure
                saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
                if ~exist(saveDir, 'dir')
                    mkdir(saveDir);
                end
                areaTag = format_areas_label(plotData.areas);
                plotBase = sprintf('session_d2_vs_popactivity_%s_%s_win%.0fs_%.0f-%.0fs', ...
                    sessionName, areaTag, d2Window, collectStart, collectEnd);
                if useLog10D2
                    plotBase = [plotBase, '_log10'];
                end
                exportgraphics(figPop, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
                exportgraphics(figPop, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
                fprintf('Saved figure: %s\n', fullfile(saveDir, plotBase));
            end
        end
    end

    % popActivity | d2 over time | ethogram (time-aligned)
    if plotD2Timeline
        timelineOpts = struct( ...
            'splitByEngagement', splitByEngagement, ...
            'engagementBufferBefore', engagementBufferBefore, ...
            'engagementBufferAfter', engagementBufferAfter, ...
            'minNonEngagedWindow', minNonEngagedWindow, ...
            'absorbSingleEvents', absorbSingleEvents);
        figTime = plot_d2_pop_ethogram_timeline(dataStruct, results, ...
            collectStart, collectEnd, d2Window, binSize, useLog10D2, plotConfig, ...
            sessionName, cellType, useRelativeTime, d2WindowAlign, timelineOpts);
        if ~isempty(figTime) && saveFigure
            saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
            if ~exist(saveDir, 'dir')
                mkdir(saveDir);
            end
            areaTag = format_areas_label(plotData.areas);
            if isempty(collectEnd)
                collectTag = sprintf('%.0f-full', collectStart);
            else
                collectTag = sprintf('%.0f-%.0f', collectStart, collectEnd);
            end
            plotBase = sprintf('session_d2_timeline_%s_%s_win%.0fs_%ss_%s%s', ...
                sessionName, areaTag, d2Window, collectTag, d2WindowAlign, ...
                cell_type_file_tag(cellType));
            if useLog10D2
                plotBase = [plotBase, '_log10'];
            end
            exportgraphics(figTime, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
            exportgraphics(figTime, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
            fprintf('Saved timeline figure: %s\n', fullfile(saveDir, plotBase));
        end
    end
end

if plotD2PopActivity && splitExcitatoryInhibitory
    figPopEi = plot_d2_vs_popactivity_ei_split(eiPopActivityResults, useLog10D2, d2Window, ...
        plotConfig, sessionName, widthCutoff);
    if saveFigure
        saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
        if ~exist(saveDir, 'dir')
            mkdir(saveDir);
        end
        areaTag = format_areas_label(brainArea);
        if isempty(areaTag)
            areaTag = format_areas_label(eiPopActivityResults{1}.results.areas);
        end
        plotBase = sprintf('session_d2_vs_popactivity_%s_%s_win%.0fs_%.0f-%.0fs_ei_split', ...
            sessionName, areaTag, d2Window, collectStart, collectEnd);
        if useLog10D2
            plotBase = [plotBase, '_log10'];
        end
        exportgraphics(figPopEi, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
        exportgraphics(figPopEi, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
        fprintf('\nSaved E/I pop-activity figure: %s\n', fullfile(saveDir, plotBase));
    end
end

if splitExcitatoryInhibitory
    areaTag = format_areas_label(brainArea);
    if isempty(areaTag)
        areaTag = 'all_areas';
    end
    summaryTitle = sprintf('%s | %s | d2 mean +/- SEM across windows', sessionName, areaTag);
    figEiSummary = plot_session_ei_summary(eiSummary, summaryTitle, get_d2_axis_label(useLog10D2), [], [], plotConfig);
    if saveFigure
        saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
        if ~exist(saveDir, 'dir')
            mkdir(saveDir);
        end
        plotBase = sprintf('session_d2_ei_summary_%s_%s_win%.0fs_%.0f-%.0fs%s', ...
            sessionName, areaTag, d2Window, collectStart, collectEnd, session_ei_summary_file_tag());
        if useLog10D2
            plotBase = [plotBase, '_log10'];
        end
        exportgraphics(figEiSummary, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
        exportgraphics(figEiSummary, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
        fprintf('\nSaved E/I summary figure: %s\n', fullfile(saveDir, plotBase));
    end
end

if splitByEngagement
    engOut = run_session_d2_engagement(sessionType, sessionName, subjectNameForLoad, opts, ...
        analysisConfig, brainArea, brainAreaCombinations, d2Window, useLog10D2, ...
        engagementBufferBefore, engagementBufferAfter, minNonEngagedWindow, ...
        minTimeNonEngaged, absorbSingleEvents, splitExcitatoryInhibitory, plotConfig);
    if saveFigure
        save_session_engagement_d2_figures(engOut, paths, sessionName, brainArea, d2Window, ...
            collectStart, collectEnd, useLog10D2);
    end
end

fprintf('\n=== Done ===\n');


%% Local functions

function engOut = run_session_d2_engagement(sessionType, sessionName, subjectName, opts, ...
    analysisConfig, brainArea, brainAreaCombinations, d2Window, useLog10D2, ...
    engagementBufferBefore, engagementBufferAfter, minNonEngagedWindow, ...
    minTimeNonEngaged, absorbSingleEvents, splitExcitatoryInhibitory, plotConfig)
% RUN_SESSION_D2_ENGAGEMENT - Engaged / non-engaged d2 via engagement modules

if ~is_manuscript_engagement_session_type(sessionType)
    error('session_d2_distributions:BadEngagementType', ...
        'splitByEngagement requires sessionType interval, reach, or semicircle (got %s).', ...
        sessionType);
end
if splitExcitatoryInhibitory
    warning('session_d2_distributions:EngagementIgnoresEI', ...
        'splitByEngagement ignores splitExcitatoryInhibitory (combined population only).');
end

fprintf('\n--- Engagement d2 pipeline ---\n');
fprintf('engagementBuffer: before=%.3g s, after=%.3g s; minNonEngagedWindow=%.1f s\n', ...
    engagementBufferBefore, engagementBufferAfter, minNonEngagedWindow);
fprintf('minTimeNonEngaged: %.1f s (blank non-engaged below this; 0 = off)\n', ...
    minTimeNonEngaged);

if strcmpi(sessionType, 'reach')
    engOpts = reach_criticality_metrics_engagement();
elseif strcmpi(sessionType, 'semicircle')
    engOpts = semicircle_criticality_metrics_engagement();
else
    engOpts = interval_criticality_metrics_engagement();
end

engOpts.collectStart = opts.collectStart;
engOpts.collectEnd = opts.collectEnd;
engOpts.minFiringRate = opts.minFiringRate;
engOpts.maxFiringRate = opts.maxFiringRate;
engOpts.firingRateCheckTime = opts.firingRateCheckTime;
engOpts.dataSource = 'spikes';
engOpts.brainArea = brainArea;
engOpts.brainAreaCombinations = brainAreaCombinations;
engOpts.analyses = {'d2'};
engOpts.makePlots = true;
engOpts.saveFigure = false;
engOpts.plotConfig = plotConfig;
engOpts.d2Window = d2Window;
engOpts.useLog10D2 = useLog10D2;
if analysisConfig.enablePermutations
    engOpts.nShufflesD2 = max(1, analysisConfig.nShuffles);
else
    engOpts.nShufflesD2 = 1;
end
engOpts.useSubsampling = analysisConfig.useSubsampling;
engOpts.nSubsamples = analysisConfig.nSubsamples;
engOpts.nNeuronsSubsample = analysisConfig.nNeuronsSubsample;
engOpts.minNeuronsMultiple = analysisConfig.minNeuronsMultiple;
engOpts.nMinNeurons = analysisConfig.nMinNeurons;
if isfield(analysisConfig, 'binSize') && ~isempty(analysisConfig.binSize)
    engOpts.binSizeD2 = analysisConfig.binSize;
end
engOpts.minNonEngagedWindow = minNonEngagedWindow;
engOpts.minTimeNonEngaged = minTimeNonEngaged;

bufOpts = struct( ...
    'engagementBufferBefore', engagementBufferBefore, ...
    'engagementBufferAfter', engagementBufferAfter);
[bufBefore, bufAfter] = resolve_engagement_buffer_pair( ...
    bufOpts, 'engagementBufferBefore', 'engagementBufferAfter', 'engagementBuffer', 1);
if strcmpi(sessionType, 'reach')
    engOpts.reachBufferBefore = bufBefore;
    engOpts.reachBufferAfter = bufAfter;
    engOpts.absorbSingleReaches = absorbSingleEvents;
    engOpts.runD2AccuracyCorrelation = false;
    engOpts.runD2ReachRateCorrelation = false;
else
    engOpts.eventBufferBefore = bufBefore;
    engOpts.eventBufferAfter = bufAfter;
    engOpts.absorbSingleEvents = absorbSingleEvents;
    engOpts.runD2TrialRateCorrelation = false;
end

if strcmpi(sessionType, 'reach')
    engOut = reach_criticality_metrics_engagement(sessionName, engOpts);
elseif strcmpi(sessionType, 'semicircle')
    engOut = semicircle_criticality_metrics_engagement(subjectName, sessionName, engOpts);
else
    engOut = interval_criticality_metrics_engagement(subjectName, sessionName, engOpts);
end

if isempty(engOut) || ~isfield(engOut, 'd2') || isempty(engOut.d2)
    error('session_d2_distributions:NoEngagementD2', ...
        'Engagement d2 pipeline returned no d2 outputs.');
end
engOut = blank_session_d2_non_engaged_if_short(engOut, minTimeNonEngaged, sessionName);
end

function engOut = blank_session_d2_non_engaged_if_short(engOut, minTimeNonEngaged, sessionName)
% BLANK_SESSION_D2_NON_ENGAGED_IF_SHORT - Omit non-engaged d2 when time is too short
%
% Variables:
%   engOut             - Engagement pipeline output (d2 split + figHandles)
%   minTimeNonEngaged  - Min total non-engaged time (s); 0 = no filter
%   sessionName        - Session id for the log line
%
% Goal:
%   Match criticality_multiple_metrics_across_tasks: if non-engaged duration
%   is below minTimeNonEngaged, blank non-engaged d2 (keep engaged / total)
%   and remove Non-engaged traces from already-drawn engagement figures.

if nargin < 2 || isempty(minTimeNonEngaged) || ~(isfinite(minTimeNonEngaged) && minTimeNonEngaged > 0)
    return;
end
if nargin < 3 || isempty(sessionName)
    sessionName = '';
end

tSec = session_d2_non_engaged_duration_sec(engOut);
if ~(isfinite(tSec) && tSec < minTimeNonEngaged)
    return;
end

fprintf(['  Blanking non-engaged %s: non-engaged time %.1f s ', ...
    '< minTimeNonEngaged %.1f s\n'], sessionName, tSec, minTimeNonEngaged);

iNon = session_d2_non_engaged_class_index(engOut);
if iNon > 0
    if isfield(engOut, 'd2') && isfield(engOut.d2, 'd2') && numel(engOut.d2.d2) >= iNon
        for a = 1:numel(engOut.d2.d2{iNon})
            engOut.d2.d2{iNon}{a} = [];
        end
    end
    if isfield(engOut, 'd2') && isfield(engOut.d2, 'd2Normalized') ...
            && numel(engOut.d2.d2Normalized) >= iNon
        for a = 1:numel(engOut.d2.d2Normalized{iNon})
            engOut.d2.d2Normalized{iNon}{a} = [];
        end
    end
    if isfield(engOut, 'summary') && isstruct(engOut.summary) ...
            && isfield(engOut.summary, 'metrics')
        for m = 1:numel(engOut.summary.metrics)
            if isfield(engOut.summary.metrics{m}, 'stats') ...
                    && numel(engOut.summary.metrics{m}.stats) >= iNon
                engOut.summary.metrics{m}.stats(iNon).mean = nan;
                engOut.summary.metrics{m}.stats(iNon).sem = nan;
                engOut.summary.metrics{m}.stats(iNon).n = 0;
            end
        end
    end
end

if isfield(engOut, 'figHandles') && isstruct(engOut.figHandles)
    metricFigs = {'d2', 'summary'};
    for iFig = 1:numel(metricFigs)
        if isfield(engOut.figHandles, metricFigs{iFig})
            remove_non_engaged_plot_objects(engOut.figHandles.(metricFigs{iFig}));
        end
    end
end
end

function tSec = session_d2_non_engaged_duration_sec(engOut)
% SESSION_D2_NON_ENGAGED_DURATION_SEC - Total non-engaged time from d2 split

tSec = nan;
if isfield(engOut, 'd2') && isstruct(engOut.d2) && isfield(engOut.d2, 'durations') ...
        && isfield(engOut.d2.durations, 'nonEngagedSec')
    tSec = engOut.d2.durations.nonEngagedSec;
    return;
end
if isfield(engOut, 'durations') && isstruct(engOut.durations) ...
        && isfield(engOut.durations, 'd2') && isfield(engOut.durations.d2, 'nonEngagedSec')
    tSec = engOut.durations.d2.nonEngagedSec;
end
end

function iNon = session_d2_non_engaged_class_index(engOut)
% SESSION_D2_NON_ENGAGED_CLASS_INDEX - Index of Non-engaged in classNames (0 if none)

iNon = 0;
classNames = {};
if isfield(engOut, 'd2') && isstruct(engOut.d2) && isfield(engOut.d2, 'classNames')
    classNames = engOut.d2.classNames;
elseif isfield(engOut, 'summary') && isstruct(engOut.summary) ...
        && isfield(engOut.summary, 'classNames')
    classNames = engOut.summary.classNames;
end
if isempty(classNames)
    iNon = 3;
    return;
end
for i = 1:numel(classNames)
    name = lower(strrep(char(classNames{i}), ' ', ''));
    if strcmp(name, 'nonengaged') || strcmp(name, 'non-engaged')
        iNon = i;
        return;
    end
end
end

function remove_non_engaged_plot_objects(fig)
% REMOVE_NON_ENGAGED_PLOT_OBJECTS - Delete Non-engaged histogram/bar objects

if isempty(fig) || ~isgraphics(fig)
    return;
end
objs = findall(fig, '-property', 'DisplayName');
didDelete = false;
for i = 1:numel(objs)
    if ~isgraphics(objs(i))
        continue;
    end
    name = objs(i).DisplayName;
    if isstring(name)
        name = char(name);
    end
    if ischar(name) && strncmpi(strtrim(name), 'Non-engaged', 11)
        delete(objs(i));
        didDelete = true;
    end
end
if ~didDelete
    return;
end
legs = findall(fig, 'Type', 'legend');
for i = 1:numel(legs)
    ax = ancestor(legs(i), 'axes');
    if ~isempty(ax) && isgraphics(ax)
        legend(ax, 'show');
    end
end
end

function save_session_engagement_d2_figures(engOut, paths, sessionName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2)
% SAVE_SESSION_ENGAGEMENT_D2_FIGURES - Export engagement d2 (+ segments) figures

saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

collectEndUsed = collectEnd;
if isfield(engOut, 'config') && isfield(engOut.config, 'collectEnd') ...
        && ~isempty(engOut.config.collectEnd)
    collectEndUsed = engOut.config.collectEnd;
end
collectStartUsed = collectStart;
if isfield(engOut, 'config') && isfield(engOut.config, 'collectStart') ...
        && ~isempty(engOut.config.collectStart)
    collectStartUsed = engOut.config.collectStart;
end

areaTag = format_areas_label(brainArea);
if isempty(areaTag) && isfield(engOut, 'd2') && isfield(engOut.d2, 'areas')
    areaTag = format_areas_label(engOut.d2.areas);
end
if isempty(areaTag)
    areaTag = 'areas';
end
if isempty(collectEndUsed)
    collectTag = sprintf('%.0f-full', collectStartUsed);
else
    collectTag = sprintf('%.0f-%.0f', collectStartUsed, collectEndUsed);
end

figNames = {'d2', 'segments'};
fileTags = {'d2', 'segments'};
for iFig = 1:numel(figNames)
    figField = figNames{iFig};
    if ~isfield(engOut, 'figHandles') || ~isfield(engOut.figHandles, figField) ...
            || ~isgraphics(engOut.figHandles.(figField))
        continue;
    end
    plotBase = sprintf('session_d2_engagement_%s_%s_%s_win%.0fs_%ss', ...
        fileTags{iFig}, sessionName, areaTag, d2Window, collectTag);
    if useLog10D2 && strcmp(figField, 'd2')
        plotBase = [plotBase, '_log10']; %#ok<AGROW>
    end
    exportgraphics(engOut.figHandles.(figField), fullfile(saveDir, [plotBase, '.png']), ...
        'Resolution', 300);
    exportgraphics(engOut.figHandles.(figField), fullfile(saveDir, [plotBase, '.eps']), ...
        'ContentType', 'vector');
    fprintf('\nSaved engagement figure: %s\n', fullfile(saveDir, plotBase));
end
end

function metricValues = extract_d2_summary_metric_values(results, useLog10D2)
% EXTRACT_D2_SUMMARY_METRIC_VALUES - Window-wise d2 values for E/I summary plot

metricValues = struct('d2', []);
if isempty(results.areas) || isempty(results.d2)
    return;
end

d2Vec = results.d2{1}(:);
if useLog10D2
    d2Vec = log10_safe_numeric(d2Vec);
end
metricValues.d2 = d2Vec(isfinite(d2Vec));
end

function yLabelText = get_d2_axis_label(useLog10D2)
if useLog10D2
    yLabelText = 'log_{10}(d2)';
else
    yLabelText = 'd2';
end
end

function results = filter_ar_results_to_brain_area(results, brainArea)
% FILTER_AR_RESULTS_TO_BRAIN_AREA - Keep one area in AR results struct

if isempty(brainArea) || ~isfield(results, 'areas')
    return;
end

areaIdx = find(strcmp(results.areas, brainArea), 1);
if isempty(areaIdx)
    results.areas = {};
    return;
end

cellFields = {'d2', 'd2Normalized', 'startS', 'd2Permuted', 'mrBrPermuted', ...
    'd2PermutedMean', 'd2PermutedSEM', 'popActivityWindows', 'popActivityFull'};

results.areas = results.areas(areaIdx);
for f = 1:length(cellFields)
    fieldName = cellFields{f};
    if isfield(results, fieldName) && numel(results.(fieldName)) >= areaIdx
        results.(fieldName) = results.(fieldName)(areaIdx);
    end
end

if isfield(results, 'binSize') && numel(results.binSize) >= areaIdx
    results.binSize = results.binSize(areaIdx);
end
if isfield(results, 'slidingWindowSize') && numel(results.slidingWindowSize) >= areaIdx
    results.slidingWindowSize = results.slidingWindowSize(areaIdx);
end
end

function print_session_d2_summary(results, useLog10D2)
% PRINT_SESSION_D2_SUMMARY - Window counts and mean d2 per area

fprintf('\n=== Session d2 summary ===\n');
for a = 1:numel(results.areas)
    if a > numel(results.d2) || isempty(results.d2{a})
        fprintf('  %s: no d2 data\n', results.areas{a});
        continue;
    end

    d2Vec = results.d2{a}(:);
    if useLog10D2
        d2Vec = log10_safe_numeric(d2Vec);
    end
    d2Vec = d2Vec(isfinite(d2Vec));

    nPermRows = 0;
    if isfield(results, 'd2Permuted') && a <= numel(results.d2Permuted) && ~isempty(results.d2Permuted{a})
        nPermRows = size(results.d2Permuted{a}, 1);
    end

    if isempty(d2Vec)
        fprintf('  %s: no finite d2 values (perm rows: %d)\n', results.areas{a}, nPermRows);
    else
        fprintf('  %s: %d finite d2 windows, mean = %.4f (perm rows: %d)\n', ...
            results.areas{a}, numel(d2Vec), mean(d2Vec), nPermRows);
    end
end
end

function plotData = build_d2_distribution_data(results, useLog10D2)
% BUILD_D2_DISTRIBUTION_DATA - Collect real d2 and per-window shuffled means
%
% Variables:
%   results    - Output from criticality_ar_analysis
%   useLog10D2 - If true, transform values with log10_safe_numeric
%
% Goal:
%   Build per-area vectors for overlapping histogram/PDF plots:
%   - realD2 values across windows
%   - shuffledMeanD2 values where each element is mean across permutations for one window

plotData = struct();
plotData.areas = {};
plotData.realD2 = {};
plotData.shuffledMeanD2 = {};

for a = 1:numel(results.areas)
    if a > numel(results.d2) || isempty(results.d2{a})
        continue;
    end

    d2Vec = results.d2{a}(:);
    if useLog10D2
        d2Vec = log10_safe_numeric(d2Vec);
    end
    d2Vec = d2Vec(isfinite(d2Vec));
    if isempty(d2Vec)
        continue;
    end

    shuffledVec = get_per_window_shuffle_mean_d2(results, a, useLog10D2);
    shuffledVec = shuffledVec(isfinite(shuffledVec));

    plotData.areas{end+1} = results.areas{a}; %#ok<AGROW>
    plotData.realD2{end+1} = d2Vec; %#ok<AGROW>
    plotData.shuffledMeanD2{end+1} = shuffledVec; %#ok<AGROW>
end
end

function fig = plot_d2_distributions(plotData, sessionType, sessionName, d2Window, collectStart, collectEnd, useLog10D2, plotConfig)
% PLOT_D2_DISTRIBUTIONS - Overlapping PDFs of real d2 and shuffled mean d2
%
% Variables:
%   plotData - Struct from build_d2_distribution_data
%   plotConfig - Manuscript axis/scatter styling
%
% Goal:
%   Plot one tile per area, with shared x-limits and identical bin edges.

if nargin < 8 || isempty(plotConfig)
    plotConfig = fill_manuscript_plot_config();
end

numAreas = numel(plotData.areas);
allVals = [];
for a = 1:numAreas
    allVals = [allVals; plotData.realD2{a}(:)]; %#ok<AGROW>
    allVals = [allVals; plotData.shuffledMeanD2{a}(:)]; %#ok<AGROW>
end
allVals = allVals(isfinite(allVals));
if isempty(allVals)
    error('No finite d2 values available for plotting.');
end

[binEdges, xMin, xMax] = build_shared_histogram_bin_edges(allVals, 28);
if useLog10D2
    xLabelText = 'log_{10}(d2)';
    labelInterpreter = 'tex';
else
    xLabelText = 'd2';
    labelInterpreter = 'none';
end

fig = figure('Color', 'w', 'Position', [120 120 900 280 * numAreas], ...
    'Name', 'd2 distributions');
tileLayout = tiledlayout(numAreas, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

for a = 1:numAreas
    ax = nexttile(tileLayout);
    plot_real_shuffled_histogram_pdfs(ax, plotData.realD2{a}, plotData.shuffledMeanD2{a}, ...
        binEdges, xMin, xMax, plotConfig, useLog10D2);
    apply_manuscript_axes_style(ax, plotConfig, xLabelText, 'Probability density', ...
        plotData.areas{a}, labelInterpreter);
end

sgtitle(tileLayout, sprintf( ...
    'Distribution of %s | real vs shuffled mean per window | %s | %.0fs windows%s [%.0f-%.0f s]', ...
    xLabelText, sessionType, d2Window, make_title_suffix(sessionName), collectStart, collectEnd), ...
    'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

function y = log10_safe_numeric(x)
% LOG10_SAFE_NUMERIC - log10 with NaN for non-positive values

validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end

function suffixStr = make_title_suffix(sessionName)
% MAKE_TITLE_SUFFIX - Optional session-name suffix for figure titles

if isempty(sessionName)
    suffixStr = '';
else
    suffixStr = [' | ' sessionName];
end
end

function label = format_areas_label(areaNames)
% FORMAT_AREAS_LABEL - Underscore-safe tag for filenames/titles

if iscell(areaNames)
    areaNames = areaNames(:)';
    label = strjoin(areaNames, '_');
else
    label = char(areaNames);
end
label = matlab.lang.makeValidName(label);
end

function fig = plot_d2_vs_popactivity(results, useLog10D2, d2Window, plotConfig)
% PLOT_D2_VS_POPACTIVITY - Scatter d2 and shuffled mean d2 vs pop activity per window

if nargin < 3 || isempty(d2Window)
    d2Window = results.params.slidingWindowSize;
end
if nargin < 4 || isempty(plotConfig)
    plotConfig = fill_manuscript_plot_config();
end

if ~isfield(results, 'popActivityWindows')
    error('results.popActivityWindows not found.');
end

numAreas = numel(results.areas);
fig = figure('Color', 'w', 'Position', [140 140 420 * numAreas 420], ...
    'Name', 'd2 vs population activity');
tileLayout = tiledlayout(fig, 1, numAreas, 'TileSpacing', 'compact', 'Padding', 'compact');
d2YLabel = get_d2_axis_label(useLog10D2);
if useLog10D2
    labelInterpreter = 'tex';
else
    labelInterpreter = 'none';
end

allYVals = [];
axesList = gobjects(numAreas, 1);
for a = 1:numAreas
    ax = nexttile(tileLayout);
    axesList(a) = ax;
    [yVals, ~, ~, ~] = plot_d2_popactivity_panel(ax, results, a, useLog10D2, plotConfig, ...
        results.areas{a}, d2YLabel, labelInterpreter, true);
    allYVals = [allYVals; yVals(:)]; %#ok<AGROW>
end
apply_shared_popactivity_ylim(axesList, allYVals);

sgtitle(tileLayout, sprintf('d2 vs mean population activity per %.0fs window', d2Window), ...
    'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

function fig = plot_d2_vs_popactivity_ei_split(eiResultsCell, useLog10D2, d2Window, ...
    plotConfig, sessionName, widthCutoff)
% PLOT_D2_VS_POPACTIVITY_EI_SPLIT - Combined, excitatory, and inhibitory on one figure
%
% Variables:
%   eiResultsCell - Cell of struct with .cellType and .results from each E/I run
%
% Goal:
%   One row per brain area, one column per population (combined, E, I) with shared
%   y-limits across all panels for direct comparison.

if nargin < 4 || isempty(plotConfig)
    plotConfig = fill_manuscript_plot_config();
end
if isempty(eiResultsCell)
    error('No E/I pop-activity results to plot.');
end

refResults = eiResultsCell{1}.results;
if ~isfield(refResults, 'popActivityWindows')
    error('results.popActivityWindows not found.');
end

numAreas = numel(refResults.areas);
numCols = numel(eiResultsCell);
fig = figure('Color', 'w', ...
    'Position', [120 120 380 * numCols max(360, 340 * numAreas)], ...
    'Name', 'd2 vs population activity (E/I split)');
tileLayout = tiledlayout(fig, numAreas, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');
d2YLabel = get_d2_axis_label(useLog10D2);
if useLog10D2
    labelInterpreter = 'tex';
else
    labelInterpreter = 'none';
end

allYVals = [];
axesList = gobjects(numAreas, numCols);
for col = 1:numCols
    entry = eiResultsCell{col};
    results = entry.results;
    panelTitle = cell_type_label(entry.cellType);
    for a = 1:numAreas
        ax = nexttile(tileLayout);
        axesList(a, col) = ax;
        areaTitle = panelTitle;
        if numAreas > 1
            areaTitle = sprintf('%s | %s', panelTitle, results.areas{a});
        end
        showYLabel = (col == 1);
        [yVals, ~, ~, ~] = plot_d2_popactivity_panel(ax, results, a, useLog10D2, plotConfig, ...
            areaTitle, d2YLabel, labelInterpreter, showYLabel);
        allYVals = [allYVals; yVals(:)]; %#ok<AGROW>
    end
end
apply_shared_popactivity_ylim(axesList(:), allYVals);

sgtitle(tileLayout, sprintf('%s | d2 vs mean population activity | %.0fs windows | width cutoff %.3f ms', ...
    sessionName, d2Window, widthCutoff), ...
    'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

function [yVals, rData, rShuf, nValid] = plot_d2_popactivity_panel(ax, results, areaIdx, ...
    useLog10D2, plotConfig, panelTitle, d2YLabel, labelInterpreter, showYLabel)
% PLOT_D2_POPACTIVITY_PANEL - One scatter panel of d2 vs mean pop activity

if nargin < 10 || isempty(showYLabel)
    showYLabel = true;
end
if nargin < 9 || isempty(labelInterpreter)
    labelInterpreter = 'none';
end
if nargin < 8 || isempty(d2YLabel)
    d2YLabel = get_d2_axis_label(useLog10D2);
end

plotColors = manuscript_plot_colors();
hold(ax, 'on');

[d2Vec, popVec, validMask] = get_aligned_d2_popactivity(results, areaIdx, useLog10D2);
yVals = collect_d2_popactivity_y_values(results, areaIdx, useLog10D2);
rData = nan;
rShuf = nan;
nValid = 0;

if ~any(validMask)
    yLabelText = d2YLabel;
    if ~showYLabel
        yLabelText = '';
    end
    apply_manuscript_axes_style(ax, plotConfig, 'Mean pop activity (spikes/bin)', yLabelText, ...
        sprintf('%s (no data)', panelTitle), labelInterpreter);
    hold(ax, 'off');
    return;
end

scatter_manuscript_open(ax, popVec(validMask), d2Vec(validMask), plotConfig, ...
    plotColors.data, 'Data');
add_manuscript_scatter_trendline(ax, popVec(validMask), d2Vec(validMask), plotConfig);

shufVec = get_shuffled_mean_d2_per_window(results, areaIdx, useLog10D2);
if ~isempty(shufVec)
    shufVec = shufVec(1:numel(d2Vec));
    shufMask = validMask & isfinite(shufVec);
    if any(shufMask)
        scatter_manuscript_open(ax, popVec(shufMask), shufVec(shufMask), plotConfig, ...
            plotColors.shuffled, 'Shuffled mean');
    end
end

rData = pearson_r(popVec(validMask), d2Vec(validMask));
if ~isempty(shufVec)
    shufMask = validMask & isfinite(shufVec);
    if any(shufMask)
        rShuf = pearson_r(popVec(shufMask), shufVec(shufMask));
    end
end
nValid = sum(validMask);

yLabelText = d2YLabel;
if ~showYLabel
    yLabelText = '';
end
apply_manuscript_axes_style(ax, plotConfig, 'Mean pop activity (spikes/bin)', yLabelText, ...
    sprintf('%s | r_{data}=%.3f, r_{shuf}=%.3f, n=%d', panelTitle, rData, rShuf, nValid), ...
    labelInterpreter);
legend(ax, 'Location', 'best', 'FontSize', plotConfig.legendFontSize);
grid(ax, 'on');
hold(ax, 'off');
end

function yVals = collect_d2_popactivity_y_values(results, areaIdx, useLog10D2)
% COLLECT_D2_POPACTIVITY_Y_VALUES - Finite d2 y-values (data + shuffled) for y-limits

[d2Vec, ~, validMask] = get_aligned_d2_popactivity(results, areaIdx, useLog10D2);
yVals = d2Vec(validMask);
shufVec = get_shuffled_mean_d2_per_window(results, areaIdx, useLog10D2);
if ~isempty(shufVec) && ~isempty(d2Vec)
    shufVec = shufVec(1:numel(d2Vec));
    shufMask = validMask & isfinite(shufVec);
    yVals = [yVals(:); shufVec(shufMask)]; %#ok<AGROW>
end
yVals = yVals(isfinite(yVals));
end

function apply_shared_popactivity_ylim(axesList, allYVals)
% APPLY_SHARED_POPACTIVITY_YLIM - Match y-limits across pop-activity scatter panels

axesList = axesList(isgraphics(axesList));
if isempty(axesList)
    return;
end
allYVals = allYVals(isfinite(allYVals));
if isempty(allYVals)
    return;
end

yMin = min(allYVals);
yMax = max(allYVals);
ySpan = yMax - yMin;
if ySpan <= 0 || ~isfinite(ySpan)
    pad = max(0.1, abs(yMin) * 0.05 + eps);
else
    pad = 0.05 * ySpan;
end
sharedYLim = [yMin - pad, yMax + pad];
for iAx = 1:numel(axesList)
    ylim(axesList(iAx), sharedYLim);
end
end

function print_d2_popactivity_correlations(results, useLog10D2, populationLabel)
% PRINT_D2_POPACTIVITY_CORRELATIONS - Command-window summary

if nargin < 3
    populationLabel = '';
end
if isempty(populationLabel)
    fprintf('\n=== d2 vs mean pop activity correlations ===\n');
else
    fprintf('\n=== d2 vs mean pop activity correlations (%s) ===\n', populationLabel);
end
for a = 1:numel(results.areas)
    [d2Vec, popVec, validMask] = get_aligned_d2_popactivity(results, a, useLog10D2);
    shufVec = get_shuffled_mean_d2_per_window(results, a, useLog10D2);
    if ~any(validMask)
        fprintf('  %s: no data\n', results.areas{a});
        continue;
    end
    rData = pearson_r(popVec(validMask), d2Vec(validMask));
    rShuf = nan;
    if ~isempty(shufVec)
        shufMask = validMask & isfinite(shufVec);
        if any(shufMask)
            rShuf = pearson_r(popVec(shufMask), shufVec(shufMask));
        end
    end
    fprintf('  %s: r(data)=%.3f, r(shuffled)=%.3f, n=%d\n', ...
        results.areas{a}, rData, rShuf, sum(validMask));
end
end

function d2Vec = get_aligned_d2_vector(results, areaIdx, useLog10D2)
% GET_ALIGNED_D2_VECTOR - d2 per window for one area (optional log10)

d2Vec = [];
if areaIdx > numel(results.d2) || isempty(results.d2{areaIdx})
    return;
end
d2Vec = results.d2{areaIdx}(:);
if useLog10D2
    d2Vec = log10_safe_numeric(d2Vec);
end
end

function [d2Vec, popVec, validMask] = get_aligned_d2_popactivity(results, areaIdx, useLog10D2)
% GET_ALIGNED_D2_POPACTIVITY - Window-aligned d2 and pop activity vectors

d2Vec = [];
popVec = [];
validMask = false(0, 1);

d2Vec = get_aligned_d2_vector(results, areaIdx, useLog10D2);
if isempty(d2Vec)
    return;
end
if ~isfield(results, 'popActivityWindows') || areaIdx > numel(results.popActivityWindows) ...
        || isempty(results.popActivityWindows{areaIdx})
    return;
end

popVec = results.popActivityWindows{areaIdx}(:);
nWindows = min(numel(d2Vec), numel(popVec));
d2Vec = d2Vec(1:nWindows);
popVec = popVec(1:nWindows);
validMask = isfinite(d2Vec) & isfinite(popVec);
end

function shufVec = get_shuffled_mean_d2_per_window(results, areaIdx, useLog10D2)
% GET_SHUFFLED_MEAN_D2_PER_WINDOW - Mean shuffled d2 per window (subsampling-aware)

shufVec = get_per_window_shuffle_mean_d2(results, areaIdx, useLog10D2);
end

function rVal = pearson_r(x, y)
% PEARSON_R - Pearson correlation or NaN when undefined

rVal = nan;
if numel(x) < 2 || numel(y) < 2
    return;
end
cMat = corrcoef(x(:), y(:));
rVal = cMat(1, 2);
end

function refAreaIdx = find_first_area_with_start_times(results)
% FIND_FIRST_AREA_WITH_START_TIMES - Index of first area with startS

refAreaIdx = find(~cellfun(@isempty, results.startS), 1);
if isempty(refAreaIdx)
    error('No window center times (startS) found in results.');
end
end

function fig = plot_d2_pop_ethogram_timeline(dataStructBhv, results, ...
    collectStart, collectEnd, d2Window, binSize, useLog10D2, plotConfig, ...
    sessionName, cellType, useRelativeTime, d2WindowAlign, timelineOpts)
% PLOT_D2_POP_ETHOGRAM_TIMELINE - Stacked mean-pop | d2 | ethogram vs time
%
% Variables:
%   dataStructBhv - Session used for bhvID / fsBhv and duration
%   results       - criticality_ar_analysis output (d2, startS, popActivityWindows)
%   binSize       - Spike bin width (s) used in d2 analysis (title only)
%   useRelativeTime - If true, shift x-axis so t=0 at collectStart (default false)
%   d2WindowAlign - 'center' (default) or 'leadingEdge' (timestamp = window end)
%   timelineOpts  - Optional: splitByEngagement + interval engagement buffers
%
% Layout (per brain area column):
%   Top:    mean popActivity per d2 window (results.popActivityWindows)
%   Middle: window-wise d2 (and shuffled mean when present)
%   Bottom: behavior ethogram (frame labels, semicircle TaskMatrix, or
%           interval beam-break schematics)
%
% Timebase: results.startS are window centers in absolute session time.
% d2WindowAlign maps those centers to plot times (center or leading edge).
% Semicircle ethogram (TaskMatrix):
%   green/red/yellow vertical lines at trialEnd for rewarded/unrewarded/failed
%   black vertical line at choicePort poke time
%   blue fill: leaveHomeLast -> choicePokeTime
%   yellow fill: choicePokeTime -> trialEnd
% Interval (until bhv labels exist):
%   green/red vertical lines at correct/error beam breaks
%   if splitByEngagement: blue engaged / orange non-engaged fills

if nargin < 8 || isempty(plotConfig)
    plotConfig = fill_manuscript_plot_config();
end
if nargin < 9 || isempty(sessionName)
    sessionName = '';
end
if nargin < 10
    cellType = '';
end
if nargin < 11 || isempty(useRelativeTime)
    useRelativeTime = false;
end
if nargin < 12 || isempty(d2WindowAlign)
    d2WindowAlign = 'center';
end
if nargin < 13 || isempty(timelineOpts)
    timelineOpts = struct();
end
d2WindowAlign = normalize_d2_window_align(d2WindowAlign);
timelineOpts = fill_d2_timeline_opts(timelineOpts);

dataPrepPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data_prep');
if exist(dataPrepPath, 'dir')
    addpath(dataPrepPath);
end

fig = [];
numAreas = numel(results.areas);
if numAreas < 1
    warning('session_d2_distributions:NoTimelineAreas', 'No areas for timeline plot.');
    return;
end

bhvRec = session_d2_behavior_record(dataStructBhv);
semiEth = session_d2_semicircle_ethogram_record(dataStructBhv);
intervalEth = session_d2_interval_ethogram_record(dataStructBhv);
tMaxAbs = session_d2_resolve_timeline_tmax([], results, collectStart, collectEnd, d2Window, ...
    dataStructBhv);
tMinAbs = collectStart;
if isempty(tMinAbs) || ~isfinite(tMinAbs)
    tMinAbs = session_time_origin(dataStructBhv);
end
if useRelativeTime
    tMin = 0;
    tMax = tMaxAbs - tMinAbs;
    timeShift = tMinAbs;
    xLabelText = 'Time from collectStart (s)';
else
    tMin = tMinAbs;
    tMax = tMaxAbs;
    timeShift = 0;
    xLabelText = 'Time (s)';
end

plotColors = manuscript_plot_colors();
d2YLabel = get_d2_axis_label(useLog10D2);
fig = figure('Color', 'w', 'Name', sprintf('d2 timeline — %s', sessionName), ...
    'Position', [100 80 max(720, 420 * numAreas) 780]);

axesToLink = gobjects(0);
for a = 1:numAreas
    areaName = results.areas{a};
    tWin = [];
    if isfield(results, 'startS') && a <= numel(results.startS) && ~isempty(results.startS{a})
        tWinAbs = d2_window_align_times(results.startS{a}(:), d2Window, d2WindowAlign);
        tWin = tWinAbs - timeShift;
    end

    axPop = subplot(3, numAreas, a, 'Parent', fig);
    hold(axPop, 'on');
    popVec = [];
    if isfield(results, 'popActivityWindows') && a <= numel(results.popActivityWindows) ...
            && ~isempty(results.popActivityWindows{a})
        popVec = results.popActivityWindows{a}(:);
    end
    if ~isempty(popVec) && ~isempty(tWin)
        nPlot = min(numel(popVec), numel(tWin));
        plot(axPop, tWin(1:nPlot), popVec(1:nPlot), '-o', ...
            'Color', [0.15 0.15 0.15], 'MarkerFaceColor', [0.15 0.15 0.15], ...
            'MarkerSize', 5, 'LineWidth', plotConfig.axesLineWidth);
    else
        text(axPop, mean([tMin tMax]), 0.5, 'no window popActivity', ...
            'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5]);
    end
    xlim(axPop, [tMin, tMax]);
    ylabel(axPop, 'mean pop', 'FontSize', plotConfig.axisLabelFontSize);
    title(axPop, areaName, 'Interpreter', 'none', 'FontSize', plotConfig.titleFontSize);
    set(axPop, 'XTickLabel', [], 'Box', 'off', 'TickDir', 'out', ...
        'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth);
    hold(axPop, 'off');

    axD2 = subplot(3, numAreas, numAreas + a, 'Parent', fig);
    hold(axD2, 'on');
    d2Vec = get_aligned_d2_vector(results, a, useLog10D2);
    if ~isempty(d2Vec) && ~isempty(tWin)
        nPlot = min(numel(d2Vec), numel(tWin));
        tD2 = tWin(1:nPlot);
        d2Vec = d2Vec(1:nPlot);
        plot(axD2, tD2, d2Vec, '-o', 'Color', plotColors.data, ...
            'MarkerFaceColor', plotColors.data, 'MarkerSize', 5, ...
            'LineWidth', plotConfig.axesLineWidth, 'DisplayName', 'Data');
        shufVec = get_shuffled_mean_d2_per_window(results, a, useLog10D2);
        if ~isempty(shufVec)
            shufVec = shufVec(1:nPlot);
            shufMask = isfinite(shufVec) & isfinite(tD2);
            if any(shufMask)
                plot(axD2, tD2(shufMask), shufVec(shufMask), '-o', 'Color', plotColors.shuffled, ...
                    'MarkerFaceColor', plotColors.shuffled, 'MarkerSize', 4, ...
                    'LineWidth', max(0.8, plotConfig.axesLineWidth - 0.3), ...
                    'DisplayName', 'Shuffled mean');
            end
        end
        legend(axD2, 'Location', 'best', 'FontSize', plotConfig.legendFontSize);
    else
        text(axD2, mean([tMin tMax]), 0.5, 'no d2 values', ...
            'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5]);
    end
    xlim(axD2, [tMin, tMax]);
    ylabel(axD2, d2YLabel, 'FontSize', plotConfig.axisLabelFontSize, ...
        'Interpreter', ternary_tex_if_log10(useLog10D2));
    set(axD2, 'XTickLabel', [], 'Box', 'off', 'TickDir', 'out', ...
        'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth);
    hold(axD2, 'off');

    axEth = subplot(3, numAreas, 2 * numAreas + a, 'Parent', fig);
    if ~isempty(intervalEth)
        session_d2_plot_interval_task_schematic(axEth, intervalEth, tMin, tMax, timeShift, ...
            tMinAbs, tMaxAbs, timelineOpts);
    elseif ~isempty(semiEth)
        session_d2_plot_semicircle_ethogram(axEth, semiEth, tMin, tMax, timeShift);
    else
        session_d2_plot_behavior_ethogram(axEth, bhvRec, tMin, tMax);
    end
    xlabel(axEth, xLabelText, 'FontSize', plotConfig.axisLabelFontSize);
    set(axEth, 'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth);

    axesToLink = [axesToLink; axPop; axD2; axEth]; %#ok<AGROW>
end
linkaxes(axesToLink, 'x');

cellTag = '';
if ~isempty(cellType) && ~strcmpi(cellType, 'combined')
    cellTag = sprintf(' | %s', cell_type_label(cellType));
end
bottomTag = 'ethogram';
if ~isempty(intervalEth)
    bottomTag = 'task events';
end
sgtitle(fig, sprintf('%s%s | mean pop / d2 (%.0fs windows, %s, bin=%.0f ms) / %s', ...
    sessionName, cellTag, d2Window, d2WindowAlign, binSize * 1000, bottomTag), ...
    'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold', 'Interpreter', 'none');
fprintf('Plotted d2 timeline (%d area(s), t=[%.1f, %.1f] s).\n', numAreas, tMin, tMax);
end

function alignMode = normalize_d2_window_align(d2WindowAlign)
% NORMALIZE_D2_WINDOW_ALIGN - Canonical 'center' or 'leadingEdge'
%
% Variables:
%   d2WindowAlign - User string (center / leadingEdge and aliases)
%
% Goal:
%   Validate timeline alignment mode. Default is window center.

if nargin < 1 || isempty(d2WindowAlign)
    alignMode = 'center';
    return;
end
key = lower(strtrim(char(d2WindowAlign)));
key = strrep(key, '-', '');
key = strrep(key, '_', '');
key = strrep(key, ' ', '');
switch key
    case {'center', 'centre', 'mid', 'middle'}
        alignMode = 'center';
    case {'leadingedge', 'leading', 'lead', 'end', 'trailing'}
        alignMode = 'leadingEdge';
    otherwise
        error('session_d2_distributions:BadD2WindowAlign', ...
            'd2WindowAlign must be ''center'' or ''leadingEdge'' (got %s).', d2WindowAlign);
end
end

function timelineOpts = fill_d2_timeline_opts(timelineOpts)
% FILL_D2_TIMELINE_OPTS - Defaults for interval schematic / engagement fills
if nargin < 1 || isempty(timelineOpts)
    timelineOpts = struct();
end
if ~isfield(timelineOpts, 'splitByEngagement') || isempty(timelineOpts.splitByEngagement)
    timelineOpts.splitByEngagement = false;
end
if ~isfield(timelineOpts, 'engagementBufferBefore') || isempty(timelineOpts.engagementBufferBefore)
    timelineOpts.engagementBufferBefore = 1;
end
if ~isfield(timelineOpts, 'engagementBufferAfter') || isempty(timelineOpts.engagementBufferAfter)
    timelineOpts.engagementBufferAfter = 1;
end
if ~isfield(timelineOpts, 'minNonEngagedWindow') || isempty(timelineOpts.minNonEngagedWindow)
    timelineOpts.minNonEngagedWindow = 30;
end
if ~isfield(timelineOpts, 'absorbSingleEvents') || isempty(timelineOpts.absorbSingleEvents)
    timelineOpts.absorbSingleEvents = true;
end
timelineOpts.splitByEngagement = logical(timelineOpts.splitByEngagement);
end

function tAlign = d2_window_align_times(startS, d2Window, d2WindowAlign)
% D2_WINDOW_ALIGN_TIMES - Map stored window-center times to plot times
%
% Variables:
%   startS         - Window center times (s) from results.startS
%   d2Window       - Window length (s)
%   d2WindowAlign  - 'center' or 'leadingEdge'
%
% Goal:
%   center: timestamp is the window midpoint (analysis default).
%   leadingEdge: timestamp is the window end; data are the trailing d2Window.

if nargin < 3 || isempty(d2WindowAlign)
    d2WindowAlign = 'center';
end
d2WindowAlign = normalize_d2_window_align(d2WindowAlign);
tAlign = startS(:);
if isempty(tAlign)
    return;
end
if strcmp(d2WindowAlign, 'leadingEdge')
    tAlign = tAlign + d2Window / 2;
end
end

function interp = ternary_tex_if_log10(useLog10D2)
if useLog10D2
    interp = 'tex';
else
    interp = 'none';
end
end

function tMax = session_d2_resolve_timeline_tmax(popTime, results, collectStart, collectEnd, ...
    d2Window, dataStruct)
% SESSION_D2_RESOLVE_TIMELINE_TMAX - Right edge of shared time axis
%
% collectEnd=[] uses the loaded session end. A finite collectEnd is an upper
% bound and is itself capped at the session end (no empty time after the data).

tMax = nan;
if ~isempty(popTime)
    tMax = max(popTime);
end
if isfield(results, 'startS')
    for a = 1:numel(results.startS)
        if ~isempty(results.startS{a})
            tMax = max(tMax, max(results.startS{a}) + d2Window / 2);
        end
    end
end

sessionEndAbs = session_d2_loaded_session_end(dataStruct);
collectEnd = session_d2_scalar_time(collectEnd, nan, @max);
sessionEndAbs = session_d2_scalar_time(sessionEndAbs, nan, @max);
collectStart = session_d2_scalar_time(collectStart, 0, @min);
tMax = session_d2_scalar_time(tMax, nan, @max);

if ~isempty(collectEnd) && isfinite(collectEnd)
    plotEnd = collectEnd;
    if isfinite(sessionEndAbs)
        plotEnd = min(plotEnd, sessionEndAbs);
    end
else
    plotEnd = sessionEndAbs;
    if ~isfinite(plotEnd)
        durationSec = session_d2_session_duration_sec(dataStruct, collectStart);
        durationSec = session_d2_scalar_time(durationSec, nan, @max);
        if isfinite(durationSec)
            plotEnd = collectStart + durationSec;
        end
    end
end
if isfinite(plotEnd)
    tMax = plotEnd;
end
if ~isfinite(tMax) || tMax <= collectStart
    tMax = collectStart + 1;
end
end

function t = session_d2_scalar_time(t, emptyDefault, reduceFcn)
% SESSION_D2_SCALAR_TIME - Finite scalar time, or emptyDefault / NaN
%
% Variables:
%   t            - Time value(s)
%   emptyDefault - Replacement when t is empty / non-finite (default NaN)
%   reduceFcn    - @min or @max when t has more than one finite value

if nargin < 2
    emptyDefault = nan;
end
if nargin < 3 || isempty(reduceFcn)
    reduceFcn = @max;
end
if isempty(t)
    t = emptyDefault;
    return;
end
t = t(isfinite(t));
if isempty(t)
    t = emptyDefault;
else
    t = reduceFcn(t(:));
end
end

function sessionEnd = session_d2_loaded_session_end(dataStruct)
% SESSION_D2_LOADED_SESSION_END - Absolute end time (s) of loaded spike data

sessionEnd = nan;
if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectEnd') ...
        && ~isempty(dataStruct.spikeData.collectEnd)
    sessionEnd = dataStruct.spikeData.collectEnd;
    return;
end
if isfield(dataStruct, 'spikeTimes') && ~isempty(dataStruct.spikeTimes)
    sessionEnd = max(dataStruct.spikeTimes);
    return;
end
if isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectEnd') ...
        && ~isempty(dataStruct.opts.collectEnd)
    sessionEnd = dataStruct.opts.collectEnd;
end
end

function durationSec = session_d2_session_duration_sec(dataStruct, collectStart)
% SESSION_D2_SESSION_DURATION_SEC - Loaded collect window length (s)

durationSec = nan;
if nargin < 2 || isempty(collectStart)
    collectStart = 0;
end
if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectEnd') ...
        && ~isempty(dataStruct.spikeData.collectEnd)
    startVal = collectStart;
    if isfield(dataStruct.spikeData, 'collectStart') && ~isempty(dataStruct.spikeData.collectStart)
        startVal = dataStruct.spikeData.collectStart;
    end
    durationSec = dataStruct.spikeData.collectEnd - startVal;
    return;
end
if isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectEnd') ...
        && ~isempty(dataStruct.opts.collectEnd)
    startVal = collectStart;
    if isfield(dataStruct.opts, 'collectStart') && ~isempty(dataStruct.opts.collectStart)
        startVal = dataStruct.opts.collectStart;
    end
    durationSec = dataStruct.opts.collectEnd - startVal;
    return;
end
if isfield(dataStruct, 'spikeTimes') && ~isempty(dataStruct.spikeTimes)
    durationSec = max(dataStruct.spikeTimes) - collectStart;
end
end

function bhvRec = session_d2_behavior_record(dataStruct)
% SESSION_D2_BEHAVIOR_RECORD - bhvID + fsBhv for ethogram plotting

bhvRec = struct('bhvID', [], 'fsBhv', nan, 'bhvTimeOrigin', 0);
if isfield(dataStruct, 'bhvID') && ~isempty(dataStruct.bhvID)
    bhvRec.bhvID = dataStruct.bhvID(:);
end
if isfield(dataStruct, 'fsBhv') && ~isempty(dataStruct.fsBhv)
    bhvRec.fsBhv = dataStruct.fsBhv;
elseif isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'fsBhv') ...
        && ~isempty(dataStruct.opts.fsBhv)
    bhvRec.fsBhv = dataStruct.opts.fsBhv;
end
bhvRec.bhvTimeOrigin = session_time_origin(dataStruct);
end

function ethRec = session_d2_semicircle_ethogram_record(dataStruct)
% SESSION_D2_SEMICIRCLE_ETHOGRAM_RECORD - TaskMatrix events for semicircle ethogram
%
% Variables:
%   dataStruct - Loaded semicircle session (trialOutcome, choicePokeTime, ...)
%
% Goal:
%   Return [] for non-semicircle sessions; otherwise pack trial event times used
%   by session_d2_plot_semicircle_ethogram.

ethRec = [];
if ~isfield(dataStruct, 'sessionType') || ~strcmpi(dataStruct.sessionType, 'semicircle')
    return;
end
if ~isfield(dataStruct, 'trialEnd') || isempty(dataStruct.trialEnd)
    return;
end

ethRec = struct();
ethRec.trialOutcome = dataStruct.trialOutcome(:);
ethRec.trialEnd = dataStruct.trialEnd(:);
ethRec.choicePokeTime = dataStruct.choicePokeTime(:);
ethRec.leaveHomeLast = dataStruct.leaveHomeLast(:);
nTrial = numel(ethRec.trialEnd);
if numel(ethRec.trialOutcome) ~= nTrial
    ethRec.trialOutcome = nan(nTrial, 1);
end
if numel(ethRec.choicePokeTime) ~= nTrial
    ethRec.choicePokeTime = nan(nTrial, 1);
end
if numel(ethRec.leaveHomeLast) ~= nTrial
    ethRec.leaveHomeLast = nan(nTrial, 1);
end
end

function ethRec = session_d2_interval_ethogram_record(dataStruct)
% SESSION_D2_INTERVAL_ETHOGRAM_RECORD - Correct/error beam breaks for interval schematic
%
% Variables:
%   dataStruct - Loaded interval session (subjectName, sessionName)
%
% Goal:
%   Return [] for non-interval sessions; otherwise pack beam-break outcome times
%   used by session_d2_plot_interval_task_schematic. Interval sessions do not
%   yet have frame-wise behavior labels.

ethRec = [];
if ~isfield(dataStruct, 'sessionType') || ~strcmpi(dataStruct.sessionType, 'interval')
    return;
end
if ~isfield(dataStruct, 'subjectName') || isempty(dataStruct.subjectName) ...
        || ~isfield(dataStruct, 'sessionName') || isempty(dataStruct.sessionName)
    return;
end

paths = get_paths();
try
    [eventTimes, eventTypes] = load_interval_beam_break_events( ...
        paths, dataStruct.subjectName, dataStruct.sessionName, 0.1);
catch ME
    warning('session_d2_distributions:IntervalEvents', ...
        'Could not load interval task events: %s', ME.message);
    return;
end

ethRec = struct();
ethRec.eventTimes = eventTimes(:);
ethRec.eventTypes = eventTypes(:);
end

function session_d2_plot_semicircle_ethogram(ax, ethRec, tMin, tMax, timeShift)
% SESSION_D2_PLOT_SEMICIRCLE_ETHOGRAM - TaskMatrix event ethogram for semicircle
%
% Variables:
%   ax        - Target axes
%   ethRec    - From session_d2_semicircle_ethogram_record
%   tMin/tMax - Shared x-limits (already relative when useRelativeTime)
%   timeShift - Absolute time subtracted for relative plotting (0 if absolute)
%
% Goal:
%   Draw:
%     blue fill: leaveHomeLast -> choicePokeTime
%     yellow fill: choicePokeTime -> trialEnd
%     black line at choicePokeTime
%     green/red/yellow line at trialEnd for rewarded/unrewarded/failed

if nargin < 5 || isempty(timeShift)
    timeShift = 0;
end

hold(ax, 'on');
if isempty(ethRec) || isempty(ethRec.trialEnd)
    text(ax, mean([tMin tMax]), 0.5, 'no semicircle TaskMatrix events', ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', [0.5 0.5 0.5]);
    xlim(ax, [tMin, tMax]);
    ylim(ax, [0 1]);
    set(ax, 'YTick', [], 'Box', 'off');
    hold(ax, 'off');
    return;
end

blueFill = [0.30, 0.55, 0.90];
yellowFill = [0.95, 0.85, 0.25];
colorReward = [0.10, 0.70, 0.25];
colorUnrewarded = [0.85, 0.15, 0.15];
colorFailed = [0.95, 0.80, 0.10];
colorChoice = [0.05, 0.05, 0.05];
lineWidthEvent = 1.1;

nTrial = numel(ethRec.trialEnd);
hBlue = gobjects(0);
hYellow = gobjects(0);
hChoice = gobjects(0);
hReward = gobjects(0);
hUnrewarded = gobjects(0);
hFailed = gobjects(0);

% Fills first (behind event lines)
for iTrial = 1:nTrial
    leaveT = ethRec.leaveHomeLast(iTrial) - timeShift;
    pokeT = ethRec.choicePokeTime(iTrial) - timeShift;
    endT = ethRec.trialEnd(iTrial) - timeShift;

    if isfinite(leaveT) && isfinite(pokeT) && pokeT > leaveT
        h = fill(ax, [leaveT, pokeT, pokeT, leaveT], [0, 0, 1, 1], blueFill, ...
            'EdgeColor', 'none', 'FaceAlpha', 0.45, 'HandleVisibility', 'off');
        if isempty(hBlue)
            set(h, 'HandleVisibility', 'on', 'DisplayName', 'leave home \rightarrow poke');
            hBlue = h;
        end
    end

    if isfinite(pokeT) && isfinite(endT) && endT > pokeT
        h = fill(ax, [pokeT, endT, endT, pokeT], [0, 0, 1, 1], yellowFill, ...
            'EdgeColor', 'none', 'FaceAlpha', 0.40, 'HandleVisibility', 'off');
        if isempty(hYellow)
            set(h, 'HandleVisibility', 'on', 'DisplayName', 'poke \rightarrow trial end');
            hYellow = h;
        end
    end
end

% Event lines
for iTrial = 1:nTrial
    pokeT = ethRec.choicePokeTime(iTrial) - timeShift;
    endT = ethRec.trialEnd(iTrial) - timeShift;
    outcome = ethRec.trialOutcome(iTrial);

    if isfinite(pokeT)
        h = plot(ax, [pokeT, pokeT], [0, 1], '-', 'Color', colorChoice, ...
            'LineWidth', lineWidthEvent, 'HandleVisibility', 'off');
        if isempty(hChoice)
            set(h, 'HandleVisibility', 'on', 'DisplayName', 'choice poke');
            hChoice = h;
        end
    end

    if ~isfinite(endT)
        continue;
    end
    if outcome == 1
        h = plot(ax, [endT, endT], [0, 1], '-', 'Color', colorReward, ...
            'LineWidth', lineWidthEvent, 'HandleVisibility', 'off');
        if isempty(hReward)
            set(h, 'HandleVisibility', 'on', 'DisplayName', 'rewarded end');
            hReward = h;
        end
    elseif outcome == 0
        h = plot(ax, [endT, endT], [0, 1], '-', 'Color', colorUnrewarded, ...
            'LineWidth', lineWidthEvent, 'HandleVisibility', 'off');
        if isempty(hUnrewarded)
            set(h, 'HandleVisibility', 'on', 'DisplayName', 'unrewarded end');
            hUnrewarded = h;
        end
    elseif outcome == -1
        h = plot(ax, [endT, endT], [0, 1], '-', 'Color', colorFailed, ...
            'LineWidth', lineWidthEvent, 'HandleVisibility', 'off');
        if isempty(hFailed)
            set(h, 'HandleVisibility', 'on', 'DisplayName', 'failed end');
            hFailed = h;
        end
    end
end

xlim(ax, [tMin, tMax]);
ylim(ax, [0 1]);
ylabel(ax, 'task', 'FontSize', 9);
set(ax, 'YTick', [], 'Box', 'off', 'TickDir', 'out');

legendHandles = [hBlue, hYellow, hChoice, hReward, hUnrewarded, hFailed];
legendHandles = legendHandles(isgraphics(legendHandles));
if ~isempty(legendHandles)
    legend(ax, legendHandles, 'Location', 'best', 'FontSize', 7, 'Box', 'off', ...
        'Interpreter', 'tex');
end
hold(ax, 'off');
end

function session_d2_plot_interval_task_schematic(ax, ethRec, tMin, tMax, timeShift, ...
    tMinAbs, tMaxAbs, timelineOpts)
% SESSION_D2_PLOT_INTERVAL_TASK_SCHEMATIC - Beam-break events (+ engagement fills)
%
% Variables:
%   ax            - Target axes
%   ethRec        - From session_d2_interval_ethogram_record
%   tMin/tMax     - Shared x-limits (already relative when useRelativeTime)
%   timeShift     - Absolute time subtracted for relative plotting (0 if absolute)
%   tMinAbs/tMaxAbs - Absolute collect bounds for engagement segments
%   timelineOpts  - splitByEngagement + buffer / min-gap / absorb flags
%
% Goal:
%   Until interval bhv labels exist, draw correct (green) / error (red) beam
%   breaks. If splitByEngagement, shade engaged (blue) and non-engaged (orange)
%   using the same event-buffer rules as the interval engagement module.

if nargin < 5 || isempty(timeShift)
    timeShift = 0;
end
if nargin < 8 || isempty(timelineOpts)
    timelineOpts = fill_d2_timeline_opts(struct());
end

hold(ax, 'on');
if isempty(ethRec) || ~isfield(ethRec, 'eventTimes') || isempty(ethRec.eventTimes)
    text(ax, mean([tMin tMax]), 0.5, 'no interval task events', ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', [0.5 0.5 0.5]);
    xlim(ax, [tMin, tMax]);
    ylim(ax, [0 1]);
    set(ax, 'YTick', [], 'Box', 'off');
    hold(ax, 'off');
    return;
end

yMin = 0;
yMax = 1;
engagedColor = [0.15, 0.45, 0.75];
nonEngagedColor = [0.85, 0.35, 0.15];
correctColor = [0.2, 0.65, 0.25];
errorColor = [0.85, 0.2, 0.2];
lineWidthEvent = 0.9;

legendHandles = gobjects(0);
eventTimes = ethRec.eventTimes(:);
eventTypes = ethRec.eventTypes(:);
if numel(eventTypes) ~= numel(eventTimes)
    eventTypes = strings(size(eventTimes));
end

if timelineOpts.splitByEngagement
    [engagedSegs, nonEngagedSegs] = session_d2_interval_engagement_segments( ...
        tMinAbs, tMaxAbs, eventTimes, timelineOpts.minNonEngagedWindow, ...
        timelineOpts.engagementBufferBefore, timelineOpts.engagementBufferAfter, ...
        timelineOpts.absorbSingleEvents);
    hNon = session_d2_add_engagement_patches(ax, nonEngagedSegs, nonEngagedColor, ...
        yMin, yMax, timeShift);
    if ~isempty(hNon)
        set(hNon, 'HandleVisibility', 'on', ...
            'DisplayName', sprintf('Non-engaged (n=%d)', numel(nonEngagedSegs)));
        legendHandles(end + 1) = hNon; %#ok<AGROW>
    end
    hEng = session_d2_add_engagement_patches(ax, engagedSegs, engagedColor, ...
        yMin, yMax, timeShift);
    if ~isempty(hEng)
        set(hEng, 'HandleVisibility', 'on', ...
            'DisplayName', sprintf('Engaged (n=%d)', numel(engagedSegs)));
        legendHandles(end + 1) = hEng; %#ok<AGROW>
    end
end

correctMask = eventTypes == "correct";
errorMask = eventTypes == "error";
hCorrect = gobjects(0);
hError = gobjects(0);
for iEvent = 1:numel(eventTimes)
    x = eventTimes(iEvent) - timeShift;
    if correctMask(iEvent)
        hLine = plot(ax, [x, x], [yMin, yMax], 'Color', correctColor, ...
            'LineWidth', lineWidthEvent, 'HandleVisibility', 'off');
        if isempty(hCorrect)
            hCorrect = hLine;
            set(hCorrect, 'HandleVisibility', 'on', 'DisplayName', ...
                sprintf('Correct (n=%d)', sum(correctMask)));
        end
    elseif errorMask(iEvent)
        hLine = plot(ax, [x, x], [yMin, yMax], 'Color', errorColor, ...
            'LineWidth', lineWidthEvent, 'HandleVisibility', 'off');
        if isempty(hError)
            hError = hLine;
            set(hError, 'HandleVisibility', 'on', 'DisplayName', ...
                sprintf('Error (n=%d)', sum(errorMask)));
        end
    else
        plot(ax, [x, x], [yMin, yMax], 'Color', [0.35, 0.35, 0.35], ...
            'LineWidth', 0.75, 'HandleVisibility', 'off');
    end
end
if ~isempty(hCorrect)
    legendHandles(end + 1) = hCorrect; %#ok<AGROW>
end
if ~isempty(hError)
    legendHandles(end + 1) = hError; %#ok<AGROW>
end

xlim(ax, [tMin, tMax]);
ylim(ax, [yMin, yMax]);
ylabel(ax, 'task', 'FontSize', 9);
set(ax, 'YTick', [], 'Box', 'off', 'TickDir', 'out');
if ~isempty(legendHandles)
    legend(ax, legendHandles, 'Location', 'best', 'FontSize', 7, 'Box', 'off');
end
hold(ax, 'off');
end

function h = session_d2_add_engagement_patches(ax, segs, colorVal, yMin, yMax, timeShift)
% SESSION_D2_ADD_ENGAGEMENT_PATCHES - Shade engagement intervals; one legend handle

h = gobjects(0);
for i = 1:numel(segs)
    t0 = segs(i).start - timeShift;
    t1 = segs(i).end - timeShift;
    hi = patch(ax, [t0, t1, t1, t0], [yMin, yMin, yMax, yMax], colorVal, ...
        'FaceAlpha', 0.35, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    if isempty(h)
        h = hi;
    end
end
end

function [engagedSegs, nonEngagedSegs] = session_d2_interval_engagement_segments( ...
    collectStart, collectEnd, eventTimes, minNonEngagedWindow, bufferBefore, bufferAfter, ...
    absorbSingleEvents)
% SESSION_D2_INTERVAL_ENGAGEMENT_SEGMENTS - Engaged / non-engaged from beam breaks
%
% Same rules as interval_criticality_metrics_engagement: each event occupies
% [event-bufferBefore, event+bufferAfter]; non-engaged gaps must be at least
% minNonEngagedWindow; isolated single events can be absorbed.

if nargin < 5 || isempty(bufferBefore)
    bufferBefore = 0;
end
if nargin < 6 || isempty(bufferAfter)
    bufferAfter = bufferBefore;
end
if nargin < 7 || isempty(absorbSingleEvents)
    absorbSingleEvents = true;
end
bufferBefore = max(0, bufferBefore);
bufferAfter = max(0, bufferAfter);

eventTimes = sort(eventTimes(:));
eventTimes = eventTimes(eventTimes >= collectStart & eventTimes <= collectEnd);

occupied = session_d2_merge_event_buffers( ...
    eventTimes, bufferBefore, bufferAfter, collectStart, collectEnd);
absorbedMask = false(1, numel(occupied));
if absorbSingleEvents && ~isempty(occupied)
    absorbedMask = session_d2_absorbed_single_event_mask( ...
        eventTimes, collectStart, collectEnd, minNonEngagedWindow, bufferBefore, bufferAfter);
end

nonEngagedSegs = struct('start', {}, 'end', {});
cursor = collectStart;
iOcc = 1;
while iOcc <= numel(occupied)
    if absorbedMask(iOcc)
        gapStart = cursor;
        if iOcc < numel(occupied)
            gapEnd = occupied(iOcc + 1).start;
        else
            gapEnd = collectEnd;
        end
        if (gapEnd - gapStart) >= minNonEngagedWindow
            nonEngagedSegs(end + 1).start = gapStart; %#ok<AGROW>
            nonEngagedSegs(end).end = gapEnd;
        end
        cursor = gapEnd;
    else
        gapStart = cursor;
        gapEnd = occupied(iOcc).start;
        if (gapEnd - gapStart) >= minNonEngagedWindow
            nonEngagedSegs(end + 1).start = gapStart; %#ok<AGROW>
            nonEngagedSegs(end).end = gapEnd;
        end
        cursor = occupied(iOcc).end;
    end
    iOcc = iOcc + 1;
end
if (collectEnd - cursor) >= minNonEngagedWindow
    nonEngagedSegs(end + 1).start = cursor;
    nonEngagedSegs(end).end = collectEnd;
end

engagedSegs = session_d2_complement_segments(collectStart, collectEnd, nonEngagedSegs);
end

function absorbedMask = session_d2_absorbed_single_event_mask( ...
    eventTimes, collectStart, collectEnd, minNonEngagedWindow, bufferBefore, bufferAfter)
% SESSION_D2_ABSORBED_SINGLE_EVENT_MASK - Isolated single events to merge into gaps

occupied = session_d2_merge_event_buffers( ...
    eventTimes, bufferBefore, bufferAfter, collectStart, collectEnd);
absorbedMask = false(1, numel(occupied));
if isempty(occupied)
    return;
end

for iOcc = 1:numel(occupied)
    nEventsInOcc = sum(eventTimes >= occupied(iOcc).start & eventTimes <= occupied(iOcc).end);
    if nEventsInOcc ~= 1
        continue;
    end
    if iOcc == 1
        gapBeforeStart = collectStart;
    else
        gapBeforeStart = occupied(iOcc - 1).end;
    end
    gapBeforeEnd = occupied(iOcc).start;
    gapAfterStart = occupied(iOcc).end;
    if iOcc == numel(occupied)
        gapAfterEnd = collectEnd;
    else
        gapAfterEnd = occupied(iOcc + 1).start;
    end
    if (gapBeforeEnd - gapBeforeStart) >= minNonEngagedWindow && ...
            (gapAfterEnd - gapAfterStart) >= minNonEngagedWindow
        absorbedMask(iOcc) = true;
    end
end
end

function occupied = session_d2_merge_event_buffers(eventTimes, bufferBefore, bufferAfter, ...
    collectStart, collectEnd)
% SESSION_D2_MERGE_EVENT_BUFFERS - Union of [event-before, event+after]

occupied = struct('start', {}, 'end', {});
if isempty(eventTimes)
    return;
end

starts = max(collectStart, eventTimes - bufferBefore);
ends = min(collectEnd, eventTimes + bufferAfter);
valid = ends > starts;
starts = starts(valid);
ends = ends(valid);
if isempty(starts)
    return;
end

[starts, ord] = sort(starts);
ends = ends(ord);

occupied(1).start = starts(1);
occupied(1).end = ends(1);
for i = 2:numel(starts)
    if starts(i) <= occupied(end).end
        occupied(end).end = max(occupied(end).end, ends(i));
    else
        occupied(end + 1).start = starts(i); %#ok<AGROW>
        occupied(end).end = ends(i);
    end
end
end

function engagedSegs = session_d2_complement_segments(collectStart, collectEnd, nonEngagedSegs)
% SESSION_D2_COMPLEMENT_SEGMENTS - Intervals in collect window not in nonEngagedSegs

engagedSegs = struct('start', {}, 'end', {});
if isempty(nonEngagedSegs)
    engagedSegs(1).start = collectStart;
    engagedSegs(1).end = collectEnd;
    return;
end

starts = [nonEngagedSegs.start];
ends = [nonEngagedSegs.end];
[starts, ord] = sort(starts);
ends = ends(ord);

cursor = collectStart;
for i = 1:numel(starts)
    if starts(i) > cursor + eps
        engagedSegs(end + 1).start = cursor; %#ok<AGROW>
        engagedSegs(end).end = starts(i);
    end
    cursor = max(cursor, ends(i));
end
if cursor < collectEnd - eps
    engagedSegs(end + 1).start = cursor;
    engagedSegs(end).end = collectEnd;
end
end

function session_d2_plot_behavior_ethogram(ax, bhvRec, tMin, tMax)
% SESSION_D2_PLOT_BEHAVIOR_ETHOGRAM - Colored behavior runs aligned to time
%
% Variables:
%   ax     - Target axes
%   bhvRec - Struct with .bhvID, .fsBhv, .bhvTimeOrigin
%   tMin, tMax - Shared x-limits (s). bhvID(1) maps to tMin (absolute collect
%                start by default, or 0 when plotting relative time).

hold(ax, 'on');
bhvID = bhvRec.bhvID;
fsBhv = bhvRec.fsBhv;
if isempty(bhvID) || ~(isfinite(fsBhv) && fsBhv > 0)
    text(ax, mean([tMin tMax]), 0.5, 'no behavior labels', ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', [0.5 0.5 0.5]);
    xlim(ax, [tMin, tMax]);
    ylim(ax, [0 1]);
    set(ax, 'YTick', [], 'Box', 'off');
    hold(ax, 'off');
    return;
end

bhvID = bhvID(:);
nFrame = numel(bhvID);
frameStarts = tMin + ((0:nFrame-1)' ) / fsBhv;
frameEnds = tMin + (1:nFrame)' / fsBhv;

uniqueCodes = unique(bhvID);
codeColorMap = containers.Map('KeyType', 'double', 'ValueType', 'any');
for iCode = 1:numel(uniqueCodes)
    code = double(uniqueCodes(iCode));
    c = colors_for_behaviors(code);
    if size(c, 1) == 1 && size(c, 2) == 3
        codeColorMap(code) = c;
    else
        codeColorMap(code) = [0.7 0.7 0.7];
    end
end

runCode = bhvID(1);
runStart = frameStarts(1);
for i = 2:nFrame
    if bhvID(i) ~= runCode
        session_d2_fill_ethogram_run(ax, runStart, frameStarts(i), runCode, codeColorMap);
        runCode = bhvID(i);
        runStart = frameStarts(i);
    end
end
session_d2_fill_ethogram_run(ax, runStart, frameEnds(end), runCode, codeColorMap);

xlim(ax, [tMin, tMax]);
ylim(ax, [0 1]);
ylabel(ax, 'bhv', 'FontSize', 9);
set(ax, 'YTick', [], 'Box', 'off', 'TickDir', 'out');
hold(ax, 'off');
end

function session_d2_fill_ethogram_run(ax, tStart, tEnd, code, codeColorMap)
% SESSION_D2_FILL_ETHOGRAM_RUN - One colored rectangle for a behavior bout

if ~(isfinite(tStart) && isfinite(tEnd)) || tEnd <= tStart
    return;
end
code = double(code);
if isKey(codeColorMap, code)
    faceColor = codeColorMap(code);
else
    faceColor = [0.7 0.7 0.7];
end
fill(ax, [tStart, tEnd, tEnd, tStart], [0, 0, 1, 1], faceColor, ...
    'EdgeColor', 'none', 'HandleVisibility', 'off');
end
