%%
% Session Kurtosis Distributions (Manuscript)
%
% For one session, runs the same PRG pipeline as criticality_prg_across_tasks.m
% (non-overlapping block windows) and plots overlapping probability densities of
% window-wise kurtosis for data vs surrogates (Cambrainha-style; see criticality_prg_plot).
%
% Variables (configure in this section):
%   sessionType      - 'spontaneous', 'interval', 'reach', 'schall'
%   sessionName      - Session identifier
%   subjectName        - Required for spontaneous/interval; '' for reach
%   dataSource         - 'spikes' or 'lfp'
%   collectStart       - Window start (seconds from session onset)
%   collectEnd         - Window end (seconds)
%   prgWindow          - Non-overlapping block length (seconds); blockWindowSize
%   brainArea              - Single or merged area (e.g. 'M56', 'M23M56'); '' uses all valid areas
%   brainAreaCombinations  - Merged areas: struct('name', 'M23M56', 'areas', {{'M23','M56'}})
%   saveFigure         - Export PNG/EPS to dropPath/criticality_manuscript
%   prgMethod          - 'pca' (momentum-space) or 'icg' (real-space ICG)
%   surrogateMethod    - 'isi' (ISI shuffle per unit) or 'circular' (circshift per neuron)
%   useSubsampling     - If true, kappa/D_JS per window = mean across neuron subsamples
%   nSubsamples, nNeuronsSubsample, minNeuronsMultiple - subsampling settings
%   splitExcitatoryInhibitory - If true, run combined (E+I), excitatory, and inhibitory;
%                               plot kurtosis and D_JS distributions for each population;
%                               also plots mean +/- SEM kappa and D_JS summaries across windows
%   widthCutoff        - Peak-to-trough width threshold in ms (narrow <= cutoff = I)
%
% Goal:
%   Visualize real vs surrogate PRG kurtosis distributions for one session,
%   with shared kappa axes and Gaussian reference at kappa = 3.

%% Configuration
% sessionType = 'interval';
% subjectName = 'ey9166';
% sessionName = 'ey9166_2026_04_03';
% dataSource = 'spikes';

collectStart = 0;
collectEnd = 45 * 60;

prgWindow = 30;  % seconds; non-overlapping blocks

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
saveFigure = false;

useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 20;
minNeuronsMultiple = 1.25;

splitExcitatoryInhibitory = true;
widthCutoff = 0.35;  % ms; peak-to-trough width (narrow <= cutoff = inhibitory)

% Surrogate null: 'isi' (Cambrainha paper) or 'circular' (per-neuron circshift on binned data)
surrogateMethod = 'isi';

opts = neuro_behavior_options();
opts.firingRateCheckTime = 5 * 60;
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.05;
opts.maxFiringRate = 100;

analysisConfig = struct();
analysisConfig.prgMethod = 'pca';  % 'pca' or 'icg'
analysisConfig.blockWindowSize = prgWindow;
analysisConfig.binSize = 0.05;
analysisConfig.cvThreshold = 5;
analysisConfig.cutoffDivisors = [1, 2, 4, 8, 16];
analysisConfig.finalCutoffDivisor = 16;
analysisConfig.kappaAxisMax = 20;
analysisConfig.enableSurrogates = true;
analysisConfig.nSurrogates = 1;
analysisConfig.surrogateMethod = surrogateMethod;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.plotTimeSeries = false;  % distribution figure only (criticality_prg_plot)
analysisConfig.nMinNeurons = 20;
analysisConfig.useSubsampling = useSubsampling;
analysisConfig.nSubsamples = nSubsamples;
analysisConfig.nNeuronsSubsample = nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = minNeuronsMultiple;

%% Paths
paths = get_paths();
scriptDir = fileparts(mfilename('fullpath'));
if contains(scriptDir, [filesep 'Editor_' filesep])
  scriptDir = fileparts(which('session_kurtosis_distributions'));
end
srcPath = fullfile(scriptDir, '..');
addpath(srcPath);
addpath(fullfile(srcPath, 'reach_task'));
addpath(fullfile(srcPath, 'schall'));
addpath(fullfile(srcPath, 'spontaneous'));
addpath(fullfile(srcPath, 'interval_timing_task'));
addpath(fullfile(srcPath, 'criticality', 'scripts'));
addpath(fullfile(srcPath, 'criticality', 'analyses'));
addpath(fullfile(srcPath, 'session_prep', 'data_prep'));
addpath(fullfile(srcPath, 'session_prep', 'utils'));
addpath(fullfile(srcPath, 'data_prep'));
addpath(fullfile(srcPath, 'sliding_window_prep', 'utils'));
addpath(fullfile(srcPath, 'criticality'));

fprintf('\n=== Session Kurtosis Distributions ===\n');
fprintf('PRG method: %s\n', analysisConfig.prgMethod);
fprintf('Session [%s]: %s\n', sessionType, sessionName);
fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', collectStart, collectEnd, (collectEnd - collectStart) / 60);
fprintf('PRG blocks: %.1f s; kappa at N/%d; surrogates: %s\n', ...
  prgWindow, analysisConfig.finalCutoffDivisor, analysisConfig.surrogateMethod);
if useSubsampling
  fprintf('Subsampling: %d subsets x %d neurons\n', nSubsamples, nNeuronsSubsample);
end
if splitExcitatoryInhibitory
  fprintf('E/I split: on (widthCutoff = %.3f ms)\n', widthCutoff);
end

%% Load session and run PRG analysis
subjectNameForLoad = '';
if exist('subjectName', 'var') && ~isempty(subjectName)
  subjectNameForLoad = subjectName;
end
loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

[dataStruct, areaOk] = apply_manuscript_brain_area_selection(dataStruct, brainArea, brainAreaCombinations, false);
if ~areaOk
  error('Brain area "%s" not available in this session.', brainArea);
end

cellTypesToRun = get_session_cell_types_to_run(splitExcitatoryInhibitory);
if splitExcitatoryInhibitory
  kappaLabel = sprintf('\\kappa (N/%d)', analysisConfig.finalCutoffDivisor);
  djsLabel = sprintf('D_{JS} (N/%d)', analysisConfig.finalCutoffDivisor);
  eiSummary = init_session_ei_summary({'kappa', 'djs'}, {kappaLabel, djsLabel});
end

for iCellRun = 1:numel(cellTypesToRun)
  cellType = cellTypesToRun{iCellRun};
  dataStructRun = prepare_session_data_for_cell_type(dataStruct, paths, cellType, widthCutoff, splitExcitatoryInhibitory);

  [dataStructRun, ~] = apply_manuscript_brain_area_selection(dataStructRun, brainArea, brainAreaCombinations);

  nUnitsRun = count_session_neurons_for_brain_area(dataStructRun, brainArea);
  if nUnitsRun < analysisConfig.nMinNeurons
    warning('session_kurtosis_distributions:TooFewUnits', ...
      'Skipping %s: %d units in analysis area(s) (min %d).', ...
      cell_type_label(cellType), nUnitsRun, analysisConfig.nMinNeurons);
    continue;
  end

  results = criticality_prg_analysis(dataStructRun, analysisConfig);

  if ~isempty(brainArea)
    results = filter_prg_results_to_brain_area(results, brainArea);
    if isempty(results.areas)
      error('No PRG results for brain area "%s" (%s).', brainArea, cell_type_label(cellType));
    end
  end

  print_session_kappa_summary(results);

  if splitExcitatoryInhibitory
    eiSummary = set_session_ei_summary_population(eiSummary, cellType, ...
      extract_prg_summary_metric_values(results));
  end

  %% Overlapping kurtosis and D_JS distributions (real vs surrogate)
  plotConfig = struct('savePlots', false);
  if saveFigure
    saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    plotConfig.saveDir = saveDir;
  end

  kappaDistFigName = 'PRG kurtosis distributions';
  djsDistFigName = 'PRG Jensen-Shannon distance distributions';
  figHandlesBefore = findall(0, 'Type', 'figure');

  criticality_prg_plot(results, plotConfig, analysisConfig, dataStructRun);

  figKappaDist = find_new_figure_by_name(kappaDistFigName, figHandlesBefore);
  if ~isgraphics(figKappaDist)
    error(['No new kurtosis distribution figure created (%s). Enable surrogates and ', ...
      'ensure valid kappa windows (check CV exclusion).'], cell_type_label(cellType));
  end
  figDjsDist = find_new_figure_by_name(djsDistFigName, figHandlesBefore);

  titleSuffix = '';
  if splitExcitatoryInhibitory
    typeTag = cell_type_file_tag(cellType);
    set(figKappaDist, 'Name', [kappaDistFigName, typeTag]);
    if isgraphics(figDjsDist)
      set(figDjsDist, 'Name', [djsDistFigName, typeTag]);
    end
    titleSuffix = sprintf(' | %s | width cutoff %.3f ms', cell_type_label(cellType), widthCutoff);
  end
  sgtitle(figKappaDist, sprintf('%s%s', sessionName, titleSuffix), 'Interpreter', 'none');
  if isgraphics(figDjsDist)
    sgtitle(figDjsDist, sprintf('%s%s', sessionName, titleSuffix), 'Interpreter', 'none');
  end

  if saveFigure
    areaTag = format_areas_label(results.areas);
    kappaPlotBase = sprintf('session_kurtosis_distributions_%s_%s_%s_%s_win%.0fs_%.0f-%.0fs_N%d%s', ...
      sessionName, areaTag, analysisConfig.prgMethod, analysisConfig.surrogateMethod, prgWindow, ...
      collectStart, collectEnd, analysisConfig.finalCutoffDivisor, cell_type_file_tag(cellType));
    exportgraphics(figKappaDist, fullfile(saveDir, [kappaPlotBase, '.png']), 'Resolution', 300);
    exportgraphics(figKappaDist, fullfile(saveDir, [kappaPlotBase, '.eps']), 'ContentType', 'vector');
    fprintf('\nSaved figure: %s\n', fullfile(saveDir, kappaPlotBase));

    if isgraphics(figDjsDist)
      djsPlotBase = sprintf('session_djs_distributions_%s_%s_%s_%s_win%.0fs_%.0f-%.0fs_N%d%s', ...
        sessionName, areaTag, analysisConfig.prgMethod, analysisConfig.surrogateMethod, prgWindow, ...
        collectStart, collectEnd, analysisConfig.finalCutoffDivisor, cell_type_file_tag(cellType));
      exportgraphics(figDjsDist, fullfile(saveDir, [djsPlotBase, '.png']), 'Resolution', 300);
      exportgraphics(figDjsDist, fullfile(saveDir, [djsPlotBase, '.eps']), 'ContentType', 'vector');
      fprintf('Saved figure: %s\n', fullfile(saveDir, djsPlotBase));
    end
  end
end

if splitExcitatoryInhibitory
  areaTag = format_areas_label(brainArea);
  if isempty(areaTag)
    areaTag = 'all_areas';
  end
  kappaLabel = sprintf('\\kappa (N/%d)', analysisConfig.finalCutoffDivisor);
  djsLabel = sprintf('D_{JS} (N/%d)', analysisConfig.finalCutoffDivisor);
  kappaSummaryTitle = sprintf('%s | %s | %s mean +/- SEM across windows', ...
    sessionName, areaTag, kappaLabel);
  figKappaEiSummary = plot_session_ei_summary(eiSummary, kappaSummaryTitle, kappaLabel, {'kappa'});
  set(figKappaEiSummary, 'Name', 'Session E/I kappa summary');

  djsSummaryTitle = sprintf('%s | %s | %s mean +/- SEM across windows', ...
    sessionName, areaTag, djsLabel);
  figDjsEiSummary = plot_session_ei_summary(eiSummary, djsSummaryTitle, djsLabel, {'djs'});
  set(figDjsEiSummary, 'Name', 'Session E/I D_JS summary');

  if saveFigure
    saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    kappaPlotBase = sprintf('session_kurtosis_ei_summary_%s_%s_%s_%s_win%.0fs_%.0f-%.0fs%s', ...
      sessionName, areaTag, analysisConfig.prgMethod, analysisConfig.surrogateMethod, prgWindow, ...
      collectStart, collectEnd, session_ei_summary_file_tag());
    exportgraphics(figKappaEiSummary, fullfile(saveDir, [kappaPlotBase, '.png']), 'Resolution', 300);
    exportgraphics(figKappaEiSummary, fullfile(saveDir, [kappaPlotBase, '.eps']), 'ContentType', 'vector');
    fprintf('\nSaved E/I kappa summary figure: %s\n', fullfile(saveDir, kappaPlotBase));

    djsPlotBase = sprintf('session_djs_ei_summary_%s_%s_%s_%s_win%.0fs_%.0f-%.0fs%s', ...
      sessionName, areaTag, analysisConfig.prgMethod, analysisConfig.surrogateMethod, prgWindow, ...
      collectStart, collectEnd, session_ei_summary_file_tag());
    exportgraphics(figDjsEiSummary, fullfile(saveDir, [djsPlotBase, '.png']), 'Resolution', 300);
    exportgraphics(figDjsEiSummary, fullfile(saveDir, [djsPlotBase, '.eps']), 'ContentType', 'vector');
    fprintf('Saved E/I D_{JS} summary figure: %s\n', fullfile(saveDir, djsPlotBase));
  end
end

fprintf('\n=== Done ===\n');

%% Local functions

function fig = find_new_figure_by_name(figName, figHandlesBefore)
% FIND_NEW_FIGURE_BY_NAME - Figure created since figHandlesBefore with exact Name
%
% Variables:
%   figName           - Figure Name property (exact match)
%   figHandlesBefore  - Figure handles before the plotting call
%
% Goal: Return the new figure for this population, not an earlier one

fig = gobjects(0);
figHandlesAfter = findall(0, 'Type', 'figure');
newFigHandles = setdiff(figHandlesAfter, figHandlesBefore, 'stable');
for figIdx = 1:numel(newFigHandles)
  candidateFig = newFigHandles(figIdx);
  if isgraphics(candidateFig) && strcmp(get(candidateFig, 'Name'), figName)
    fig = candidateFig;
  end
end
end

function nUnits = count_session_neurons_for_brain_area(dataStruct, brainArea)
% COUNT_SESSION_NEURONS_FOR_BRAIN_AREA - Neurons in selected analysis area(s)
%
% Variables:
%   dataStruct - Session struct with areas and idMatIdx
%   brainArea  - Single area name; '' sums all areas
%
% Goal: Pre-check unit count before PRG analysis for one population

nUnits = 0;
if ~isfield(dataStruct, 'idMatIdx') || isempty(dataStruct.idMatIdx)
  return;
end

if isempty(brainArea)
  for areaIdx = 1:numel(dataStruct.idMatIdx)
    nUnits = nUnits + numel(dataStruct.idMatIdx{areaIdx});
  end
  return;
end

areaIdx = find(strcmp(dataStruct.areas, brainArea), 1);
if isempty(areaIdx)
  return;
end
nUnits = numel(dataStruct.idMatIdx{areaIdx});
end

function metricValues = extract_prg_summary_metric_values(results)
% EXTRACT_PRG_SUMMARY_METRIC_VALUES - Window-wise kappa and D_JS for E/I summary plots

metricValues = struct('kappa', [], 'djs', []);
if isempty(results.areas) || isempty(results.kappa)
  return;
end

validMask = isfinite(results.kappa{1}) & ~results.windowExcluded{1};
metricValues.kappa = results.kappa{1}(validMask);

if isfield(results, 'djs') && ~isempty(results.djs) && ~isempty(results.djs{1})
  djsValid = results.djs{1}(validMask);
  metricValues.djs = djsValid(isfinite(djsValid));
end
end

function results = filter_prg_results_to_brain_area(results, brainArea)
% FILTER_PRG_RESULTS_TO_BRAIN_AREA - Keep one area in PRG results struct

if isempty(brainArea) || ~isfield(results, 'areas')
  return;
end

areaIdx = find(strcmp(results.areas, brainArea), 1);
if isempty(areaIdx)
  results.areas = {};
  return;
end

cellFields = {'kappa', 'kappaByCutoff', 'windowStartS', 'popCv', 'windowExcluded', ...
  'nNeuronsPerWindow', 'kappaSurrogate', 'djs', 'djsSurrogate', 'nCutoffList'};

results.areas = results.areas(areaIdx);
for f = 1:length(cellFields)
  fieldName = cellFields{f};
  if isfield(results, fieldName) && numel(results.(fieldName)) >= areaIdx
    results.(fieldName) = results.(fieldName)(areaIdx);
  end
end
end

function print_session_kappa_summary(results)
% PRINT_SESSION_KAPPA_SUMMARY - Window counts and mean kappa / D_JS per area

fprintf('\n=== PRG kurtosis and D_{JS} summary ===\n');
for a = 1:numel(results.areas)
  if isempty(results.kappa{a})
    fprintf('  %s: no kappa data\n', results.areas{a});
    continue;
  end
  validMask = isfinite(results.kappa{a}) & ~results.windowExcluded{a};
  kappaValid = results.kappa{a}(validMask);
  nValid = numel(kappaValid);
  nExcluded = sum(results.windowExcluded{a});
  if nValid > 0
    fprintf('  %s: %d valid windows (%d CV-excluded), mean \\kappa = %.3f', ...
      results.areas{a}, nValid, nExcluded, mean(kappaValid));
    if isfield(results, 'djs') && ~isempty(results.djs{a})
      djsValid = results.djs{a}(validMask);
      djsValid = djsValid(isfinite(djsValid));
      if ~isempty(djsValid)
        fprintf(', mean D_{JS} = %.4f', mean(djsValid));
      end
    end
    fprintf('\n');
  else
    fprintf('  %s: no valid windows (%d CV-excluded)\n', results.areas{a}, nExcluded);
  end
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
