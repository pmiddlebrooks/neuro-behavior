%%
% Criticality PRG Analysis Across Task Types (Manuscript)
%
% Runs momentum-space PRG kurtosis in non-overlapping windows of length
% prgWindow seconds, batches across session types, and plots session summaries
% grouped by sessionType.
%
% Variables (configure in this section):
%   sessionTypes   - Cell array of session types to include
%   dataSource     - 'spikes' or 'lfp'
%   collectStart   - Analysis window start (seconds from session onset)
%   collectEnd     - Analysis window end (seconds); [] = full session
%   prgWindow      - Non-overlapping block length (seconds); blockWindowSize.
%                    [] = one block spanning the full collect duration (like d2Window)
%   brainArea              - Single or merged area (e.g. 'M56', 'M23M56'); '' = all valid areas
%   brainAreaCombinations  - Merged areas: struct('name', 'M23M56', 'areas', {{'M23','M56'}})
%   areasToPlot            - Area names to plot; {} uses brainArea if set
%   runBatch       - If true, run criticality_prg_analysis per session
%   plotResults    - If true, create summary figures after batch
%   prgMethod        - 'pca' (momentum-space) or 'icg' (real-space ICG, Morales 2023)
%   surrogateMethod  - 'isi' (ISI shuffle per unit) or 'circular' (circshift per neuron)
%   useSubsampling   - If true, kappa/D_JS per window = mean across neuron subsamples
%   nSubsamples, nNeuronsSubsample, minNeuronsMultiple - subsampling settings
%   splitExcitatoryInhibitory - If true, run combined (E+I), excitatory, and inhibitory
%   widthCutoff      - Peak-to-trough width threshold in ms (narrow <= cutoff = I)
%
% Goal:
%   Compare PRG kurtosis (kappa at N/finalCutoffDivisor) and Jensen-Shannon distance
%   (D_JS vs Gaussian) across session types, including values normalized by per-window
%   surrogate means. Per session: mean and SEM across valid windows.
%   See: CambrainhaC_etal_2025_PRXLife; CambrainhaC_etal_2026_arXiv

function out = criticality_prg_across_tasks(opts)
% CRITICALITY_PRG_ACROSS_TASKS - Batch and plot PRG kurtosis across session types
%
% Usage:
%   opts = criticality_prg_across_tasks();                  % default opts
%   out = criticality_prg_across_tasks(opts);
%   out = criticality_prg_across_tasks(struct('plotResults', false));
%
% Returns:
%   out.batchResults, out.plotData, out.batchMeta, out.paths
%   (or default opts struct when called with no arguments)

if nargin == 0
  out = fill_criticality_prg_across_tasks_opts(struct());
  return;
end

if nargin < 1 || isempty(opts)
  opts = struct();
end
opts = fill_criticality_prg_across_tasks_opts(opts);
setup_criticality_manuscript_paths('criticality_prg_across_tasks');
paths = get_paths();

if isempty(opts.batchResultsFile)
  batchName = 'criticality_prg_across_tasks_batch.mat';
  if opts.splitExcitatoryInhibitory
    batchName = 'criticality_prg_across_tasks_batch_ei_split.mat';
  end
  opts.batchResultsFile = fullfile(paths.dropPath, 'criticality_manuscript', batchName);
end

fprintf('\n=== Criticality PRG Across Task Types ===\n');
fprintf('PRG method: %s\n', opts.prgMethod);
if isempty(opts.collectEnd)
  fprintf('Collect window: [%.1f, full] s\n', opts.collectStart);
else
  fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', opts.collectStart, opts.collectEnd, ...
    (opts.collectEnd - opts.collectStart) / 60);
end
if isempty(opts.prgWindow)
  fprintf('PRG blocks: full collect duration (one block per session)\n');
else
  fprintf('PRG blocks: %.1f s, non-overlapping\n', opts.prgWindow);
end
fprintf('Kappa at N/%d; bin size: %.3f s; surrogates: %s\n', ...
  opts.finalCutoffDivisor, opts.binSize, opts.surrogateMethod);
if opts.useSubsampling
  fprintf('Subsampling: %d subsets x %d neurons (min neurons x %.2f)\n', ...
    opts.nSubsamples, opts.nNeuronsSubsample, opts.minNeuronsMultiple);
else
  fprintf('Subsampling: off\n');
end
if opts.splitExcitatoryInhibitory
  fprintf('E/I split: on (widthCutoff = %.3f ms)\n', opts.widthCutoff);
end
fprintf('Session types: %s\n', strjoin(opts.sessionTypes, ', '));
if ~isempty(opts.brainArea)
  fprintf('Brain area: %s (single-area analysis)\n', opts.brainArea);
else
  fprintf('Brain area: all areas in each session\n');
end

sessionTable = build_session_table(opts.sessionTypes);
numSessions = size(sessionTable, 1);
fprintf('Total sessions: %d\n', numSessions);
if numSessions == 0
  error('No sessions found for the requested session types.');
end

cellTypesToRun = get_session_cell_types_to_run(opts.splitExcitatoryInhibitory);

if opts.runBatch
  batchByCell = run_prg_across_tasks_batch(sessionTable, opts, paths);
  batchMeta = pack_prg_across_tasks_batch_meta(opts);
  plotDataByCell = cell(1, numel(cellTypesToRun));
  for iCell = 1:numel(cellTypesToRun)
    plotDataByCell{iCell} = aggregate_prg_metrics(batchByCell{iCell}, opts.sessionTypes, opts.finalCutoffDivisor);
    plotDataByCell{iCell}.surrogateMethod = opts.surrogateMethod;
  end
  if opts.saveBatchResults
    batchResults = batchByCell{1};
    plotData = plotDataByCell{1};
    save(opts.batchResultsFile, 'batchByCell', 'plotDataByCell', 'cellTypesToRun', ...
      'batchResults', 'plotData', 'batchMeta', '-v7.3');
    fprintf('\nSaved batch results: %s\n', opts.batchResultsFile);
  end
else
  if ~isfile(opts.batchResultsFile)
    error('criticality_prg_across_tasks:NoBatchFile', ...
      'Batch file not found: %s. Set runBatch true to compute.', opts.batchResultsFile);
  end
  loaded = load(opts.batchResultsFile);
  batchMeta = loaded.batchMeta;
  if isfield(loaded, 'batchByCell')
    batchByCell = loaded.batchByCell;
    cellTypesToRun = loaded.cellTypesToRun;
    if isfield(loaded, 'plotDataByCell')
      plotDataByCell = loaded.plotDataByCell;
    else
      plotDataByCell = cell(1, numel(batchByCell));
      for iCell = 1:numel(batchByCell)
        plotDataByCell{iCell} = aggregate_prg_metrics(batchByCell{iCell}, opts.sessionTypes, opts.finalCutoffDivisor);
      end
    end
  else
    batchByCell = {loaded.batchResults};
    plotDataByCell = {loaded.plotData};
    cellTypesToRun = {''};
  end
  for iCell = 1:numel(plotDataByCell)
    if (~isfield(plotDataByCell{iCell}, 'surrogateMethod') || isempty(plotDataByCell{iCell}.surrogateMethod)) ...
        && isfield(batchMeta, 'surrogateMethod')
      plotDataByCell{iCell}.surrogateMethod = batchMeta.surrogateMethod;
    elseif ~isfield(plotDataByCell{iCell}, 'surrogateMethod') || isempty(plotDataByCell{iCell}.surrogateMethod)
      plotDataByCell{iCell}.surrogateMethod = opts.surrogateMethod;
    end
  end
  fprintf('\nLoaded batch results: %s\n', opts.batchResultsFile);
end

batchResults = batchByCell{1};
plotData = plotDataByCell{1};

anyAreas = false;
for iCell = 1:numel(cellTypesToRun)
  cellType = cellTypesToRun{iCell};
  plotDataCell = plotDataByCell{iCell};
  if isempty(plotDataCell.areas)
    warning('criticality_prg_across_tasks:NoMetrics', ...
      'No PRG metrics for %s. Skipping.', cell_type_label(cellType));
    continue;
  end
  anyAreas = true;
  if opts.plotResults
    commonAreas = resolve_areas_to_plot(plotDataCell.areas, opts.areasToPlot, opts.brainArea);

    fprintf('\n=== Areas for plotting (%s) ===\n', cell_type_label(cellType));
    fprintf('  %s\n', strjoin(commonAreas, ', '));

    plot_prg_across_tasks(plotDataCell, commonAreas, opts.sessionTypes, opts.collectStart, ...
      opts.collectEnd, opts.prgWindow, paths, opts.brainArea, opts.prgMethod, ...
      opts.surrogateMethod, cellType);
  end
end
if ~anyAreas
  error('No PRG metrics extracted. Check that batch analyses succeeded.');
end

fprintf('\n=== Done ===\n');

out = struct();
out.batchResults = batchResults;
out.plotData = plotData;
out.batchByCell = batchByCell;
out.plotDataByCell = plotDataByCell;
out.cellTypesToRun = cellTypesToRun;
out.batchMeta = batchMeta;
out.paths = paths;
out.areasToPlot = opts.areasToPlot;
end

function opts = fill_criticality_prg_across_tasks_opts(opts)
defaults = struct();
defaults.sessionTypes = {'spontaneous', 'interval', 'reach'};
defaults.dataSource = 'spikes';
defaults.collectStart = 0;
defaults.collectEnd = 45 * 60;
defaults.prgWindow = 30;
defaults.brainArea = 'M23M56';
defaults.brainAreaCombinations = default_manuscript_brain_area_combinations();
defaults.areasToPlot = {};
defaults.runBatch = true;
defaults.plotResults = true;
defaults.saveBatchResults = false;
defaults.batchResultsFile = '';
defaults.useSubsampling = false;
defaults.nSubsamples = 10;
defaults.nNeuronsSubsample = 40;
defaults.minNeuronsMultiple = 1.25;
defaults.surrogateMethod = 'isi';
defaults.prgMethod = 'pca';
defaults.binSize = 0.05;
defaults.cvThreshold = 5;
defaults.cutoffDivisors = [1, 2, 4, 8, 16, 32];
defaults.finalCutoffDivisor = 4;
defaults.kappaAxisMax = 20;
defaults.enableSurrogates = true;
defaults.nSurrogates = 10;
defaults.nMinNeurons = 32;
defaults.firingRateCheckTime = [];
defaults.minFiringRate = 0.05;
defaults.maxFiringRate = 100;
defaults.splitExcitatoryInhibitory = false;
defaults.widthCutoff = 0.35;
% Empty collectEnd / prgWindow are sentinels for "full session" — do not replace
preserveCollectEndEmpty = isfield(opts, 'collectEnd') && isempty(opts.collectEnd);
preservePrgWindowEmpty = isfield(opts, 'prgWindow') && isempty(opts.prgWindow);
opts = merge_struct_defaults(opts, defaults);
if preserveCollectEndEmpty
  opts.collectEnd = [];
end
if preservePrgWindowEmpty
  opts.prgWindow = [];
end
end

function batchMeta = pack_prg_across_tasks_batch_meta(opts)
batchMeta = struct( ...
  'sessionTypes', {opts.sessionTypes}, ...
  'collectStart', opts.collectStart, ...
  'collectEnd', opts.collectEnd, ...
  'prgWindow', opts.prgWindow, ...
  'brainArea', opts.brainArea, ...
  'areasToPlot', {opts.areasToPlot}, ...
  'prgMethod', opts.prgMethod, ...
  'surrogateMethod', opts.surrogateMethod, ...
  'finalCutoffDivisor', opts.finalCutoffDivisor, ...
  'splitExcitatoryInhibitory', opts.splitExcitatoryInhibitory, ...
  'widthCutoff', opts.widthCutoff);
end

function batchByCell = run_prg_across_tasks_batch(sessionTable, opts, paths)
analysisConfig = build_prg_analysis_config(opts);
loadOpts = neuro_behavior_options();
loadOpts.firingRateCheckTime = opts.firingRateCheckTime;
loadOpts.collectStart = opts.collectStart;
loadOpts.collectEnd = opts.collectEnd;
loadOpts.minFiringRate = opts.minFiringRate;
loadOpts.maxFiringRate = opts.maxFiringRate;

cellTypesToRun = get_session_cell_types_to_run(opts.splitExcitatoryInhibitory);
numSessions = size(sessionTable, 1);
batchByCell = cell(1, numel(cellTypesToRun));
emptyEntry = struct('sessionType', '', 'sessionName', '', 'subjectName', '', ...
  'label', '', 'cellType', '', 'success', false, 'results', [], 'skipReason', '');
for iCell = 1:numel(cellTypesToRun)
  batchByCell{iCell} = repmat(emptyEntry, numSessions, 1);
end
if isempty(opts.prgWindow)
  fprintf('\n=== Running PRG analysis (one block over full collect duration) ===\n');
else
  fprintf('\n=== Running PRG analysis (%.0f s non-overlapping blocks) ===\n', opts.prgWindow);
end

for s = 1:numSessions
  sessionType = sessionTable.sessionType{s};
  sessionName = sessionTable.sessionName{s};
  subjectName = sessionTable.subjectName{s};

  fprintf('\n%s\n', repmat('=', 1, 80));
  fprintf('Session %d/%d [%s]: %s\n', s, numSessions, sessionType, sessionName);
  if ~isempty(subjectName)
    fprintf('  subjectName: %s\n', subjectName);
  end

  for iCell = 1:numel(cellTypesToRun)
    batchByCell{iCell}(s).sessionType = sessionType;
    batchByCell{iCell}(s).sessionName = sessionName;
    batchByCell{iCell}(s).subjectName = subjectName;
    batchByCell{iCell}(s).label = sessionTable.label{s};
    batchByCell{iCell}(s).cellType = cellTypesToRun{iCell};
    batchByCell{iCell}(s).success = false;
    batchByCell{iCell}(s).results = [];
  end

  try
    loadArgs = build_session_load_args(sessionType, sessionName, loadOpts, subjectName);
    dataStruct = load_session_data(sessionType, opts.dataSource, loadArgs{:});
    [dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
      dataStruct, opts.brainArea, opts.brainAreaCombinations);
    if ~areaOk
      fprintf('  Brain area "%s" not available in this session; skipping.\n', opts.brainArea);
      continue;
    end

    if opts.splitExcitatoryInhibitory
      eiCheck = check_session_ei_neuron_counts(dataStruct, paths, opts.widthCutoff, ...
        opts.brainArea, opts.brainAreaCombinations, analysisConfig.nMinNeurons);
      if ~eiCheck.isOk
        for iCell = 1:numel(cellTypesToRun)
          batchByCell{iCell}(s).skipReason = 'Insufficient E/I neurons';
        end
        continue;
      end
    end

    sessionConfig = analysisConfig;
    sessionDuration = get_session_collect_duration(dataStruct, opts);
    durationToleranceSec = 1;
    useFullSessionBlock = isempty(opts.prgWindow) ...
      || sessionDuration < (opts.prgWindow - durationToleranceSec);
    if useFullSessionBlock
      if isempty(opts.prgWindow)
        fprintf('  Using full session block (%.1f s).\n', sessionDuration);
      else
        fprintf('  Session duration %.1f s < requested prgWindow %.1f s; using full session block.\n', ...
          sessionDuration, opts.prgWindow);
      end
      sessionConfig.blockWindowSize = sessionDuration;
    end

    for iCell = 1:numel(cellTypesToRun)
      cellType = cellTypesToRun{iCell};
      try
        dataStructRun = prepare_session_data_for_cell_type(dataStruct, paths, cellType, ...
          opts.widthCutoff, opts.splitExcitatoryInhibitory);
        [dataStructRun, ~] = apply_manuscript_brain_area_selection( ...
          dataStructRun, opts.brainArea, opts.brainAreaCombinations);

        prgResults = criticality_prg_analysis(dataStructRun, sessionConfig);
        if ~isempty(opts.brainArea)
          prgResults = filter_prg_results_to_brain_area(prgResults, opts.brainArea);
          if isempty(prgResults.areas)
            fprintf('  No results for brain area "%s" (%s); skipping.\n', ...
              opts.brainArea, cell_type_label(cellType));
            continue;
          end
        end

        batchByCell{iCell}(s).success = true;
        batchByCell{iCell}(s).results = prgResults;
        fprintf('  Analysis completed (%s).\n', cell_type_label(cellType));
      catch MECell
        if is_skippable_session_analysis_error(MECell)
          fprintf('  Skipping %s (insufficient neurons / no valid areas): %s\n', ...
            cell_type_label(cellType), MECell.message);
          batchByCell{iCell}(s).skipReason = MECell.message;
          continue;
        end
        rethrow(MECell);
      end
    end
  catch ME
    if is_skippable_session_analysis_error(ME)
      fprintf('  Skipping session (insufficient neurons / no valid areas): %s\n', ME.message);
      for iCell = 1:numel(cellTypesToRun)
        batchByCell{iCell}(s).skipReason = ME.message;
      end
      continue;
    end
    fprintf('  Error: %s\n', ME.message);
    for st = 1:length(ME.stack)
      fprintf('    %s (line %d)\n', ME.stack(st).name, ME.stack(st).line);
    end
    error('criticality_prg_across_tasks:SessionFailed', ...
      'Batch stopped at session %d/%d [%s] %s: %s', ...
      s, numSessions, sessionType, sessionName, ME.message);
  end
end
end

function tf = is_skippable_session_analysis_error(ME)
% IS_SKIPPABLE_SESSION_ANALYSIS_ERROR - True for expected per-session skip cases
tf = contains(ME.message, 'No valid areas to process') ...
  || contains(ME.message, 'insufficient neurons');
end

function analysisConfig = build_prg_analysis_config(opts)
analysisConfig = struct();
analysisConfig.prgMethod = opts.prgMethod;
if isempty(opts.prgWindow)
  % Placeholder; overwritten per session with collect duration
  analysisConfig.blockWindowSize = 1;
else
  analysisConfig.blockWindowSize = opts.prgWindow;
end
analysisConfig.binSize = opts.binSize;
analysisConfig.cvThreshold = opts.cvThreshold;
analysisConfig.cutoffDivisors = opts.cutoffDivisors;
analysisConfig.finalCutoffDivisor = opts.finalCutoffDivisor;
analysisConfig.kappaAxisMax = opts.kappaAxisMax;
analysisConfig.enableSurrogates = opts.enableSurrogates;
analysisConfig.nSurrogates = opts.nSurrogates;
analysisConfig.surrogateMethod = opts.surrogateMethod;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.nMinNeurons = opts.nMinNeurons;
analysisConfig.useSubsampling = opts.useSubsampling;
analysisConfig.nSubsamples = opts.nSubsamples;
analysisConfig.nNeuronsSubsample = opts.nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = opts.minNeuronsMultiple;
end

function sessionDuration = get_session_collect_duration(dataStruct, opts)
% GET_SESSION_COLLECT_DURATION - Actual loaded collect window length (s)

if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectEnd') ...
    && isfield(dataStruct.spikeData, 'collectStart')
  sessionDuration = dataStruct.spikeData.collectEnd - dataStruct.spikeData.collectStart;
elseif isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectEnd') ...
    && ~isempty(dataStruct.opts.collectEnd)
  collectStart = 0;
  if isfield(dataStruct.opts, 'collectStart') && ~isempty(dataStruct.opts.collectStart)
    collectStart = dataStruct.opts.collectStart;
  end
  sessionDuration = dataStruct.opts.collectEnd - collectStart;
elseif isfield(dataStruct, 'spikeTimes') && ~isempty(dataStruct.spikeTimes)
  collectStart = opts.collectStart;
  if isempty(collectStart)
    collectStart = 0;
  end
  sessionDuration = max(dataStruct.spikeTimes) - collectStart;
else
  sessionDuration = opts.collectEnd - opts.collectStart;
  if isempty(sessionDuration)
    error('criticality_prg_across_tasks:UnknownSessionDuration', ...
      'Could not determine session collect duration.');
  end
end
end

function commonAreas = resolve_areas_to_plot(plotAreas, areasToPlot, brainArea)
if isempty(areasToPlot) && ~isempty(brainArea)
  areasToPlot = {brainArea};
end
commonAreas = plotAreas;
if ~isempty(areasToPlot)
  commonAreas = intersect(commonAreas, areasToPlot, 'stable');
  if isempty(commonAreas)
    error('None of areasToPlot are present in the aggregated results.');
  end
elseif ~isempty(brainArea)
  commonAreas = intersect(commonAreas, {brainArea}, 'stable');
  if isempty(commonAreas)
    error('No results for brainArea "%s". Check that sessions include this area.', brainArea);
  end
end
end

function sessionTable = build_session_table(sessionTypes)
% BUILD_SESSION_TABLE - Flatten session lists from each session type

sessionTypeCol = {};
sessionNameCol = {};
subjectNameCol = {};
labelCol = {};

for t = 1:length(sessionTypes)
  sessionType = sessionTypes{t};
  entries = get_sessions_for_type(sessionType);
  nEntries = numel(entries);

  for i = 1:nEntries
    sessionTypeCol{end+1, 1} = sessionType; %#ok<AGROW>
    sessionNameCol{end+1, 1} = entries(i).sessionName; %#ok<AGROW>
    if isfield(entries, 'subjectName')
      subjectNameCol{end+1, 1} = entries(i).subjectName; %#ok<AGROW>
    else
      subjectNameCol{end+1, 1} = ''; %#ok<AGROW>
    end
    labelCol{end+1, 1} = make_session_label(sessionType, entries(i)); %#ok<AGROW>
  end
end

sessionTable = table(sessionTypeCol, sessionNameCol, subjectNameCol, labelCol, ...
  'VariableNames', {'sessionType', 'sessionName', 'subjectName', 'label'});
end

function entries = get_sessions_for_type(sessionType)
% GET_SESSIONS_FOR_TYPE - Struct array with subjectName and sessionName

switch lower(sessionType)
  case 'spontaneous'
    entries = spontaneous_session_list();
  case 'interval'
    entries = interval_session_list();
  case 'reach'
    names = reach_session_list();
    entries = struct('subjectName', {}, 'sessionName', {});
    for i = 1:length(names)
      entries(i).subjectName = '';
      entries(i).sessionName = names{i};
    end
  case 'schall'
    names = schall_session_list();
    entries = struct('subjectName', {}, 'sessionName', {});
    for i = 1:length(names)
      parts = strsplit(names{i}, '/');
      if numel(parts) >= 2
        entries(i).subjectName = parts{1};
        entries(i).sessionName = parts{2};
      else
        entries(i).subjectName = '';
        entries(i).sessionName = names{i};
      end
    end
  otherwise
    error('Unknown sessionType: %s', sessionType);
end

if ~isstruct(entries) || ~isfield(entries, 'sessionName')
  error('Session list for %s must return a struct array with sessionName.', sessionType);
end
end

function label = make_session_label(~, entry)
% MAKE_SESSION_LABEL - Short display label for plots
label = entry.sessionName;
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
  'nNeuronsPerWindow', 'kappaSurrogate', 'djs', 'djsSurrogate', 'nCutoffList', ...
  'kappaSubsamples', 'djsSubsamples'};

results.areas = results.areas(areaIdx);
for f = 1:length(cellFields)
  fieldName = cellFields{f};
  if isfield(results, fieldName) && numel(results.(fieldName)) >= areaIdx
    results.(fieldName) = results.(fieldName)(areaIdx);
  end
end
end

function plotData = aggregate_prg_metrics(batchResults, sessionTypes, finalCutoffDivisor)
% AGGREGATE_PRG_METRICS - Per-session mean and SEM of window kappa / D_JS and surrogates
%
% Variables:
%   finalCutoffDivisor - Reported in plot labels (N/divisor)

if nargin < 3 || isempty(finalCutoffDivisor)
  finalCutoffDivisor = 16;
end

plotData = struct();
plotData.areas = {};
plotData.sessionTypes = sessionTypes;
plotData.byType = struct();
plotData.finalCutoffDivisor = finalCutoffDivisor;
plotData.surrogateMethod = '';

metricFields = {'kappaMean', 'kappaSem', 'kappaShuffleMean', 'kappaShuffleSem', ...
  'kappaNormMean', 'kappaNormSem', ...
  'djsMean', 'djsSem', 'djsShuffleMean', 'djsShuffleSem', ...
  'djsNormMean', 'djsNormSem'};

for s = 1:length(batchResults)
  if ~batchResults(s).success || isempty(batchResults(s).results)
    continue;
  end

  results = batchResults(s).results;
  sessionType = batchResults(s).sessionType;

  if isempty(plotData.areas)
    for t = 1:length(sessionTypes)
      st = sessionTypes{t};
      plotData.byType.(matlab.lang.makeValidName(st)) = init_type_prg_metrics(metricFields, 0);
    end
  end

  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    plotData.byType.(typeKey) = init_type_prg_metrics(metricFields, length(plotData.areas));
  end
  typeData = plotData.byType.(typeKey);

  for a = 1:length(results.areas)
    areaName = results.areas{a};
    areaIdx = find(strcmp(plotData.areas, areaName), 1);
    if isempty(areaIdx)
      plotData.areas{end+1} = areaName;
      areaIdx = length(plotData.areas);
      plotData = extend_prg_plot_data_areas(plotData, sessionTypes, metricFields, areaIdx);
      typeData = plotData.byType.(typeKey);
    end

    summary = summarize_session_kappa_windows(results, a);
    for m = 1:length(metricFields)
      fieldName = metricFields{m};
      typeData.(fieldName){areaIdx} = [typeData.(fieldName){areaIdx}, summary.(fieldName)];
    end
  end

  typeData.sessionLabels{end+1} = batchResults(s).label;
  typeData.sessionNames{end+1} = batchResults(s).sessionName;
  plotData.byType.(typeKey) = typeData;
end
end

function summary = summarize_session_kappa_windows(results, areaIdx)
% SUMMARIZE_SESSION_KAPPA_WINDOWS - Mean and SEM across valid windows; shuffle summary
%
% Variables:
%   results  - Output from criticality_prg_analysis
%   areaIdx  - Index into results.areas
%
% Returns:
%   summary - kappaMean, kappaSem, kappaShuffleMean, kappaShuffleSem,
%             kappaNormMean, kappaNormSem,
%             djsMean, djsSem, djsShuffleMean, djsShuffleSem,
%             djsNormMean, djsNormSem
%
%   With useSubsampling and a single valid window, kappaSem / djsSem are SEM
%   across neuron subsamples. Otherwise SEM is across window means.
%   Normalized metrics: per-window value / mean(surrogate for that window), then
%   mean and SEM across valid windows. *ShuffleSem - SEM across windows of per-window surrogate means

summary = struct('kappaMean', nan, 'kappaSem', nan, 'kappaShuffleMean', nan, ...
  'kappaShuffleSem', nan, 'kappaNormMean', nan, 'kappaNormSem', nan, ...
  'djsMean', nan, 'djsSem', nan, 'djsShuffleMean', nan, 'djsShuffleSem', nan, ...
  'djsNormMean', nan, 'djsNormSem', nan);

if areaIdx > length(results.kappa) || isempty(results.kappa{areaIdx})
  return;
end

useSubsampling = isfield(results, 'useSubsampling') && results.useSubsampling;
kappaVec = results.kappa{areaIdx}(:);
nWin = numel(kappaVec);
excluded = false(nWin, 1);
if isfield(results, 'windowExcluded') && areaIdx <= length(results.windowExcluded) ...
    && ~isempty(results.windowExcluded{areaIdx})
  excluded = results.windowExcluded{areaIdx}(:);
  if numel(excluded) ~= nWin
    excluded = false(nWin, 1);
  end
end
validMask = isfinite(kappaVec) & ~excluded;
kappaValid = kappaVec;
kappaValid(~validMask) = nan;
kappaSubMat = [];
if useSubsampling && isfield(results, 'kappaSubsamples') && areaIdx <= numel(results.kappaSubsamples) ...
    && ~isempty(results.kappaSubsamples{areaIdx})
  kappaSubMat = results.kappaSubsamples{areaIdx};
end
[summary.kappaMean, summary.kappaSem] = mean_sem_across_windows_or_subsamples( ...
  kappaValid, kappaSubMat, useSubsampling);

if isfield(results, 'djs') && areaIdx <= length(results.djs) && ~isempty(results.djs{areaIdx})
  djsVec = results.djs{areaIdx}(:);
  if numel(djsVec) == nWin
    djsValid = djsVec;
    djsValid(~validMask) = nan;
    djsSubMat = [];
    if useSubsampling && isfield(results, 'djsSubsamples') && areaIdx <= numel(results.djsSubsamples) ...
        && ~isempty(results.djsSubsamples{areaIdx})
      djsSubMat = results.djsSubsamples{areaIdx};
    end
    [summary.djsMean, summary.djsSem] = mean_sem_across_windows_or_subsamples( ...
      djsValid, djsSubMat, useSubsampling);
  end
end

if isfield(results, 'kappaSurrogate') && areaIdx <= length(results.kappaSurrogate) ...
    && ~isempty(results.kappaSurrogate{areaIdx})
  surrMat = results.kappaSurrogate{areaIdx};
  if size(surrMat, 1) == nWin
    perWindowShuffleMean = get_per_window_shuffle_mean_matrix(surrMat, results);

    normVec = normalize_prg_by_surrogate(kappaVec, perWindowShuffleMean);
    normValid = normVec;
    normValid(~validMask) = nan;
    [summary.kappaNormMean, summary.kappaNormSem] = mean_sem_across_windows_or_subsamples( ...
      normValid, [], false);

    perWindowShuffleMean = perWindowShuffleMean(validMask);
    shuffleValid = perWindowShuffleMean(isfinite(perWindowShuffleMean));
    if ~isempty(shuffleValid)
      summary.kappaShuffleMean = mean(shuffleValid);
      nSh = numel(shuffleValid);
      if nSh > 1
        summary.kappaShuffleSem = std(shuffleValid) / sqrt(nSh);
      else
        summary.kappaShuffleSem = 0;
      end
    end
  end
end

if isfield(results, 'djsSurrogate') && areaIdx <= length(results.djsSurrogate) ...
    && ~isempty(results.djsSurrogate{areaIdx}) ...
    && isfield(results, 'djs') && areaIdx <= length(results.djs) ...
    && ~isempty(results.djs{areaIdx})
  djsVec = results.djs{areaIdx}(:);
  djsSurrMat = results.djsSurrogate{areaIdx};
  if numel(djsVec) == nWin && size(djsSurrMat, 1) == nWin
    perWindowDjsShuffleMean = get_per_window_shuffle_mean_matrix(djsSurrMat, results);

    normVec = normalize_prg_by_surrogate(djsVec, perWindowDjsShuffleMean);
    normValid = normVec;
    normValid(~validMask) = nan;
    [summary.djsNormMean, summary.djsNormSem] = mean_sem_across_windows_or_subsamples( ...
      normValid, [], false);

    perWindowDjsShuffleMean = perWindowDjsShuffleMean(validMask);
    shuffleValid = perWindowDjsShuffleMean(isfinite(perWindowDjsShuffleMean));
    if ~isempty(shuffleValid)
      summary.djsShuffleMean = mean(shuffleValid);
      nDjsSh = numel(shuffleValid);
      if nDjsSh > 1
        summary.djsShuffleSem = std(shuffleValid) / sqrt(nDjsSh);
      else
        summary.djsShuffleSem = 0;
      end
    end
  end
end
end

function normVec = normalize_prg_by_surrogate(dataVec, surrogateMeanVec)
% NORMALIZE_PRG_BY_SURROGATE - Per-window ratio to mean surrogate value
%
% Variables:
%   dataVec           - Metric per window (column vector)
%   surrogateMeanVec  - Mean surrogate metric per window
%
% Goal:
%   Match criticality_av_analysis normalization: metric / mean(surrogate).

dataVec = dataVec(:);
surrogateMeanVec = surrogateMeanVec(:);
normVec = nan(size(dataVec));
if numel(dataVec) ~= numel(surrogateMeanVec)
  return;
end

validDenom = isfinite(dataVec) & isfinite(surrogateMeanVec) & surrogateMeanVec > 0;
normVec(validDenom) = dataVec(validDenom) ./ surrogateMeanVec(validDenom);
end

function plotData = extend_prg_plot_data_areas(plotData, sessionTypes, metricFields, newAreaIdx)
% EXTEND_PRG_PLOT_DATA_AREAS - Grow metric storage when a new area appears

typeNames = fieldnames(plotData.byType);
for i = 1:length(typeNames)
  typeData = plotData.byType.(typeNames{i});
  for m = 1:length(metricFields)
    fieldName = metricFields{m};
    while length(typeData.(fieldName)) < newAreaIdx
      typeData.(fieldName){end+1} = [];
    end
  end
  plotData.byType.(typeNames{i}) = typeData;
end
end

function typeData = init_type_prg_metrics(metricFields, numAreas)
% INIT_TYPE_PRG_METRICS - Empty storage per area for one session type

typeData = struct();
typeData.sessionLabels = {};
typeData.sessionNames = {};
for m = 1:length(metricFields)
  typeData.(metricFields{m}) = cell(1, numAreas);
  for a = 1:numAreas
    typeData.(metricFields{m}){a} = [];
  end
end
end

function plot_prg_across_tasks(plotData, areasToPlot, sessionTypes, collectStart, collectEnd, prgWindow, paths, brainArea, prgMethod, surrogateMethod, cellType)
% PLOT_PRG_ACROSS_TASKS - Session kappa and D_JS with surrogate summary, by session type

if nargin < 8 || isempty(brainArea)
  brainArea = '';
end
if nargin < 9 || isempty(prgMethod)
  prgMethod = 'pca';
end
if nargin < 10 || isempty(surrogateMethod)
  if isfield(plotData, 'surrogateMethod') && ~isempty(plotData.surrogateMethod)
    surrogateMethod = plotData.surrogateMethod;
  else
    surrogateMethod = 'isi';
  end
end
if nargin < 11 || isempty(cellType)
  cellType = '';
end
cellTag = cell_type_file_tag(cellType);

finalCutoffDivisor = 16;
if isfield(plotData, 'finalCutoffDivisor') && ~isempty(plotData.finalCutoffDivisor)
  finalCutoffDivisor = plotData.finalCutoffDivisor;
end

kappaYLabel = sprintf('\\kappa (N/%d, mean \\pm SEM across windows)', finalCutoffDivisor);
kappaTitleWord = sprintf('\\kappa (N/%d)', finalCutoffDivisor);
djsYLabel = sprintf('D_{JS} (N/%d vs Gaussian, mean \\pm SEM across windows)', finalCutoffDivisor);
djsTitleWord = sprintf('D_{JS} (N/%d)', finalCutoffDivisor);

numSessionTypes = length(sessionTypes);
typeColors = lines(max(numSessionTypes, 3));
shuffleBarColor = [0.55, 0.55, 0.55];

saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

for a = 1:length(areasToPlot)
  areaName = areasToPlot{a};
  areaIdx = find(strcmp(plotData.areas, areaName), 1);
  if isempty(areaIdx) || ~area_has_prg_plot_data(plotData, sessionTypes, areaIdx)
    continue;
  end

  figKappa = figure(6000 + a);
  clf(figKappa);
  position_figure_full_monitor(figKappa);
  axKappa = axes(figKappa);
  plot_kappa_with_shuffle(axKappa, plotData, sessionTypes, areaIdx, typeColors, shuffleBarColor);
  ylabel(axKappa, kappaYLabel);
  title(axKappa, sprintf('%s — %s', areaName, kappaTitleWord));
  yline(axKappa, 3, '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');

  if ~isempty(brainArea)
    titleStr = format_prg_sgtitle(prgMethod, kappaTitleWord, brainArea, ...
      collectStart, collectEnd, prgWindow);
  else
    titleStr = format_prg_sgtitle(prgMethod, kappaTitleWord, areaName, ...
      collectStart, collectEnd, prgWindow);
  end
  if ~isempty(cellType)
    titleStr = sprintf('%s | %s', titleStr, cell_type_label(cellType));
  end
  sgtitle(figKappa, titleStr, 'FontWeight', 'bold');

  plotBase = make_prg_plot_basename('criticality_prg_across_tasks', areaName, brainArea, ...
    prgWindow, collectStart, collectEnd, length(areasToPlot) > 1, finalCutoffDivisor, ...
    prgMethod, surrogateMethod);
  plotBase = [plotBase, cellTag];
  exportgraphics(figKappa, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(figKappa, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBase));

  if area_has_prg_metric_plot_data(plotData, sessionTypes, areaIdx, 'kappaNormMean')
    kappaNormYLabel = sprintf('\\kappa / surrogate (N/%d, mean \\pm SEM across windows)', finalCutoffDivisor);
    kappaNormTitleWord = sprintf('\\kappa / surrogate (N/%d)', finalCutoffDivisor);

    figKappaNorm = figure(6200 + a);
    clf(figKappaNorm);
    position_figure_full_monitor(figKappaNorm);
    axKappaNorm = axes(figKappaNorm);
    plot_prg_normalized_sessions(axKappaNorm, plotData, sessionTypes, areaIdx, ...
      typeColors, 'kappaNormMean', 'kappaNormSem', 'session \kappa / surrogate');
    ylabel(axKappaNorm, kappaNormYLabel);
    title(axKappaNorm, sprintf('%s — %s', areaName, kappaNormTitleWord));
    yline(axKappaNorm, 1, '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');

    if ~isempty(brainArea)
      kappaNormTitleStr = format_prg_sgtitle(prgMethod, kappaNormTitleWord, brainArea, ...
        collectStart, collectEnd, prgWindow);
    else
      kappaNormTitleStr = format_prg_sgtitle(prgMethod, kappaNormTitleWord, areaName, ...
        collectStart, collectEnd, prgWindow);
    end
    if ~isempty(cellType)
      kappaNormTitleStr = sprintf('%s | %s', kappaNormTitleStr, cell_type_label(cellType));
    end
    sgtitle(figKappaNorm, kappaNormTitleStr, 'FontWeight', 'bold');

    plotBaseNorm = make_prg_plot_basename('criticality_prg_kappa_norm_across_tasks', areaName, brainArea, ...
      prgWindow, collectStart, collectEnd, length(areasToPlot) > 1, finalCutoffDivisor, ...
      prgMethod, surrogateMethod);
    plotBaseNorm = [plotBaseNorm, cellTag];
    exportgraphics(figKappaNorm, fullfile(saveDir, [plotBaseNorm, '.png']), 'Resolution', 300);
    exportgraphics(figKappaNorm, fullfile(saveDir, [plotBaseNorm, '.eps']), 'ContentType', 'vector');
    fprintf('Saved figure: %s\n', fullfile(saveDir, plotBaseNorm));
  end

  if area_has_prg_djs_plot_data(plotData, sessionTypes, areaIdx)
    figDjs = figure(6100 + a);
    clf(figDjs);
    position_figure_full_monitor(figDjs);
    axDjs = axes(figDjs);
    plot_djs_with_shuffle(axDjs, plotData, sessionTypes, areaIdx, typeColors, shuffleBarColor);
    ylabel(axDjs, djsYLabel);
    title(axDjs, sprintf('%s — %s vs Gaussian', areaName, djsTitleWord));

    if ~isempty(brainArea)
      djsTitleStr = format_prg_sgtitle(prgMethod, djsTitleWord, brainArea, ...
        collectStart, collectEnd, prgWindow);
    else
      djsTitleStr = format_prg_sgtitle(prgMethod, djsTitleWord, areaName, ...
        collectStart, collectEnd, prgWindow);
    end
    if ~isempty(cellType)
      djsTitleStr = sprintf('%s | %s', djsTitleStr, cell_type_label(cellType));
    end
    sgtitle(figDjs, djsTitleStr, 'FontWeight', 'bold');

    plotBaseDjs = make_prg_plot_basename('criticality_prg_djs_across_tasks', areaName, brainArea, ...
      prgWindow, collectStart, collectEnd, length(areasToPlot) > 1, finalCutoffDivisor, ...
      prgMethod, surrogateMethod);
    plotBaseDjs = [plotBaseDjs, cellTag];
    exportgraphics(figDjs, fullfile(saveDir, [plotBaseDjs, '.png']), 'Resolution', 300);
    exportgraphics(figDjs, fullfile(saveDir, [plotBaseDjs, '.eps']), 'ContentType', 'vector');
    fprintf('Saved figure: %s\n', fullfile(saveDir, plotBaseDjs));
  end

  if area_has_prg_metric_plot_data(plotData, sessionTypes, areaIdx, 'djsNormMean')
    djsNormYLabel = sprintf('D_{JS} / surrogate (N/%d vs Gaussian, mean \\pm SEM across windows)', finalCutoffDivisor);
    djsNormTitleWord = sprintf('D_{JS} / surrogate (N/%d)', finalCutoffDivisor);

    figDjsNorm = figure(6300 + a);
    clf(figDjsNorm);
    position_figure_full_monitor(figDjsNorm);
    axDjsNorm = axes(figDjsNorm);
    plot_prg_normalized_sessions(axDjsNorm, plotData, sessionTypes, areaIdx, ...
      typeColors, 'djsNormMean', 'djsNormSem', 'session D_{JS} / surrogate');
    ylabel(axDjsNorm, djsNormYLabel);
    title(axDjsNorm, sprintf('%s — %s vs Gaussian', areaName, djsNormTitleWord));
    yline(axDjsNorm, 1, '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');

    if ~isempty(brainArea)
      djsNormTitleStr = format_prg_sgtitle(prgMethod, djsNormTitleWord, brainArea, ...
        collectStart, collectEnd, prgWindow);
    else
      djsNormTitleStr = format_prg_sgtitle(prgMethod, djsNormTitleWord, areaName, ...
        collectStart, collectEnd, prgWindow);
    end
    if ~isempty(cellType)
      djsNormTitleStr = sprintf('%s | %s', djsNormTitleStr, cell_type_label(cellType));
    end
    sgtitle(figDjsNorm, djsNormTitleStr, 'FontWeight', 'bold');

    plotBaseDjsNorm = make_prg_plot_basename('criticality_prg_djs_norm_across_tasks', areaName, brainArea, ...
      prgWindow, collectStart, collectEnd, length(areasToPlot) > 1, finalCutoffDivisor, ...
      prgMethod, surrogateMethod);
    plotBaseDjsNorm = [plotBaseDjsNorm, cellTag];
    exportgraphics(figDjsNorm, fullfile(saveDir, [plotBaseDjsNorm, '.png']), 'Resolution', 300);
    exportgraphics(figDjsNorm, fullfile(saveDir, [plotBaseDjsNorm, '.eps']), 'ContentType', 'vector');
    fprintf('Saved figure: %s\n', fullfile(saveDir, plotBaseDjsNorm));
  end
end

fprintf('\nAll figures saved to %s\n', saveDir);
end

function plot_kappa_with_shuffle(ax, plotData, sessionTypes, areaIdx, typeColors, shuffleBarColor)
% PLOT_KAPPA_WITH_SHUFFLE - Session mean kappa with SEM; surrogate mean with SEM beside each session

hold(ax, 'on');
xCursor = 0;
xticksCenters = [];
xtickLabels = {};
legendShown = false;

for t = 1:length(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if ~isfield(typeData, 'kappaMean') || areaIdx > length(typeData.kappaMean)
    continue;
  end

  kappaMeans = typeData.kappaMean{areaIdx};
  kappaSems = typeData.kappaSem{areaIdx};
  shuffleMeans = typeData.kappaShuffleMean{areaIdx};
  shuffleSems = typeData.kappaShuffleSem{areaIdx};
  kappaMeans = kappaMeans(:)';
  kappaSems = kappaSems(:)';
  shuffleMeans = shuffleMeans(:)';
  shuffleSems = shuffleSems(:)';
  if numel(shuffleSems) ~= numel(shuffleMeans)
    shuffleSems = nan(size(shuffleMeans));
  end
  numBars = numel(kappaMeans);
  if numBars == 0
    continue;
  end

  xPos = xCursor + (1:numBars);
  errorbar(ax, xPos, kappaMeans, kappaSems, 'o', 'Color', typeColors(t, :), ...
    'MarkerFaceColor', typeColors(t, :), 'MarkerSize', 20, 'LineWidth', 1.2, ...
    'CapSize', 8, 'DisplayName', 'session \kappa');

  if any(isfinite(shuffleMeans))
    xShuffle = xPos + 0.22;
    semPlot = shuffleSems(:)';
    semPlot(~isfinite(semPlot)) = 0;
    errorbar(ax, xShuffle, shuffleMeans, semPlot, 's', ...
      'Color', shuffleBarColor, 'MarkerFaceColor', shuffleBarColor, ...
      'MarkerEdgeColor', [0.2, 0.2, 0.2], 'MarkerSize', 20, 'LineWidth', 1.2, ...
      'CapSize', 8, 'DisplayName', 'surrogate mean \pm SEM (across windows)');
  end

  barLabels = get_session_bar_labels(typeData, numBars, sessionType);
  xticksCenters = [xticksCenters, xPos]; %#ok<AGROW>
  xtickLabels = [xtickLabels, barLabels]; %#ok<AGROW>

  validMeans = kappaMeans(isfinite(kappaMeans));
  if ~isempty(validMeans)
    yline(ax, mean(validMeans), '--', 'Color', typeColors(t, :), 'LineWidth', 1.5, ...
      'HandleVisibility', 'off');
  end

  xCursor = xPos(end) + 1.5;
  legendShown = true;
end

if ~isempty(xticksCenters)
  xticks(ax, xticksCenters);
  xticklabels(ax, xtickLabels);
  xtickangle(ax, 45);
end
grid(ax, 'on');
if legendShown
  legend(ax, {'session \kappa', 'surrogate mean \pm SEM (across windows)'}, 'Location', 'best');
end
hold(ax, 'off');
end

function plot_prg_normalized_sessions(ax, plotData, sessionTypes, areaIdx, typeColors, meanField, semField, legendLabel)
% PLOT_PRG_NORMALIZED_SESSIONS - Session mean normalized metric with SEM by session type
%
% Variables:
%   meanField, semField - Fields in plotData.byType (e.g. kappaNormMean, kappaNormSem)
%   legendLabel         - Display name for errorbar series

hold(ax, 'on');
xCursor = 0;
xticksCenters = [];
xtickLabels = {};
legendShown = false;

for t = 1:length(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if ~isfield(typeData, meanField) || areaIdx > length(typeData.(meanField))
    continue;
  end

  metricMeans = typeData.(meanField){areaIdx};
  metricSems = typeData.(semField){areaIdx};
  metricMeans = metricMeans(:)';
  metricSems = metricSems(:)';
  if numel(metricSems) ~= numel(metricMeans)
    metricSems = nan(size(metricMeans));
  end
  numBars = numel(metricMeans);
  if numBars == 0
    continue;
  end

  xPos = xCursor + (1:numBars);
  errorbar(ax, xPos, metricMeans, metricSems, 'o', 'Color', typeColors(t, :), ...
    'MarkerFaceColor', typeColors(t, :), 'MarkerSize', 20, 'LineWidth', 1.2, ...
    'CapSize', 8, 'DisplayName', legendLabel);

  barLabels = get_session_bar_labels(typeData, numBars, sessionType);
  xticksCenters = [xticksCenters, xPos]; %#ok<AGROW>
  xtickLabels = [xtickLabels, barLabels]; %#ok<AGROW>

  validMeans = metricMeans(isfinite(metricMeans));
  if ~isempty(validMeans)
    yline(ax, mean(validMeans), '--', 'Color', typeColors(t, :), 'LineWidth', 1.5, ...
      'HandleVisibility', 'off');
  end

  xCursor = xPos(end) + 1.5;
  legendShown = true;
end

if ~isempty(xticksCenters)
  xticks(ax, xticksCenters);
  xticklabels(ax, xtickLabels);
  xtickangle(ax, 45);
end
grid(ax, 'on');
if legendShown
  legend(ax, legendLabel, 'Location', 'best');
end
hold(ax, 'off');
end

function plot_djs_with_shuffle(ax, plotData, sessionTypes, areaIdx, typeColors, shuffleBarColor)
% PLOT_DJS_WITH_SHUFFLE - Session mean D_JS with SEM; surrogate mean with SEM beside each session

hold(ax, 'on');
xCursor = 0;
xticksCenters = [];
xtickLabels = {};
legendShown = false;

for t = 1:length(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if ~isfield(typeData, 'djsMean') || areaIdx > length(typeData.djsMean)
    continue;
  end

  djsMeans = typeData.djsMean{areaIdx};
  djsSems = typeData.djsSem{areaIdx};
  shuffleMeans = typeData.djsShuffleMean{areaIdx};
  shuffleSems = typeData.djsShuffleSem{areaIdx};
  djsMeans = djsMeans(:)';
  djsSems = djsSems(:)';
  shuffleMeans = shuffleMeans(:)';
  shuffleSems = shuffleSems(:)';
  if numel(shuffleSems) ~= numel(shuffleMeans)
    shuffleSems = nan(size(shuffleMeans));
  end
  numBars = numel(djsMeans);
  if numBars == 0
    continue;
  end

  xPos = xCursor + (1:numBars);
  errorbar(ax, xPos, djsMeans, djsSems, 'o', 'Color', typeColors(t, :), ...
    'MarkerFaceColor', typeColors(t, :), 'MarkerSize', 20, 'LineWidth', 1.2, ...
    'CapSize', 8, 'DisplayName', 'session D_{JS}');

  if any(isfinite(shuffleMeans))
    xShuffle = xPos + 0.22;
    semPlot = shuffleSems(:)';
    semPlot(~isfinite(semPlot)) = 0;
    errorbar(ax, xShuffle, shuffleMeans, semPlot, 's', ...
      'Color', shuffleBarColor, 'MarkerFaceColor', shuffleBarColor, ...
      'MarkerEdgeColor', [0.2, 0.2, 0.2], 'MarkerSize', 20, 'LineWidth', 1.2, ...
      'CapSize', 8, 'DisplayName', 'surrogate mean \pm SEM (across windows)');
  end

  barLabels = get_session_bar_labels(typeData, numBars, sessionType);
  xticksCenters = [xticksCenters, xPos]; %#ok<AGROW>
  xtickLabels = [xtickLabels, barLabels]; %#ok<AGROW>

  validMeans = djsMeans(isfinite(djsMeans));
  if ~isempty(validMeans)
    yline(ax, mean(validMeans), '--', 'Color', typeColors(t, :), 'LineWidth', 1.5, ...
      'HandleVisibility', 'off');
  end

  xCursor = xPos(end) + 1.5;
  legendShown = true;
end

if ~isempty(xticksCenters)
  xticks(ax, xticksCenters);
  xticklabels(ax, xtickLabels);
  xtickangle(ax, 45);
end
grid(ax, 'on');
if legendShown
  legend(ax, {'session D_{JS}', 'surrogate mean \pm SEM (across windows)'}, 'Location', 'best');
end
hold(ax, 'off');
end

function barLabels = get_session_bar_labels(typeData, numBars, sessionType)
% GET_SESSION_BAR_LABELS - One x-axis label per session bar (sessionName)
%
% Variables:
%   typeData    - Aggregated metrics for one session type
%   numBars     - Number of bars in the current group
%   sessionType - Fallback label if sessionNames unavailable
%
% Returns:
%   barLabels - 1 x numBars cell of char labels

if isfield(typeData, 'sessionNames') && numel(typeData.sessionNames) >= numBars
  barLabels = typeData.sessionNames(1:numBars);
elseif isfield(typeData, 'sessionLabels') && numel(typeData.sessionLabels) >= numBars
  barLabels = typeData.sessionLabels(1:numBars);
else
  barLabels = repmat({sessionType}, 1, numBars);
end
barLabels = cellfun(@char, barLabels, 'UniformOutput', false);
end

function plotBase = make_prg_plot_basename(prefix, areaName, brainArea, prgWindow, collectStart, collectEnd, multiArea, finalCutoffDivisor, prgMethod, surrogateMethod)
% MAKE_PRG_PLOT_BASENAME - Filename stem for saved figures

if nargin < 8 || isempty(finalCutoffDivisor)
  finalCutoffDivisor = 16;
end
if nargin < 9 || isempty(prgMethod)
  prgMethod = 'pca';
end
if nargin < 10 || isempty(surrogateMethod)
  surrogateMethod = 'isi';
end

if isempty(prgWindow)
  winTag = 'full';
else
  winTag = sprintf('%.0fs', prgWindow);
end
if isempty(collectEnd)
  collectTag = sprintf('%.0f-full', collectStart);
else
  collectTag = sprintf('%.0f-%.0fs', collectStart, collectEnd);
end

if ~isempty(brainArea)
  plotBase = sprintf('%s_%s_%s_win%s_%s_N%d_%s', prefix, prgMethod, brainArea, ...
    winTag, collectTag, finalCutoffDivisor, surrogateMethod);
else
  plotBase = sprintf('%s_%s_%s_win%s_%s_N%d_%s', prefix, prgMethod, areaName, ...
    winTag, collectTag, finalCutoffDivisor, surrogateMethod);
end
if multiArea
  plotBase = sprintf('%s_area%s', plotBase, areaName);
end
end

function titleStr = format_prg_sgtitle(prgMethod, metricWord, areaLabel, collectStart, collectEnd, prgWindow)
% FORMAT_PRG_SGTITLE - Super-title with collect range and block length tags

if isempty(collectEnd)
  collectTag = sprintf('%.0f–full s', collectStart);
else
  collectTag = sprintf('%.0f–%.0f s', collectStart, collectEnd);
end
if isempty(prgWindow)
  winTag = 'full-session blocks';
else
  winTag = sprintf('%.0fs blocks', prgWindow);
end
titleStr = sprintf('PRG (%s) %s — %s [%s, %s]', ...
  prgMethod, metricWord, areaLabel, collectTag, winTag);
end

function hasData = area_has_prg_djs_plot_data(plotData, sessionTypes, areaIdx)
% AREA_HAS_PRG_DJS_PLOT_DATA - True if any session type has D_JS values for this area

hasData = false;
for t = 1:length(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if isfield(typeData, 'djsMean') && areaIdx <= length(typeData.djsMean) ...
      && ~isempty(typeData.djsMean{areaIdx})
    hasData = true;
    return;
  end
end
end

function hasData = area_has_prg_metric_plot_data(plotData, sessionTypes, areaIdx, metricField)
% AREA_HAS_PRG_METRIC_PLOT_DATA - True if any session type has values for metricField

hasData = false;
for t = 1:length(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if isfield(typeData, metricField) && areaIdx <= length(typeData.(metricField)) ...
      && ~isempty(typeData.(metricField){areaIdx})
    hasData = true;
    return;
  end
end
end

function hasData = area_has_prg_plot_data(plotData, sessionTypes, areaIdx)
% AREA_HAS_PRG_PLOT_DATA - True if any session type has kappa values for this area

hasData = area_has_prg_metric_plot_data(plotData, sessionTypes, areaIdx, 'kappaMean');
end

function position_figure_full_monitor(fig)
% POSITION_FIGURE_FULL_MONITOR - Size figure to fill a monitor

monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) >= 2
  targetPos = monitorPositions(end, :);
  monitorLabel = 'second';
else
  targetPos = monitorPositions(1, :);
  monitorLabel = 'primary';
end

set(fig, 'Units', 'pixels', 'Position', targetPos);
fprintf('Figure positioned on %s monitor [%d %d %d %d]\n', monitorLabel, targetPos);
end
