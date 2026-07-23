%%
% Criticality AR (d2) Analysis Across Task Types (Manuscript)
%
% Runs d2 criticality in non-overlapping windows of length d2Window seconds,
% batches across session types, and plots session summaries grouped by sessionType.
%
% Variables (configure in this section):
%   sessionTypes   - Cell array of session types to include
%   dataSource     - 'spikes' or 'lfp'
%   collectStart   - Analysis window start (seconds from session onset)
%   collectEnd     - Analysis window end (seconds); [] = full session
%   d2Window       - Non-overlapping window length (seconds); stepSize = d2Window.
%                    [] = one window over the loaded collect duration
%   brainArea              - Single or merged area (e.g. 'M56', 'M23M56'); '' = all areas
%   brainAreaCombinations  - Merged areas: struct('name', 'M23M56', 'areas', {{'M23','M56'}})
%   areasToPlot            - Area names to plot; {} uses brainArea if set
%   runBatch       - If true, run criticality_ar_analysis per session
%   plotResults    - If true, create summary figures after batch
%   useLog10D2     - If true, aggregate and plot log10(d2); values <= 0 become NaN (default true)
%   useSubsampling - If true, d2 per window = mean across neuron subsamples; shuffled
%                    summary = mean across subsamples of per-subsample shuffle means
%   nSubsamples, nNeuronsSubsample, minNeuronsMultiple - subsampling settings (see run_criticality_ar.m)
%   nMinNeurons    - Minimum neurons required per area for criticality_ar_analysis
%   splitExcitatoryInhibitory - If true, run combined (E+I), excitatory, and inhibitory;
%                               plots only the E/I overlay summary (same axes) with
%                               combined (o, solid mean), excitatory (open o, --),
%                               inhibitory (x, dotted mean), colored by colors_for_tasks
%   widthCutoff        - Peak-to-trough width threshold in ms (narrow <= cutoff = I)
%
% Goal:
%   Compare d2 (raw + shuffled) and normalized d2 across spontaneous, reach,
%   interval, and other session types. Per session: mean and SEM of window d2 values;
%   shuffled: mean across windows of (mean shuffle d2 per window), with SEM across
%   windows of those per-window shuffle means.

function out = criticality_ar_across_tasks(opts)
% CRITICALITY_AR_ACROSS_TASKS - Batch and plot d2 across session types (manuscript)
%
% Usage:
%   opts = criticality_ar_across_tasks();                  % default opts
%   out = criticality_ar_across_tasks(opts);
%   out = criticality_ar_across_tasks(struct('plotResults', false));
%
% Options (see fill_criticality_ar_across_tasks_opts):
%   sessionTypes, dataSource, collectStart, collectEnd, d2Window, brainArea, ...
%   runBatch, plotResults, saveBatchResults, batchResultsFile, useLog10D2, ...
%
% Returns:
%   out.batchResults, out.plotData, out.batchMeta, out.paths
%   (or default opts struct when called with no arguments)

if nargin == 0
  out = fill_criticality_ar_across_tasks_opts(struct());
  return;
end

if nargin < 1 || isempty(opts)
  opts = struct();
end
opts = fill_criticality_ar_across_tasks_opts(opts);
setup_criticality_manuscript_paths('criticality_ar_across_tasks');
paths = get_paths();

if isempty(opts.batchResultsFile)
  batchName = 'criticality_ar_across_tasks_batch.mat';
  if opts.splitExcitatoryInhibitory
    batchName = 'criticality_ar_across_tasks_batch_ei_split.mat';
  end
  opts.batchResultsFile = fullfile(paths.dropPath, 'criticality_manuscript', batchName);
end

fprintf('\n=== Criticality d2 Across Task Types ===\n');
fprintf('Collect window: [%.1f, %s] s\n', opts.collectStart, format_collect_end_label(opts.collectEnd));
if isempty(opts.d2Window)
  fprintf('d2 windows: full collect duration (one window per session)\n');
else
  fprintf('d2 windows: %.1f s, non-overlapping (step = window)\n', opts.d2Window);
end
fprintf('useLog10D2 (aggregate/plot): %d\n', opts.useLog10D2);
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
  batchByCell = run_ar_across_tasks_batch(sessionTable, opts, paths);
  batchMeta = pack_ar_across_tasks_batch_meta(opts);
  plotDataByCell = cell(1, numel(cellTypesToRun));
  for iCell = 1:numel(cellTypesToRun)
    plotDataByCell{iCell} = aggregate_ar_metrics(batchByCell{iCell}, opts.sessionTypes, opts.useLog10D2);
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
    error('criticality_ar_across_tasks:NoBatchFile', ...
      'Batch file not found: %s. Set runBatch true to compute.', opts.batchResultsFile);
  end
  loaded = load(opts.batchResultsFile);
  batchMeta = loaded.batchMeta;
  useLog10D2Load = opts.useLog10D2;
  if isfield(batchMeta, 'useLog10D2')
    useLog10D2Load = batchMeta.useLog10D2;
  end
  if isfield(loaded, 'batchByCell')
    batchByCell = loaded.batchByCell;
    cellTypesToRun = loaded.cellTypesToRun;
  else
    batchByCell = {loaded.batchResults};
    cellTypesToRun = {''};
  end
  % Prefer saved plotData (matches AV/PRG load); re-aggregate only if missing
  if isfield(loaded, 'plotDataByCell') && ~isempty(loaded.plotDataByCell)
    plotDataByCell = loaded.plotDataByCell;
  elseif isfield(loaded, 'plotData') && ~isempty(loaded.plotData)
    plotDataByCell = {loaded.plotData};
  else
    plotDataByCell = cell(1, numel(batchByCell));
    for iCell = 1:numel(batchByCell)
      plotDataByCell{iCell} = aggregate_ar_metrics(batchByCell{iCell}, opts.sessionTypes, useLog10D2Load);
    end
  end
  fprintf('\nLoaded batch results: %s\n', opts.batchResultsFile);
end

batchResults = batchByCell{1};
plotData = plotDataByCell{1};
areasToPlot = opts.areasToPlot;
if isempty(areasToPlot) && ~isempty(opts.brainArea)
  areasToPlot = {opts.brainArea};
end

% When E/I split is on, keep only sessions with both excitatory and inhibitory results
if opts.splitExcitatoryInhibitory
  plotDataByCell = filter_ar_plotdata_to_ei_complete_sessions( ...
    plotDataByCell, cellTypesToRun, opts.sessionTypes);
  plotData = plotDataByCell{1};
  batchResults = batchByCell{1};
end

anyAreas = false;
commonAreasUnion = {};
for iCell = 1:numel(cellTypesToRun)
  cellType = cellTypesToRun{iCell};
  plotDataCell = plotDataByCell{iCell};
  if isempty(plotDataCell.areas)
    warning('criticality_ar_across_tasks:NoMetrics', ...
      'No d2 metrics for %s. Skipping.', cell_type_label(cellType));
    continue;
  end
  anyAreas = true;
  if opts.plotResults
    commonAreas = resolve_areas_to_plot(plotDataCell.areas, areasToPlot, opts.brainArea);
    commonAreasUnion = unique([commonAreasUnion, commonAreas], 'stable');

    fprintf('\n=== Areas for plotting (%s) ===\n', cell_type_label(cellType));
    fprintf('  %s\n', strjoin(commonAreas, ', '));

    % With E/I split, skip per-population figures; only the summary overlay is plotted
    if ~opts.splitExcitatoryInhibitory
      plot_ar_across_tasks(plotDataCell, commonAreas, opts.sessionTypes, opts.collectStart, ...
        opts.collectEnd, opts.d2Window, paths, opts.brainArea, opts.useLog10D2, cellType, ...
        opts.enablePermutations);
    end
  end
end
if ~anyAreas
  error('No d2 metrics extracted. Check that batch analyses succeeded.');
end

if opts.plotResults && opts.splitExcitatoryInhibitory
  fprintf('\n=== E/I summary plots (combined / excitatory / inhibitory) ===\n');
  plot_ar_ei_summary_across_tasks(plotDataByCell, cellTypesToRun, commonAreasUnion, ...
    opts.sessionTypes, opts.collectStart, opts.collectEnd, opts.d2Window, paths, ...
    opts.brainArea, opts.useLog10D2, opts.enablePermutations);
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
out.areasToPlot = areasToPlot;
end

function opts = fill_criticality_ar_across_tasks_opts(opts)
% FILL_CRITICALITY_AR_ACROSS_TASKS_OPTS - Defaults for criticality_ar_across_tasks

defaults = struct();
defaults.sessionTypes = {'spontaneous', 'interval', 'reach'};
defaults.dataSource = 'spikes';
defaults.collectStart = 0;
defaults.collectEnd = [];
defaults.d2Window = 30;
defaults.brainArea = 'M23M56';
defaults.brainAreaCombinations = default_manuscript_brain_area_combinations();
defaults.areasToPlot = {};
defaults.runBatch = true;
defaults.plotResults = true;
defaults.saveBatchResults = false;
defaults.batchResultsFile = '';
defaults.useLog10D2 = true;
defaults.useSubsampling = false;
defaults.nSubsamples = 20;
defaults.nNeuronsSubsample = 20;
defaults.minNeuronsMultiple = 1.5;
defaults.nMinNeurons = 30;
defaults.firingRateCheckTime = [];
defaults.minFiringRate = 0.05;
defaults.maxFiringRate = 150;
defaults.enablePermutations = true;
defaults.nShuffles = 10;
defaults.splitExcitatoryInhibitory = false;
defaults.widthCutoff = 0.35;
% Empty collectEnd / d2Window are sentinels for "full session" — do not replace
preserveCollectEndEmpty = isfield(opts, 'collectEnd') && isempty(opts.collectEnd);
preserveD2WindowEmpty = isfield(opts, 'd2Window') && isempty(opts.d2Window);
opts = merge_struct_defaults(opts, defaults);
if preserveCollectEndEmpty
  opts.collectEnd = [];
end
if preserveD2WindowEmpty
  opts.d2Window = [];
end
end

function batchMeta = pack_ar_across_tasks_batch_meta(opts)
batchMeta = struct( ...
  'sessionTypes', {opts.sessionTypes}, ...
  'useLog10D2', opts.useLog10D2, ...
  'collectStart', opts.collectStart, ...
  'collectEnd', opts.collectEnd, ...
  'd2Window', opts.d2Window, ...
  'brainArea', opts.brainArea, ...
  'areasToPlot', {opts.areasToPlot}, ...
  'nMinNeurons', opts.nMinNeurons, ...
  'splitExcitatoryInhibitory', opts.splitExcitatoryInhibitory, ...
  'widthCutoff', opts.widthCutoff);
end

function batchByCell = run_ar_across_tasks_batch(sessionTable, opts, paths)
% RUN_AR_ACROSS_TASKS_BATCH - Per-session d2 analysis (optional E/I split)

analysisConfig = build_ar_analysis_config(opts);
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

if isempty(opts.d2Window)
  fprintf('\n=== Running d2 analysis (one window over full collect duration) ===\n');
else
  fprintf('\n=== Running d2 analysis (%.0f s non-overlapping windows) ===\n', opts.d2Window);
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
    useFullSessionWindow = isempty(opts.d2Window) ...
      || sessionDuration < (opts.d2Window - durationToleranceSec);
    if useFullSessionWindow
      if isempty(opts.d2Window)
        fprintf('  Using full session window (%.1f s).\n', sessionDuration);
      else
        fprintf('  Session duration %.1f s < requested d2Window %.1f s; using full session window.\n', ...
          sessionDuration, opts.d2Window);
      end
      sessionConfig.slidingWindowSize = sessionDuration;
      sessionConfig.stepSize = sessionDuration;
    end

    for iCell = 1:numel(cellTypesToRun)
      cellType = cellTypesToRun{iCell};
      try
        dataStructRun = prepare_session_data_for_cell_type(dataStruct, paths, cellType, ...
          opts.widthCutoff, opts.splitExcitatoryInhibitory);
        [dataStructRun, ~] = apply_manuscript_brain_area_selection( ...
          dataStructRun, opts.brainArea, opts.brainAreaCombinations);

        arResults = criticality_ar_analysis(dataStructRun, sessionConfig);
        if ~isempty(opts.brainArea)
          arResults = filter_ar_results_to_brain_area(arResults, opts.brainArea);
          if isempty(arResults.areas)
            fprintf('  No results for brain area "%s" (%s); skipping.\n', ...
              opts.brainArea, cell_type_label(cellType));
            continue;
          end
        end

        batchByCell{iCell}(s).success = true;
        batchByCell{iCell}(s).results = arResults;
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

    % E/I split requires both populations; drop the whole session if either fails
    if opts.splitExcitatoryInhibitory
      batchByCell = invalidate_ar_session_if_ei_incomplete(batchByCell, cellTypesToRun, s);
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
    error('criticality_ar_across_tasks:SessionFailed', ...
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

function batchByCell = invalidate_ar_session_if_ei_incomplete(batchByCell, cellTypesToRun, sessionIdx)
% INVALIDATE_AR_SESSION_IF_EI_INCOMPLETE - Drop session unless E and I both succeeded

eIdx = find(strcmpi(cellTypesToRun, 'excitatory'), 1);
iIdx = find(strcmpi(cellTypesToRun, 'inhibitory'), 1);
if isempty(eIdx) || isempty(iIdx)
  return;
end
eOk = batchByCell{eIdx}(sessionIdx).success;
iOk = batchByCell{iIdx}(sessionIdx).success;
if eOk && iOk
  return;
end

fprintf(['  Incomplete E/I for this session (excitatory=%d, inhibitory=%d); ', ...
  'skipping all populations.\n'], eOk, iOk);
for iCell = 1:numel(cellTypesToRun)
  batchByCell{iCell}(sessionIdx).success = false;
  batchByCell{iCell}(sessionIdx).results = [];
  batchByCell{iCell}(sessionIdx).skipReason = 'Incomplete excitatory/inhibitory results';
end
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
    error('criticality_ar_across_tasks:UnknownSessionDuration', ...
      'Could not determine session collect duration.');
  end
end
end

function analysisConfig = build_ar_analysis_config(opts)
analysisConfig = struct();
if isempty(opts.d2Window)
  % Placeholder; per-session loop replaces with actual collect duration
  analysisConfig.slidingWindowSize = 1;
  analysisConfig.stepSize = 1;
else
  analysisConfig.slidingWindowSize = opts.d2Window;
  analysisConfig.stepSize = opts.d2Window;
end
analysisConfig.binSize = 0.05;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.analyzeD2 = true;
analysisConfig.analyzeMrBr = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 4;
analysisConfig.enablePermutations = opts.enablePermutations;
analysisConfig.nShuffles = opts.nShuffles;
analysisConfig.normalizeD2 = opts.enablePermutations;
analysisConfig.useLog10D2 = opts.useLog10D2;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.pOrder = 10;
analysisConfig.critType = 2;
analysisConfig.minSpikesPerBin = 2.5;
analysisConfig.minBinsPerWindow = 1000;
analysisConfig.maxSpikesPerBin = 100;
analysisConfig.nMinNeurons = opts.nMinNeurons;
analysisConfig.useSubsampling = opts.useSubsampling;
analysisConfig.nSubsamples = opts.nSubsamples;
analysisConfig.nNeuronsSubsample = opts.nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = opts.minNeuronsMultiple;
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
if isfield(results, 'nNeurons') && numel(results.nNeurons) >= areaIdx
  results.nNeurons = results.nNeurons(areaIdx);
end
end

function plotData = aggregate_ar_metrics(batchResults, sessionTypes, useLog10D2)
% AGGREGATE_AR_METRICS - Per-session mean and SEM of window d2 and shuffle summary
%
% Variables:
%   useLog10D2 - If true, apply log10 to d2, shuffles, and normalized d2 before stats

if nargin < 3 || isempty(useLog10D2)
  useLog10D2 = false;
end

plotData = struct();
plotData.areas = {};
plotData.sessionTypes = sessionTypes;
plotData.byType = struct();
plotData.useLog10D2 = useLog10D2;

metricFields = {'d2Mean', 'd2Sem', 'd2ShuffleMean', 'd2ShuffleSem', 'd2NormMean', 'd2NormSem', ...
  'meanSpikesPerBinPerNeuron'};

for s = 1:length(batchResults)
  if ~batchResults(s).success || isempty(batchResults(s).results)
    continue;
  end

  results = batchResults(s).results;
  sessionType = batchResults(s).sessionType;

  if isempty(plotData.areas)
    for t = 1:length(sessionTypes)
      st = sessionTypes{t};
      plotData.byType.(matlab.lang.makeValidName(st)) = init_type_ar_metrics(metricFields, 0);
    end
  end

  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    plotData.byType.(typeKey) = init_type_ar_metrics(metricFields, length(plotData.areas));
  end
  typeData = plotData.byType.(typeKey);

  for a = 1:length(results.areas)
    areaName = results.areas{a};
    areaIdx = find(strcmp(plotData.areas, areaName), 1);
    if isempty(areaIdx)
      plotData.areas{end+1} = areaName;
      areaIdx = length(plotData.areas);
      plotData = extend_ar_plot_data_areas(plotData, sessionTypes, metricFields, areaIdx);
      typeData = plotData.byType.(typeKey);
    end

    summary = summarize_session_d2_windows(results, a, useLog10D2);
    for m = 1:length(metricFields)
      fieldName = metricFields{m};
      if strcmp(fieldName, 'meanSpikesPerBinPerNeuron')
        typeData.(fieldName){areaIdx} = [typeData.(fieldName){areaIdx}, ...
          summarize_mean_spikes_per_bin_per_neuron(results, a)];
      else
        typeData.(fieldName){areaIdx} = [typeData.(fieldName){areaIdx}, summary.(fieldName)];
      end
    end
  end

  typeData.sessionLabels{end+1} = batchResults(s).label;
  typeData.sessionNames{end+1} = batchResults(s).sessionName;
  plotData.byType.(typeKey) = typeData;
end
end

function rateVal = summarize_mean_spikes_per_bin_per_neuron(results, areaIdx)
% SUMMARIZE_MEAN_SPIKES_PER_BIN_PER_NEURON - Session mean of pop spikes/bin / nNeurons
%
% Variables:
%   results  - Output from criticality_ar_analysis (popActivity = sum across neurons)
%   areaIdx  - Index into results.areas
%
% Goal:
%   One scalar balancing sessions with different neuron counts.

rateVal = nan;
if ~isfield(results, 'popActivityWindows') || areaIdx > numel(results.popActivityWindows) ...
    || isempty(results.popActivityWindows{areaIdx})
  return;
end
popWin = results.popActivityWindows{areaIdx}(:);
popWin = popWin(isfinite(popWin));
if isempty(popWin)
  return;
end

nNeurons = nan;
if isfield(results, 'nNeurons') && numel(results.nNeurons) >= areaIdx
  nNeurons = results.nNeurons(areaIdx);
end
if ~(isfinite(nNeurons) && nNeurons > 0)
  return;
end
rateVal = mean(popWin) / nNeurons;
end

function summary = summarize_session_d2_windows(results, areaIdx, useLog10D2)
% SUMMARIZE_SESSION_D2_WINDOWS - Mean and SEM across windows; shuffle = mean of window means
%
% Variables:
%   results      - Output from criticality_ar_analysis
%   areaIdx      - Index into results.areas
%   useLog10D2   - If true, use log10(d2) etc. (x>0 only; same as criticality_ar_plot)
%
% Returns:
%   summary - Struct with d2Mean, d2Sem, d2ShuffleMean, d2ShuffleSem, d2NormMean, d2NormSem
%
%   d2ShuffleSem - SEM across windows of per-window shuffle summary
%   (with subsampling: mean across subsamples of per-subsample shuffle means)

if nargin < 3 || isempty(useLog10D2)
  useLog10D2 = false;
end

summary = struct('d2Mean', nan, 'd2Sem', nan, 'd2ShuffleMean', nan, ...
  'd2ShuffleSem', nan, 'd2NormMean', nan, 'd2NormSem', nan);

if areaIdx > length(results.d2) || isempty(results.d2{areaIdx})
  return;
end

d2Vec = results.d2{areaIdx}(:);
if useLog10D2
  d2Vec = log10_safe_numeric(d2Vec);
end
d2Vec = d2Vec(isfinite(d2Vec));
if ~isempty(d2Vec)
  summary.d2Mean = mean(d2Vec);
  nD2 = numel(d2Vec);
  if nD2 > 1
    summary.d2Sem = std(d2Vec) / sqrt(nD2);
  else
    summary.d2Sem = 0;
  end
end

if isfield(results, 'd2Normalized') && areaIdx <= length(results.d2Normalized) ...
    && ~isempty(results.d2Normalized{areaIdx})
  d2NormVec = results.d2Normalized{areaIdx}(:);
  if useLog10D2
    d2NormVec = log10_safe_numeric(d2NormVec);
  end
  d2NormVec = d2NormVec(isfinite(d2NormVec));
  if ~isempty(d2NormVec)
    summary.d2NormMean = mean(d2NormVec);
    nNorm = numel(d2NormVec);
    if nNorm > 1
      summary.d2NormSem = std(d2NormVec) / sqrt(nNorm);
    else
      summary.d2NormSem = 0;
    end
  end
end

if isfield(results, 'd2Permuted') && areaIdx <= length(results.d2Permuted) ...
    && ~isempty(results.d2Permuted{areaIdx})
  perWindowShuffleMean = get_per_window_shuffle_mean_d2(results, areaIdx, useLog10D2);
  perWindowShuffleMean = perWindowShuffleMean(isfinite(perWindowShuffleMean));
  if ~isempty(perWindowShuffleMean)
    summary.d2ShuffleMean = mean(perWindowShuffleMean);
    nSh = numel(perWindowShuffleMean);
    if nSh > 1
      summary.d2ShuffleSem = std(perWindowShuffleMean) / sqrt(nSh);
    else
      summary.d2ShuffleSem = 0;
    end
  end
end
end

function y = log10_safe_numeric(x)
% LOG10_SAFE_NUMERIC - log10 with NaN for non-positive (matches criticality_ar_plot)

validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end

function plotData = extend_ar_plot_data_areas(plotData, sessionTypes, metricFields, newAreaIdx)
% EXTEND_AR_PLOT_DATA_AREAS - Grow metric storage when a new area appears

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

function typeData = init_type_ar_metrics(metricFields, numAreas)
% INIT_TYPE_AR_METRICS - Empty storage per area for one session type

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

function plot_ar_across_tasks(plotData, areasToPlot, sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, cellType, enablePermutations)
% PLOT_AR_ACROSS_TASKS - Raw+shuffled d2 and normalized d2 by session type

if nargin < 8 || isempty(brainArea)
  brainArea = '';
end
if nargin < 9 || isempty(useLog10D2)
  if isfield(plotData, 'useLog10D2')
    useLog10D2 = plotData.useLog10D2;
  else
    useLog10D2 = false;
  end
end
if nargin < 10 || isempty(cellType)
  cellType = '';
end
if nargin < 11 || isempty(enablePermutations)
  enablePermutations = true;
end
useLog10D2 = logical(useLog10D2);
enablePermutations = logical(enablePermutations);
cellTag = cell_type_file_tag(cellType);
plotConfig = fill_manuscript_plot_config();
labelInterpreter = 'tex';

if useLog10D2
  rawYLabel = 'log_{10}(d2) (mean \pm SEM across windows)';
  normYLabel = 'log_{10}(d2 normalized) (mean \pm SEM across windows)';
  rawTitleWord = 'log_{10}(d2)';
  normTitleWord = 'log_{10}(d2 normalized)';
else
  rawYLabel = 'd2 (mean \pm SEM across windows)';
  normYLabel = 'd2 normalized (mean \pm SEM across windows)';
  rawTitleWord = 'd2';
  normTitleWord = 'd2 normalized';
  labelInterpreter = 'none';
end

shuffleBarColor = [0.55, 0.55, 0.55];

saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

for a = 1:length(areasToPlot)
  areaName = areasToPlot{a};
  areaIdx = find(strcmp(plotData.areas, areaName), 1);
  if isempty(areaIdx) || ~area_has_ar_plot_data(plotData, sessionTypes, areaIdx)
    continue;
  end

  % Figure 1: raw d2 + shuffled mean
  figRaw = figure(5000 + 2 * a - 1);
  clf(figRaw);
  set(figRaw, 'Color', 'w');
  position_figure_full_monitor(figRaw);
  axRaw = axes(figRaw);
  plot_d2_raw_with_shuffle(axRaw, plotData, sessionTypes, areaIdx, shuffleBarColor, ...
    enablePermutations, plotConfig);
  apply_manuscript_axes_style(axRaw, plotConfig, '', rawYLabel, '', labelInterpreter);
  collectTag = format_collect_window_tag(collectStart, collectEnd);
  if isempty(d2Window)
    winTag = 'full';
  else
    winTag = sprintf('%.0fs', d2Window);
  end
  if ~isempty(brainArea)
    titleStrRaw = sprintf('%s (raw) — %s [%s, %s windows]', ...
      rawTitleWord, brainArea, collectTag, winTag);
  else
    titleStrRaw = sprintf('%s (raw) — %s [%s, %s windows]', ...
      rawTitleWord, areaName, collectTag, winTag);
  end
  if ~isempty(cellType)
    titleStrRaw = sprintf('%s | %s', titleStrRaw, cell_type_label(cellType));
  end
  sgtitle(figRaw, titleStrRaw, 'FontWeight', 'bold', ...
    'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', labelInterpreter);

  plotBaseRaw = make_ar_plot_basename('criticality_ar_across_tasks_raw', areaName, brainArea, ...
    d2Window, collectStart, collectEnd, length(areasToPlot) > 1, useLog10D2);
  plotBaseRaw = [plotBaseRaw, cellTag];
  exportgraphics(figRaw, fullfile(saveDir, [plotBaseRaw, '.png']), 'Resolution', 300);
  exportgraphics(figRaw, fullfile(saveDir, [plotBaseRaw, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBaseRaw));

  if ~enablePermutations
    continue;
  end

  % Figure 2: normalized d2 (requires permutations)
  figNorm = figure(5000 + 2 * a);
  clf(figNorm);
  set(figNorm, 'Color', 'w');
  position_figure_full_monitor(figNorm);
  axNorm = axes(figNorm);
  plot_d2_normalized(axNorm, plotData, sessionTypes, areaIdx, plotConfig);
  apply_manuscript_axes_style(axNorm, plotConfig, '', normYLabel, '', labelInterpreter);
  if ~isempty(brainArea)
    titleStrNorm = sprintf('%s — %s [%s, %s windows]', ...
      normTitleWord, brainArea, collectTag, winTag);
  else
    titleStrNorm = sprintf('%s — %s [%s, %s windows]', ...
      normTitleWord, areaName, collectTag, winTag);
  end
  if ~isempty(cellType)
    titleStrNorm = sprintf('%s | %s', titleStrNorm, cell_type_label(cellType));
  end
  sgtitle(figNorm, titleStrNorm, 'FontWeight', 'bold', ...
    'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', labelInterpreter);

  plotBaseNorm = make_ar_plot_basename('criticality_ar_across_tasks_normalized', areaName, brainArea, ...
    d2Window, collectStart, collectEnd, length(areasToPlot) > 1, useLog10D2);
  plotBaseNorm = [plotBaseNorm, cellTag];
  exportgraphics(figNorm, fullfile(saveDir, [plotBaseNorm, '.png']), 'Resolution', 300);
  exportgraphics(figNorm, fullfile(saveDir, [plotBaseNorm, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBaseNorm));
end

fprintf('\nAll figures saved to %s\n', saveDir);
end

function plot_ar_ei_summary_across_tasks(plotDataByCell, cellTypesToRun, areasToPlot, ...
    sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
    enablePermutations)
% PLOT_AR_EI_SUMMARY_ACROSS_TASKS - Combined / E / I on one axes per area
%
% Variables:
%   plotDataByCell  - Cell of aggregated plotData (one per cell type)
%   cellTypesToRun  - Matching cell-type tags ('all', 'excitatory', 'inhibitory')
%   areasToPlot     - Areas to figure
%   enablePermutations - If false, skip normalized-d2 summary
%
% Goal:
%   Overlay session d2 for combined (o, solid mean), excitatory (open o, --),
%   and inhibitory (x, dotted mean) using task colors from colors_for_tasks.
%   Only sessions with finite E and I (and combined when present) are shown.

if nargin < 11 || isempty(enablePermutations)
  enablePermutations = true;
end
useLog10D2 = logical(useLog10D2);
enablePermutations = logical(enablePermutations);
plotConfig = fill_manuscript_plot_config();
labelInterpreter = 'tex';
if useLog10D2
  rawYLabel = 'log_{10}(d2) (mean \pm SEM across windows)';
  normYLabel = 'log_{10}(d2 normalized) (mean \pm SEM across windows)';
  rawTitleWord = 'log_{10}(d2)';
  normTitleWord = 'log_{10}(d2 normalized)';
else
  rawYLabel = 'd2 (mean \pm SEM across windows)';
  normYLabel = 'd2 normalized (mean \pm SEM across windows)';
  rawTitleWord = 'd2';
  normTitleWord = 'd2 normalized';
  labelInterpreter = 'none';
end

saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

for a = 1:numel(areasToPlot)
  areaName = areasToPlot{a};
  if ~any_ar_ei_area_has_data(plotDataByCell, cellTypesToRun, sessionTypes, areaName)
    continue;
  end

  figRaw = figure(5200 + 2 * a - 1);
  clf(figRaw);
  set(figRaw, 'Color', 'w');
  position_figure_full_monitor(figRaw);
  axRaw = axes(figRaw);
  plot_ar_ei_metric_overlay(axRaw, plotDataByCell, cellTypesToRun, sessionTypes, ...
    areaName, 'd2Mean', 'd2Sem', plotConfig);
  apply_manuscript_axes_style(axRaw, plotConfig, '', rawYLabel, '', labelInterpreter);

  collectTag = format_collect_window_tag(collectStart, collectEnd);
  if isempty(d2Window)
    winTag = 'full';
  else
    winTag = sprintf('%.0fs', d2Window);
  end
  if ~isempty(brainArea)
    titleStrRaw = sprintf('%s E/I summary — %s [%s, %s windows]', ...
      rawTitleWord, brainArea, collectTag, winTag);
  else
    titleStrRaw = sprintf('%s E/I summary — %s [%s, %s windows]', ...
      rawTitleWord, areaName, collectTag, winTag);
  end
  sgtitle(figRaw, titleStrRaw, 'FontWeight', 'bold', ...
    'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', labelInterpreter);

  plotBaseRaw = make_ar_plot_basename('criticality_ar_across_tasks_ei_summary_raw', ...
    areaName, brainArea, d2Window, collectStart, collectEnd, numel(areasToPlot) > 1, useLog10D2);
  exportgraphics(figRaw, fullfile(saveDir, [plotBaseRaw, '.png']), 'Resolution', 300);
  exportgraphics(figRaw, fullfile(saveDir, [plotBaseRaw, '.eps']), 'ContentType', 'vector');
  fprintf('Saved E/I summary: %s\n', fullfile(saveDir, plotBaseRaw));

  if ~enablePermutations
    continue;
  end

  figNorm = figure(5200 + 2 * a);
  clf(figNorm);
  set(figNorm, 'Color', 'w');
  position_figure_full_monitor(figNorm);
  axNorm = axes(figNorm);
  plot_ar_ei_metric_overlay(axNorm, plotDataByCell, cellTypesToRun, sessionTypes, ...
    areaName, 'd2NormMean', 'd2NormSem', plotConfig);
  apply_manuscript_axes_style(axNorm, plotConfig, '', normYLabel, '', labelInterpreter);
  if ~isempty(brainArea)
    titleStrNorm = sprintf('%s E/I summary — %s [%s, %s windows]', ...
      normTitleWord, brainArea, collectTag, winTag);
  else
    titleStrNorm = sprintf('%s E/I summary — %s [%s, %s windows]', ...
      normTitleWord, areaName, collectTag, winTag);
  end
  sgtitle(figNorm, titleStrNorm, 'FontWeight', 'bold', ...
    'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', labelInterpreter);

  plotBaseNorm = make_ar_plot_basename('criticality_ar_across_tasks_ei_summary_normalized', ...
    areaName, brainArea, d2Window, collectStart, collectEnd, numel(areasToPlot) > 1, useLog10D2);
  exportgraphics(figNorm, fullfile(saveDir, [plotBaseNorm, '.png']), 'Resolution', 300);
  exportgraphics(figNorm, fullfile(saveDir, [plotBaseNorm, '.eps']), 'ContentType', 'vector');
  fprintf('Saved E/I summary: %s\n', fullfile(saveDir, plotBaseNorm));
end
end

function plot_ar_ei_metric_overlay(ax, plotDataByCell, cellTypesToRun, sessionTypes, ...
    areaName, meanField, semField, plotConfig)
% PLOT_AR_EI_METRIC_OVERLAY - Combined / E / I markers + within-task mean lines

if nargin < 8 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end

hold(ax, 'on');
xCursor = 0;
xticksCenters = [];
xtickLabels = {};
legendHandles = gobjects(0);
legendLabels = {};
markerSize = max(8, plotConfig.scatterMarkerSize / 2);

for t = 1:numel(sessionTypes)
  sessionType = sessionTypes{t};
  taskColor = colors_for_tasks(sessionType);
  masterNames = get_ar_ei_complete_session_names(plotDataByCell, cellTypesToRun, ...
    sessionType, areaName, meanField);
  if isempty(masterNames)
    continue;
  end
  numSessions = numel(masterNames);
  xPos = xCursor + (1:numSessions);

  for iCell = 1:numel(cellTypesToRun)
    cellType = cellTypesToRun{iCell};
    [markerSpec, lineStyle, faceColor] = get_ar_ei_marker_style(cellType, taskColor);
    [yVals, ySem] = get_ar_ei_aligned_metric(plotDataByCell{iCell}, ...
      sessionType, areaName, masterNames, meanField, semField);
    if ~any(isfinite(yVals))
      continue;
    end

    semPlot = ySem(:);
    semPlot(~isfinite(semPlot)) = 0;
    hEb = errorbar(ax, xPos, yVals, semPlot, markerSpec, ...
      'Color', taskColor, 'MarkerFaceColor', faceColor, ...
      'MarkerEdgeColor', taskColor, 'MarkerSize', markerSize, ...
      'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize, ...
      'LineStyle', 'none', 'DisplayName', cell_type_label(cellType));
    if ~any(strcmp(legendLabels, cell_type_label(cellType)))
      legendHandles(end + 1) = hEb; %#ok<AGROW>
      legendLabels{end + 1} = cell_type_label(cellType); %#ok<AGROW>
    end

    yMeanTask = mean(yVals(isfinite(yVals)));
    if isfinite(yMeanTask) && numSessions >= 1
      plot(ax, [xPos(1), xPos(end)], [yMeanTask, yMeanTask], lineStyle, ...
        'Color', taskColor, 'LineWidth', plotConfig.lineWidth + 0.3, ...
        'HandleVisibility', 'off');
    end
  end

  for i = 1:numSessions
    xticksCenters(end + 1) = xPos(i); %#ok<AGROW>
    xtickLabels{end + 1} = truncate_session_xtick_label(masterNames{i}, 7); %#ok<AGROW>
  end
  xCursor = xPos(end) + 1.5;
end

if ~isempty(xticksCenters)
  xticks(ax, xticksCenters);
  xticklabels(ax, xtickLabels);
  xtickangle(ax, 45);
  xlim(ax, [min(xticksCenters) - 0.8, max(xticksCenters) + 0.8]);
end
if ~isempty(legendHandles)
  legend(ax, legendHandles, legendLabels, 'Location', 'best', ...
    'FontSize', plotConfig.legendFontSize);
end
hold(ax, 'off');
end

function tf = any_ar_ei_area_has_data(plotDataByCell, cellTypesToRun, sessionTypes, areaName)
tf = false;
for iCell = 1:numel(cellTypesToRun)
  plotData = plotDataByCell{iCell};
  areaIdx = find(strcmp(plotData.areas, areaName), 1);
  if isempty(areaIdx)
    continue;
  end
  if area_has_ar_plot_data(plotData, sessionTypes, areaIdx)
    tf = true;
    return;
  end
end
end

function masterNames = get_ar_ei_complete_session_names(plotDataByCell, cellTypesToRun, ...
    sessionType, areaName, meanField)
% GET_AR_EI_COMPLETE_SESSION_NAMES - Sessions with finite E and I (and combined)
%
% Goal:
%   Only keep sessions that have finite meanField for excitatory and inhibitory
%   in this area. If combined ('all') is present, also require it.

eIdx = find(strcmpi(cellTypesToRun, 'excitatory'), 1);
iIdx = find(strcmpi(cellTypesToRun, 'inhibitory'), 1);
allIdx = find(strcmpi(cellTypesToRun, 'all'), 1);
masterNames = {};
if isempty(eIdx) || isempty(iIdx)
  return;
end

namesE = get_ar_finite_session_names(plotDataByCell{eIdx}, sessionType, areaName, meanField);
namesI = get_ar_finite_session_names(plotDataByCell{iIdx}, sessionType, areaName, meanField);
masterNames = intersect(namesE, namesI, 'stable');
if ~isempty(allIdx)
  namesAll = get_ar_finite_session_names(plotDataByCell{allIdx}, sessionType, areaName, meanField);
  masterNames = intersect(namesAll, masterNames, 'stable');
end
end

function names = get_ar_finite_session_names(plotData, sessionType, areaName, meanField)
% Session names with finite metric values for one area

names = {};
if isempty(plotData) || ~isfield(plotData, 'areas')
  return;
end
areaIdx = find(strcmp(plotData.areas, areaName), 1);
if isempty(areaIdx)
  return;
end
typeKey = matlab.lang.makeValidName(sessionType);
if ~isfield(plotData.byType, typeKey)
  return;
end
typeData = plotData.byType.(typeKey);
if ~isfield(typeData, meanField) || areaIdx > numel(typeData.(meanField)) ...
    || isempty(typeData.(meanField){areaIdx})
  return;
end
vals = typeData.(meanField){areaIdx}(:)';
n = numel(vals);
if isfield(typeData, 'sessionNames') && numel(typeData.sessionNames) >= n
  allNames = cellfun(@char, typeData.sessionNames(1:n), 'UniformOutput', false);
elseif isfield(typeData, 'sessionLabels') && numel(typeData.sessionLabels) >= n
  allNames = cellfun(@char, typeData.sessionLabels(1:n), 'UniformOutput', false);
else
  allNames = arrayfun(@(i) sprintf('%s_%d', sessionType, i), 1:n, 'UniformOutput', false);
end
names = allNames(isfinite(vals));
end

function plotDataByCell = filter_ar_plotdata_to_ei_complete_sessions(plotDataByCell, ...
    cellTypesToRun, sessionTypes)
% FILTER_AR_PLOTDATA_TO_EI_COMPLETE_SESSIONS - Keep sessions present in E and I
%
% Goal:
%   For loaded or partially successful batches, drop sessions missing excitatory
%   or inhibitory results so all populations share the same session list.

eIdx = find(strcmpi(cellTypesToRun, 'excitatory'), 1);
iIdx = find(strcmpi(cellTypesToRun, 'inhibitory'), 1);
if isempty(eIdx) || isempty(iIdx)
  return;
end

for t = 1:numel(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  namesE = get_ar_type_session_names(plotDataByCell{eIdx}, typeKey);
  namesI = get_ar_type_session_names(plotDataByCell{iIdx}, typeKey);
  keepNames = intersect(namesE, namesI, 'stable');
  allIdx = find(strcmpi(cellTypesToRun, 'all'), 1);
  if ~isempty(allIdx)
    namesAll = get_ar_type_session_names(plotDataByCell{allIdx}, typeKey);
    keepNames = intersect(namesAll, keepNames, 'stable');
  end

  nBefore = max([numel(namesE), numel(namesI)]);
  if numel(keepNames) < nBefore
    fprintf(['  %s: keeping %d/%d sessions with both excitatory and inhibitory ', ...
      'results.\n'], sessionType, numel(keepNames), nBefore);
  end

  for iCell = 1:numel(cellTypesToRun)
    plotDataByCell{iCell} = keep_ar_type_sessions(plotDataByCell{iCell}, typeKey, keepNames);
  end
end
end

function names = get_ar_type_session_names(plotData, typeKey)
names = {};
if isempty(plotData) || ~isfield(plotData, 'byType') || ~isfield(plotData.byType, typeKey)
  return;
end
typeData = plotData.byType.(typeKey);
if isfield(typeData, 'sessionNames') && ~isempty(typeData.sessionNames)
  names = cellfun(@char, typeData.sessionNames(:), 'UniformOutput', false);
elseif isfield(typeData, 'sessionLabels') && ~isempty(typeData.sessionLabels)
  names = cellfun(@char, typeData.sessionLabels(:), 'UniformOutput', false);
end
end

function plotData = keep_ar_type_sessions(plotData, typeKey, keepNames)
% KEEP_AR_TYPE_SESSIONS - Subset one session-type's metric vectors by session name

if isempty(plotData) || ~isfield(plotData, 'byType') || ~isfield(plotData.byType, typeKey)
  return;
end
typeData = plotData.byType.(typeKey);
names = get_ar_type_session_names(plotData, typeKey);
if isempty(names)
  return;
end

keepIdx = zeros(1, numel(keepNames));
nKeep = 0;
for i = 1:numel(keepNames)
  j = find(strcmp(names, keepNames{i}), 1);
  if ~isempty(j)
    nKeep = nKeep + 1;
    keepIdx(nKeep) = j;
  end
end
keepIdx = keepIdx(1:nKeep);

if isfield(typeData, 'sessionNames') && ~isempty(typeData.sessionNames)
  typeData.sessionNames = typeData.sessionNames(keepIdx);
end
if isfield(typeData, 'sessionLabels') && ~isempty(typeData.sessionLabels)
  typeData.sessionLabels = typeData.sessionLabels(keepIdx);
end

fieldNames = fieldnames(typeData);
for f = 1:numel(fieldNames)
  fieldName = fieldNames{f};
  if ismember(fieldName, {'sessionNames', 'sessionLabels'})
    continue;
  end
  if ~iscell(typeData.(fieldName))
    continue;
  end
  for a = 1:numel(typeData.(fieldName))
    series = typeData.(fieldName){a};
    if isempty(series)
      continue;
    end
    series = series(:)';
    if numel(series) >= max(keepIdx)
      typeData.(fieldName){a} = series(keepIdx);
    elseif isempty(keepIdx)
      typeData.(fieldName){a} = series([]);
    end
  end
end

plotData.byType.(typeKey) = typeData;
end

function [yVals, ySem] = get_ar_ei_aligned_metric(plotData, sessionType, ...
    areaName, masterNames, meanField, semField)
% Align one cell-type metric series onto master session names

n = numel(masterNames);
yVals = nan(1, n);
ySem = nan(1, n);
if isempty(plotData) || ~isfield(plotData, 'areas')
  return;
end
areaIdx = find(strcmp(plotData.areas, areaName), 1);
if isempty(areaIdx)
  return;
end
typeKey = matlab.lang.makeValidName(sessionType);
if ~isfield(plotData.byType, typeKey)
  return;
end
typeData = plotData.byType.(typeKey);
if ~isfield(typeData, meanField) || areaIdx > numel(typeData.(meanField)) ...
    || isempty(typeData.(meanField){areaIdx})
  return;
end
means = typeData.(meanField){areaIdx}(:)';
sems = [];
if isfield(typeData, semField) && areaIdx <= numel(typeData.(semField))
  sems = typeData.(semField){areaIdx}(:)';
end
if numel(sems) ~= numel(means)
  sems = nan(size(means));
end

if isfield(typeData, 'sessionNames') && numel(typeData.sessionNames) >= numel(means)
  srcNames = cellfun(@char, typeData.sessionNames(1:numel(means)), 'UniformOutput', false);
elseif isfield(typeData, 'sessionLabels') && numel(typeData.sessionLabels) >= numel(means)
  srcNames = cellfun(@char, typeData.sessionLabels(1:numel(means)), 'UniformOutput', false);
else
  srcNames = {};
end

for i = 1:n
  matchIdx = [];
  if ~isempty(srcNames)
    matchIdx = find(strcmp(srcNames, masterNames{i}), 1);
  elseif numel(means) == n
    matchIdx = i;
  end
  if isempty(matchIdx)
    continue;
  end
  yVals(i) = means(matchIdx);
  ySem(i) = sems(matchIdx);
end
end

function [markerSpec, lineStyle, faceColor] = get_ar_ei_marker_style(cellType, taskColor)
% GET_AR_EI_MARKER_STYLE - Marker / mean-line style per E/I population

cellType = lower(char(cellType));
switch cellType
  case {'excitatory', 'e'}
    markerSpec = 'o';
    lineStyle = '--';
    faceColor = 'none';
  case {'inhibitory', 'i'}
    markerSpec = 'x';
    lineStyle = ':';
    faceColor = 'none';
  otherwise
    % combined (all) or unspecified
    markerSpec = 'o';
    lineStyle = '-';
    faceColor = taskColor;
end
end

function plot_d2_raw_with_shuffle(ax, plotData, sessionTypes, areaIdx, shuffleBarColor, enablePermutations, plotConfig)
% PLOT_D2_RAW_WITH_SHUFFLE - Session mean d2 with SEM; shuffle mean with SEM beside each session

if nargin < 6 || isempty(enablePermutations)
  enablePermutations = true;
end
if nargin < 7 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
enablePermutations = logical(enablePermutations);
markerSize = max(8, plotConfig.scatterMarkerSize / 2);

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
  if ~isfield(typeData, 'd2Mean') || areaIdx > length(typeData.d2Mean)
    continue;
  end

  taskColor = colors_for_tasks(sessionType);
  d2Means = typeData.d2Mean{areaIdx};
  d2Sems = typeData.d2Sem{areaIdx};
  shuffleMeans = typeData.d2ShuffleMean{areaIdx};
  shuffleSems = typeData.d2ShuffleSem{areaIdx};
  d2Means = d2Means(:)';
  d2Sems = d2Sems(:)';
  shuffleMeans = shuffleMeans(:)';
  shuffleSems = shuffleSems(:)';
  if numel(shuffleSems) ~= numel(shuffleMeans)
    shuffleSems = nan(size(shuffleMeans));
  end
  numBars = numel(d2Means);
  if numBars == 0
    continue;
  end

  xPos = xCursor + (1:numBars);
  errorbar(ax, xPos, d2Means, d2Sems, 'o', 'Color', taskColor, ...
    'MarkerFaceColor', taskColor, 'MarkerSize', markerSize, ...
    'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize, ...
    'DisplayName', 'session d2');

  if enablePermutations && any(isfinite(shuffleMeans))
    xShuffle = xPos + 0.22;
    semPlot = shuffleSems(:)';
    semPlot(~isfinite(semPlot)) = 0;
    errorbar(ax, xShuffle, shuffleMeans, semPlot, 's', ...
      'Color', shuffleBarColor, 'MarkerFaceColor', shuffleBarColor, ...
      'MarkerEdgeColor', [0.2, 0.2, 0.2], 'MarkerSize', markerSize, ...
      'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize, ...
      'DisplayName', 'shuffled mean \pm SEM (across windows)');
  end

  barLabels = get_session_bar_labels(typeData, numBars, sessionType);
  xticksCenters = [xticksCenters, xPos]; %#ok<AGROW>
  xtickLabels = [xtickLabels, barLabels]; %#ok<AGROW>

  validMeans = d2Means(isfinite(d2Means));
  if ~isempty(validMeans)
    yMeanTask = mean(validMeans);
    plot(ax, [xPos(1), xPos(end)], [yMeanTask, yMeanTask], '-', ...
      'Color', taskColor, 'LineWidth', plotConfig.lineWidth, 'HandleVisibility', 'off');
  end

  xCursor = xPos(end) + 1.5;
  legendShown = true;
end

if ~isempty(xticksCenters)
  xticks(ax, xticksCenters);
  xticklabels(ax, xtickLabels);
  xtickangle(ax, 45);
  xlim(ax, [min(xticksCenters) - 0.8, max(xticksCenters) + 0.8]);
end
if legendShown && enablePermutations
  legend(ax, {'session d2', 'shuffled mean \pm SEM (across windows)'}, ...
    'Location', 'best', 'FontSize', plotConfig.legendFontSize);
elseif legendShown
  legend(ax, {'session d2'}, 'Location', 'best', 'FontSize', plotConfig.legendFontSize);
end
hold(ax, 'off');
end

function plot_d2_normalized(ax, plotData, sessionTypes, areaIdx, plotConfig)
% PLOT_D2_NORMALIZED - Session mean normalized d2 with SEM across windows

if nargin < 5 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
markerSize = max(8, plotConfig.scatterMarkerSize / 2);

hold(ax, 'on');
xCursor = 0;
xticksCenters = [];
xtickLabels = {};

for t = 1:length(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if ~isfield(typeData, 'd2NormMean') || areaIdx > length(typeData.d2NormMean)
    continue;
  end

  taskColor = colors_for_tasks(sessionType);
  normMeans = typeData.d2NormMean{areaIdx};
  normSems = typeData.d2NormSem{areaIdx};
  normMeans = normMeans(:)';
  normSems = normSems(:)';
  numBars = numel(normMeans);
  if numBars == 0
    continue;
  end

  xPos = xCursor + (1:numBars);
  errorbar(ax, xPos, normMeans, normSems, 'o', 'Color', taskColor, ...
    'MarkerFaceColor', taskColor, 'MarkerSize', markerSize, ...
    'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize);

  barLabels = get_session_bar_labels(typeData, numBars, sessionType);
  xticksCenters = [xticksCenters, xPos]; %#ok<AGROW>
  xtickLabels = [xtickLabels, barLabels]; %#ok<AGROW>

  validMeans = normMeans(isfinite(normMeans));
  if ~isempty(validMeans)
    yMeanTask = mean(validMeans);
    plot(ax, [xPos(1), xPos(end)], [yMeanTask, yMeanTask], '-', ...
      'Color', taskColor, 'LineWidth', plotConfig.lineWidth, 'HandleVisibility', 'off');
  end

  xCursor = xPos(end) + 1.5;
end

if ~isempty(xticksCenters)
  xticks(ax, xticksCenters);
  xticklabels(ax, xtickLabels);
  xtickangle(ax, 45);
  xlim(ax, [min(xticksCenters) - 0.8, max(xticksCenters) + 0.8]);
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
barLabels = cellfun(@(s) truncate_session_xtick_label(s, 7), barLabels, ...
  'UniformOutput', false);
end

function label = truncate_session_xtick_label(label, maxChars)
% TRUNCATE_SESSION_XTICK_LABEL - Cap session-name tick text length
if nargin < 2 || isempty(maxChars)
  maxChars = 7;
end
label = char(label);
if numel(label) > maxChars
  label = label(1:maxChars);
end
end

function plotBase = make_ar_plot_basename(prefix, areaName, brainArea, d2Window, collectStart, collectEnd, multiArea, useLog10D2)
% MAKE_AR_PLOT_BASENAME - Filename stem for saved figures

if nargin < 8 || isempty(useLog10D2)
  useLog10D2 = false;
else
  useLog10D2 = logical(useLog10D2);
end

collectTag = format_collect_window_tag(collectStart, collectEnd);
if isempty(d2Window)
  winTag = 'full';
else
  winTag = sprintf('%.0fs', d2Window);
end
if ~isempty(brainArea)
  plotBase = sprintf('%s_%s_win%s_%s', prefix, brainArea, winTag, collectTag);
else
  plotBase = sprintf('%s_%s_win%s_%s', prefix, areaName, winTag, collectTag);
end
if multiArea
  plotBase = sprintf('%s_area%s', plotBase, areaName);
end
if useLog10D2
  plotBase = [plotBase, '_log10'];
end
end

function tag = format_collect_window_tag(collectStart, collectEnd)
% FORMAT_COLLECT_WINDOW_TAG - Filename fragment for collect window
if isempty(collectEnd)
  tag = sprintf('%.0f-full', collectStart);
else
  tag = sprintf('%.0f-%.0f', collectStart, collectEnd);
end
end

function label = format_collect_end_label(collectEnd)
% FORMAT_COLLECT_END_LABEL - Display string for collectEnd ([] = full)
if isempty(collectEnd)
  label = 'full';
else
  label = sprintf('%.1f', collectEnd);
end
end

function hasData = area_has_ar_plot_data(plotData, sessionTypes, areaIdx)
% AREA_HAS_AR_PLOT_DATA - True if any session type has d2 values for this area

hasData = false;
for t = 1:length(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if isfield(typeData, 'd2Mean') && areaIdx <= length(typeData.d2Mean) ...
      && ~isempty(typeData.d2Mean{areaIdx})
    hasData = true;
    return;
  end
end
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
