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
%   brainArea          - Area to analyze (e.g. 'M56'); '' uses all valid areas
%   saveFigure         - Export PNG/EPS to dropPath/criticality_manuscript
%   prgMethod          - 'pca' (momentum-space) or 'icg' (real-space ICG)
%   surrogateMethod    - 'isi' (ISI shuffle per unit) or 'circular' (circshift per neuron)
%   useSubsampling     - If true, kappa/D_JS per window = mean across neuron subsamples
%   nSubsamples, nNeuronsSubsample, minNeuronsMultiple - subsampling settings
%   splitExcitatoryInhibitory - If true, run separately for E and I units (waveforms.mat)
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

brainArea = 'M56';
saveFigure = false;

useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 20;
minNeuronsMultiple = 1.25;

splitExcitatoryInhibitory = false;
widthCutoff = 0.035;  % ms; peak-to-trough width (narrow <= cutoff = inhibitory)

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
analysisConfig.includeM2356 = false;
if ~isempty(brainArea) && strcmpi(brainArea, 'M2356')
  analysisConfig.includeM2356 = true;
end

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
subjectNameForLoad = subjectName;
loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

[dataStruct, areaOk] = apply_brain_area_selection(dataStruct, brainArea);
if ~areaOk
  error('Brain area "%s" not available in this session.', brainArea);
end

cellTypesToRun = get_cell_types_to_run(splitExcitatoryInhibitory);
for iCellRun = 1:numel(cellTypesToRun)
  cellType = cellTypesToRun{iCellRun};
  dataStructRun = copy_neuron_selection(dataStruct);

  if splitExcitatoryInhibitory
    fprintf('\n--- Cell type: %s ---\n', cellType);
    [dataStructRun, ~] = apply_session_cell_type_filter(dataStructRun, paths, cellType, widthCutoff);
  end

  results = criticality_prg_analysis(dataStructRun, analysisConfig);

  if ~isempty(brainArea)
    results = filter_prg_results_to_brain_area(results, brainArea);
    if isempty(results.areas)
      error('No PRG results for brain area "%s" (%s).', brainArea, cell_type_label(cellType));
    end
  end

  print_session_kappa_summary(results);

  %% Overlapping kurtosis distributions (real vs surrogate)
  plotConfig = struct('savePlots', false);
  if saveFigure
    saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    plotConfig.saveDir = saveDir;
  end

  criticality_prg_plot(results, plotConfig, analysisConfig, dataStructRun);

  figDist = findobj(0, 'Type', 'figure', 'Name', 'PRG kurtosis distributions');
  if isempty(figDist)
    error(['No distribution figure created. Enable surrogates and ensure valid ' ...
      'kappa windows (check CV exclusion).']);
  end
  figDist = figDist(1);

  if splitExcitatoryInhibitory
    sgtitle(figDist, sprintf('%s | %s cells | width cutoff %.3f ms', ...
      sessionName, cellType, widthCutoff), 'Interpreter', 'none');
  end

  if saveFigure
    areaTag = format_areas_label(results.areas);
    plotBase = sprintf('session_kurtosis_distributions_%s_%s_%s_%s_win%.0fs_%.0f-%.0fs_N%d%s', ...
      sessionName, areaTag, analysisConfig.prgMethod, analysisConfig.surrogateMethod, prgWindow, ...
      collectStart, collectEnd, analysisConfig.finalCutoffDivisor, cell_type_file_tag(cellType));
    exportgraphics(figDist, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
    exportgraphics(figDist, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
    fprintf('\nSaved figure: %s\n', fullfile(saveDir, plotBase));
  end
end

fprintf('\n=== Done ===\n');

%% Local functions

function [dataStruct, areaOk] = apply_brain_area_selection(dataStruct, brainArea)
% APPLY_BRAIN_AREA_SELECTION - Restrict analysis to one brain area

areaOk = true;
if isempty(brainArea)
  return;
end

if strcmpi(brainArea, 'M2356')
  areaOk = any(strcmp(dataStruct.areas, 'M23')) && any(strcmp(dataStruct.areas, 'M56'));
  return;
end

areaIdx = find(strcmp(dataStruct.areas, brainArea), 1);
if isempty(areaIdx)
  areaOk = false;
  return;
end

dataStruct.areasToTest = areaIdx;
fprintf('  Restricting analysis to area: %s\n', brainArea);
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

function cellTypes = get_cell_types_to_run(splitExcitatoryInhibitory)
% GET_CELL_TYPES_TO_RUN - Cell types to analyze when E/I split is enabled

if splitExcitatoryInhibitory
  cellTypes = {'excitatory', 'inhibitory'};
else
  cellTypes = {''};
end
end

function dataStructCopy = copy_neuron_selection(dataStruct)
% COPY_NEURON_SELECTION - Copy idLabel/idMatIdx before waveform filtering

dataStructCopy = dataStruct;
for a = 1:numel(dataStruct.areas)
  dataStructCopy.idLabel{a} = dataStruct.idLabel{a}(:)';
  dataStructCopy.idMatIdx{a} = dataStruct.idMatIdx{a}(:)';
end
end

function tag = cell_type_file_tag(cellType)
% CELL_TYPE_FILE_TAG - Filename suffix for E/I runs

if isempty(cellType)
  tag = '';
else
  tag = ['_' cellType];
end
end

function label = cell_type_label(cellType)
% CELL_TYPE_LABEL - Display label for command-window messages

if isempty(cellType)
  label = 'all units';
else
  label = cellType;
end
end
