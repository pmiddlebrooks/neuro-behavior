%%
% Criticality Spontaneous d2 (temporary)
%
% Stripped-down spontaneous-only d2 from criticality_spontaneous_vs_task.m.
% One subject, spontaneous sessions only. Sleep periods stay in the recording
% (restrictToAwake = false). Window d2 values are left in spontaneousD2;
% this script does not save them.
%
% Variables:
%   subjectName - Must match spontaneous_session_list
%   d2Method    - 'euclidean' or 'kl'
%   d2Window    - Non-overlapping window length (s)
%   collectEnd  - Analysis window end (s from session onset); [] = full session
%
% Goal:
%   Obtain per-window d2 for each spontaneous session of one subject,
%   including sleep, and store the values in spontaneousD2.

%% Configuration
subjectName = 'ey9387';
d2Method = 'kl';         % 'euclidean' or 'kl'
klFitMethod = 'MaxLikelihood';
klErrBars = false;
runParallel = true;
nWorkers = 5;            % used only when runParallel is true
klParallel = runParallel;

collectStart = 0;
collectEnd = 120 * 60;   % [] = full session
d2Window = 45;
binSizeD2 = 0.04;

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();

useSessionCache = true;
forceRecompute = false;
useLog10D2 = false;
useSubsampling = false;
nSubsamples = 40;
nNeuronsSubsample = 75;
minNeuronsMultiple = 1.1;
enablePermutations = false;
pcaFlag = false;
pcaFirstFlag = true;
nDim = 10;
restrictToAwake = true;  % keep sleep windows

if isempty(subjectName)
  error('Set subjectName to a subject in spontaneous_session_list.');
end

setup_criticality_manuscript_paths('criticality_spontaneous_temp');

[d2Method, klFitMethod, klErrBars, klParallel] = normalize_kl_d2_options( ...
  d2Method, klFitMethod, klErrBars, klParallel);
if ~strcmp(d2Method, 'kl')
  runParallel = false;
  klParallel = false;
end

fprintf('\n=== Criticality spontaneous d2 (include sleep) ===\n');
fprintf('Subject: %s\n', char(subjectName));
fprintf('d2Method: %s\n', d2Method);
if strcmp(d2Method, 'kl')
  fprintf('KL fit: %s; klErrBars=%d; klParallel=%d; nWorkers=%d\n', ...
    klFitMethod, klErrBars, klParallel, nWorkers);
end
if isempty(collectEnd)
  fprintf('Collect window: [%.1f, full] s\n', collectStart);
else
  fprintf('Collect window: [%.1f, %.1f] s\n', collectStart, collectEnd);
end
fprintf('d2 windows: %.0f s (non-overlapping)\n', d2Window);
fprintf('binSizeD2: %.3f s; brainArea: %s; restrictToAwake: %d\n', ...
  binSizeD2, brainArea, restrictToAwake);

arOpts = struct( ...
  'sessionTypes', {{'spontaneous'}}, ...
  'collectStart', collectStart, ...
  'collectEnd', collectEnd, ...
  'd2Window', d2Window, ...
  'binSize', binSizeD2, ...
  'brainArea', brainArea, ...
  'brainAreaCombinations', {brainAreaCombinations}, ...
  'areasToPlot', {{}}, ...
  'useLog10D2', useLog10D2, ...
  'useSubsampling', useSubsampling, ...
  'nSubsamples', nSubsamples, ...
  'nNeuronsSubsample', nNeuronsSubsample, ...
  'minNeuronsMultiple', minNeuronsMultiple, ...
  'enablePermutations', enablePermutations, ...
  'pcaFlag', pcaFlag, ...
  'pcaFirstFlag', pcaFirstFlag, ...
  'nDim', nDim, ...
  'useSessionCache', useSessionCache, ...
  'forceRecompute', forceRecompute, ...
  'plotResults', false, ...
  'saveBatchResults', false, ...
  'subjectName', subjectName, ...
  'd2Method', d2Method, ...
  'klFitMethod', klFitMethod, ...
  'klErrBars', klErrBars, ...
  'klParallel', klParallel, ...
  'nWorkers', nWorkers, ...
  'restrictToAwake', restrictToAwake);

arOut = criticality_ar_across_tasks(arOpts);
spontaneousD2 = pack_spontaneous_d2(arOut, subjectName, brainArea, d2Window, ...
  binSizeD2, collectStart, collectEnd, d2Method, klFitMethod, useLog10D2, ...
  restrictToAwake);

nOk = sum([spontaneousD2.sessions.success]);
fprintf('\nStored d2 for %d/%d spontaneous sessions in spontaneousD2.\n', ...
  nOk, numel(spontaneousD2.sessions));

function d2Struct = pack_spontaneous_d2(arOut, subjectName, brainArea, d2Window, ...
    binSizeD2, collectStart, collectEnd, d2Method, klFitMethod, useLog10D2, ...
    restrictToAwake)
% PACK_SPONTANEOUS_D2 - Copy window d2 values into a saveable struct
%
% Variables:
%   arOut            - Output of criticality_ar_across_tasks
%   subjectName      - Subject filter used for the batch
%   brainArea        - Area requested for the batch
%   d2Window         - Window length (s)
%   binSizeD2        - Spike bin width (s)
%   collectStart     - Collect start (s)
%   collectEnd       - Collect end (s); [] = full session
%   d2Method         - 'euclidean' or 'kl'
%   klFitMethod      - KL fit name (unused for euclidean)
%   useLog10D2       - True when stored values are log10(d2)
%   restrictToAwake  - False when sleep windows are included
%
% Goal:
%   Keep per-session window d2 (and window start times) without the rest of
%   the AR batch output.
%
% Returns:
%   d2Struct - Metadata plus d2Struct.sessions(i).d2

if isempty(arOut) || ~isfield(arOut, 'batchResults') || isempty(arOut.batchResults)
  error('criticality_spontaneous_temp:NoD2', ...
    'No d2 batch results for subject "%s".', char(subjectName));
end

batchResults = arOut.batchResults;
nSessions = numel(batchResults);
emptySession = struct( ...
  'subjectName', '', ...
  'sessionName', '', ...
  'success', false, ...
  'skipReason', '', ...
  'areas', {{}}, ...
  'd2', {{}}, ...
  'startS', {{}}, ...
  'nNeurons', []);
sessions = repmat(emptySession, nSessions, 1);

for iSession = 1:nSessions
  entry = batchResults(iSession);
  sessions(iSession).subjectName = char(entry.subjectName);
  sessions(iSession).sessionName = char(entry.sessionName);
  sessions(iSession).success = logical(entry.success);
  if isfield(entry, 'skipReason') && ~isempty(entry.skipReason)
    sessions(iSession).skipReason = char(entry.skipReason);
  end
  if ~entry.success || isempty(entry.results) || ~isfield(entry.results, 'd2')
    continue;
  end

  results = entry.results;
  sessions(iSession).areas = results.areas;
  sessions(iSession).d2 = results.d2;
  if isfield(results, 'startS')
    sessions(iSession).startS = results.startS;
  end
  if isfield(results, 'nNeurons')
    sessions(iSession).nNeurons = results.nNeurons;
  end

  nWindows = 0;
  if ~isempty(results.d2) && ~isempty(results.d2{1})
    nWindows = numel(results.d2{1});
  end
  fprintf('  %s: %d d2 windows\n', sessions(iSession).sessionName, nWindows);
end

d2Struct = struct();
d2Struct.subjectName = char(subjectName);
d2Struct.sessionType = 'spontaneous';
d2Struct.brainArea = char(brainArea);
d2Struct.d2Window = d2Window;
d2Struct.binSize = binSizeD2;
d2Struct.collectStart = collectStart;
d2Struct.collectEnd = collectEnd;
d2Struct.d2Method = char(d2Method);
d2Struct.klFitMethod = char(klFitMethod);
d2Struct.useLog10D2 = logical(useLog10D2);
d2Struct.restrictToAwake = logical(restrictToAwake);
d2Struct.sessions = sessions;
end
