function m56_for_vaishnavi(sessionType, subjectName, sessionName)
% M56_FOR_VAISHNAVI - Write M56 spike times and unit IDs as .npy files
%
% Variables:
%   sessionType - Task that holds the session: 'spontaneous', 'interval',
%                 'semicircle', 'reach', or 'schall'
%   subjectName - Subject folder (e.g. 'ey9166')
%   sessionName - Session folder (e.g. 'ey9166_2026_03_27')
%
% Goal:
%   Find the session directory from sessionType, keep spikes from accepted
%   M56 units, and write two files into that directory:
%     spike_times_M56.npy    - spike times in seconds (float64, one column)
%     spike_unit_ids_M56.npy - cluster / unit ID for each spike (int64)
%   The two arrays are the same length and are sorted by spike time.
%
% Notes:
%   Uses the same unit rules as load_data: good, mua, or real
%   (cluster_quality_mask), 1.5 ms ISI filter, and M56 from the session
%   depth ranges. The full recording is saved (not the default 45 min window).

sessionType = char(sessionType);
subjectName = char(subjectName);
sessionName = char(sessionName);
if isempty(sessionType) || isempty(subjectName) || isempty(sessionName)
    error('sessionType, subjectName, and sessionName are required.');
end

paths = get_paths;
sessionFolder = find_kilosort_session_folder(paths, sessionType, subjectName, sessionName);
ensure_npy_matlab_on_path(paths);

opts = neuro_behavior_options;
opts.dataPath = fileparts(sessionFolder);
opts.sessionName = sessionName;
opts.subjectName = subjectName;
opts.collectStart = 0;
opts.collectEnd = [];

data = load_data(opts, 'spikes');
if ~ismember('area', data.ci.Properties.VariableNames)
    error('Cluster table in %s has no area labels.', sessionFolder);
end

isM56 = strcmp(string(data.ci.area), "M56");
if ismember('id', data.ci.Properties.VariableNames)
    m56UnitIds = data.ci.id(isM56);
else
    m56UnitIds = data.ci.cluster_id(isM56);
end
m56UnitIds = m56UnitIds(:);

keepSpikes = ismember(data.spikeClusters, m56UnitIds);
spikeTimes = double(data.spikeTimes(keepSpikes));
unitIds = data.spikeClusters(keepSpikes);
spikeTimes = spikeTimes(:);
unitIds = unitIds(:);

if isempty(m56UnitIds) || isempty(spikeTimes)
    error('No accepted M56 spikes in %s / %s / %s.', sessionType, subjectName, sessionName);
end

[spikeTimes, sortIdx] = sort(spikeTimes);
unitIds = int64(unitIds(sortIdx));

spikeTimesFile = fullfile(sessionFolder, 'spike_times_M56.npy');
unitIdsFile = fullfile(sessionFolder, 'spike_unit_ids_M56.npy');
writeNPY(spikeTimes, spikeTimesFile);
writeNPY(unitIds, unitIdsFile);

fprintf('M56: %d units, %d spikes\n', numel(m56UnitIds), numel(spikeTimes));
fprintf('Wrote %s\n', spikeTimesFile);
fprintf('Wrote %s\n', unitIdsFile);
end

function sessionFolder = find_kilosort_session_folder(paths, sessionType, subjectName, sessionName)
% FIND_KILOSORT_SESSION_FOLDER - Session directory that already stores spikes
%
% Variables:
%   paths       - Paths from get_paths
%   sessionType - Task name used to pick the data root
%   subjectName - Subject folder under that data root
%   sessionName - Session folder under subject
%
% Goal:
%   Return <task data root>/subjectName/sessionName when it contains
%   spike_times.npy.

switch lower(sessionType)
    case 'spontaneous'
        basePath = paths.spontaneousDataPath;
    case 'interval'
        basePath = paths.intervalDataPath;
    case 'semicircle'
        basePath = paths.semicircleDataPath;
    case 'reach'
        basePath = paths.reachDataPath;
    case 'schall'
        basePath = paths.schallDataPath;
    otherwise
        error(['sessionType must be ''spontaneous'', ''interval'', ', ...
            '''semicircle'', ''reach'', or ''schall''. Got ''%s''.'], sessionType);
end

sessionFolder = fullfile(basePath, subjectName, sessionName);
if ~isfolder(sessionFolder)
    error('Session folder not found: %s', sessionFolder);
end
if ~isfile(fullfile(sessionFolder, 'spike_times.npy'))
    error('spike_times.npy not found in %s', sessionFolder);
end
end

function ensure_npy_matlab_on_path(paths)
% ENSURE_NPY_MATLAB_ON_PATH - Add npy-matlab when writeNPY is not visible
%
% Variables:
%   paths - Paths from get_paths
%
% Goal:
%   Make writeNPY available without requiring a prior addpath.

if exist('writeNPY', 'file') == 2
    return
end
addpath(fullfile(paths.homePath, 'toolboxes', 'npy-matlab', 'npy-matlab'));
if exist('writeNPY', 'file') ~= 2
    error('writeNPY was not found. Expected npy-matlab under toolboxes.');
end
end
