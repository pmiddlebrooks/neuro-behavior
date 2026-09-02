function brainAreaDepths = brain_area_depths(sessionType, subjectName, sessionName)
% BRAIN_AREA_DEPTHS - Interactive brain-area depth boundaries for a session
%
% Variables:
%   sessionType - Task name: 'spontaneous', 'interval', or any kilosort
%                 session folder with cluster_info.tsv / cluster_rf.tsv
%   subjectName - Subject folder under the task data root (e.g. 'ag', 'ey9166')
%   sessionName - Session folder under subject (e.g. 'ag112321_1')
%
% Goal:
%   Load cluster_info.tsv (or cluster_rf.tsv), plot a jittered scatter of
%   good / mua / real unit depths (noise excluded; see cluster_quality_mask),
%   overlay default area boundaries from load_data.m,
%   then prompt for a single depth shift (um, positive or negative) applied
%   to all area range values. Save results to brain_area_depths.mat in the
%   session folder.
%
% Usage (function):
%   brainAreaDepths = brain_area_depths('spontaneous', 'ag', 'ag112321_1');
%   brainAreaDepths = brain_area_depths('interval', 'ey9166', 'ey9166_2026_04_03');
%
% Usage (script): set sessionType, subjectName, and sessionName in the workspace,
%   then run: brain_area_depths

if nargin < 3
    if ~exist('sessionType', 'var') || ~exist('subjectName', 'var') || ~exist('sessionName', 'var')
        error(['Provide sessionType, subjectName, and sessionName as arguments, ', ...
            'or define them in the workspace before running brain_area_depths.']);
    end
end

paths = get_paths;
sessionFolder = get_cluster_session_folder(sessionType, paths, subjectName, sessionName);
ci = load_session_cluster_info(sessionFolder, sessionName);
nUnitsAll = height(ci);
ci = ci(cluster_quality_mask(ci, struct('sessionName', sessionName)), :);
[spikeTimes, spikeClusters] = load_session_spike_trains(sessionFolder);
ci.firingRateHz = compute_unit_firing_rates(ci, spikeTimes, spikeClusters);

areaNames = {'M23', 'M56', 'CC', 'DS', 'VS'};
[m23, m56, cc, ds, vs, depthSource] = get_brain_area_depth_ranges(sessionFolder);
depthRanges = [m23; m56; cc; ds; vs];
if strcmp(depthSource, 'session')
    fprintf('Starting from existing brain_area_depths.mat\n');
end

figHandle = plot_unit_depths(ci, areaNames, depthRanges, sessionType, subjectName, sessionName);
draw_area_border_lines(depthRanges);

fprintf('\nSession: %s / %s (%s)\n', subjectName, sessionName, sessionType);
fprintf('Folder: %s\n', sessionFolder);
fprintf('Units plotted (good / mua / real): %d / %d in cluster_info\n\n', height(ci), nUnitsAll);
print_depth_ranges(areaNames, depthRanges);

while true
    userInput = input('Press Enter to accept, or shift all ranges by um (e.g. 80 or -50): ', 's');
    if isempty(userInput)
        fprintf('Keeping ranges as shown.\n');
        break
    end

    depthShift = parse_depth_shift_input(userInput);
    if isempty(depthShift)
        continue
    end

    depthRanges = depthRanges + depthShift;
    fprintf('Shifted all range values by %d um.\n', depthShift);
    print_depth_ranges(areaNames, depthRanges);
    plot_unit_depths(ci, areaNames, depthRanges, sessionType, subjectName, sessionName, figHandle);
    draw_area_border_lines(depthRanges);
end

brainAreaDepths = struct();
brainAreaDepths.sessionType = sessionType;
brainAreaDepths.subjectName = subjectName;
brainAreaDepths.sessionName = sessionName;
brainAreaDepths.sessionFolder = sessionFolder;
brainAreaDepths.areaNames = areaNames;
brainAreaDepths.depthRanges = depthRanges;
brainAreaDepths.m23 = depthRanges(1, :);
brainAreaDepths.m56 = depthRanges(2, :);
brainAreaDepths.cc = depthRanges(3, :);
brainAreaDepths.ds = depthRanges(4, :);
brainAreaDepths.vs = depthRanges(5, :);

m23 = brainAreaDepths.m23;
m56 = brainAreaDepths.m56;
cc = brainAreaDepths.cc;
ds = brainAreaDepths.ds;
vs = brainAreaDepths.vs;

savePath = fullfile(sessionFolder, 'brain_area_depths.mat');
save(savePath, 'm23', 'm56', 'cc', 'ds', 'vs', 'sessionType', 'subjectName', 'sessionName', ...
    'areaNames', 'depthRanges', '-v7.3');

fprintf('\nSaved brain area depths to:\n  %s\n', savePath);
end

function sessionFolder = get_cluster_session_folder(sessionType, paths, subjectName, sessionName)
% GET_CLUSTER_SESSION_FOLDER - Resolve session folder for cluster_info / cluster_rf
%
% Variables:
%   sessionType - task name used to pick the data root
%   paths       - struct from get_paths
%   subjectName - subject folder under the task data root
%   sessionName - session folder under subject
%
% Goal:
%   Return the folder containing cluster_info.tsv or cluster_rf.tsv.

if isempty(subjectName)
    error('subjectName is required for %s sessions.', sessionType);
end

switch lower(sessionType)
    case 'spontaneous'
        basePath = paths.spontaneousDataPath;
    case 'interval'
        basePath = paths.intervalDataPath;
    case 'semicircle'
        basePath = paths.semicircleDataPath;
    case 'reach'
        basePath = paths.reachDataPath;
    otherwise
        error(['brain_area_depths needs a kilosort session folder ', ...
            '(cluster_info.tsv or cluster_rf.tsv). Got sessionType = %s.'], sessionType);
end

sessionFolder = fullfile(basePath, subjectName, sessionName);

clusterInfoPath = fullfile(sessionFolder, 'cluster_info.tsv');
clusterRfPath = fullfile(sessionFolder, 'cluster_rf.tsv');
if ~isfolder(sessionFolder)
    error('Session folder not found: %s', sessionFolder);
end
if ~isfile(clusterInfoPath) && ~isfile(clusterRfPath)
    error('Neither cluster_info.tsv nor cluster_rf.tsv found in %s', sessionFolder);
end
end

function [spikeTimes, spikeClusters] = load_session_spike_trains(sessionFolder)
% LOAD_SESSION_SPIKE_TRAINS - Load spike times and cluster ids for a session
%
% Variables:
%   sessionFolder - path to session directory
%
% Returns:
%   spikeTimes    - spike times in seconds
%   spikeClusters - cluster id per spike

opts = neuro_behavior_options;
spikeTimesPath = fullfile(sessionFolder, 'spike_times.npy');
spikeClustersPath = fullfile(sessionFolder, 'spike_clusters.npy');
if ~isfile(spikeTimesPath)
    error('spike_times.npy not found in %s', sessionFolder);
end
if ~isfile(spikeClustersPath)
    error('spike_clusters.npy not found in %s', sessionFolder);
end
spikeTimes = double(readNPY(spikeTimesPath)) / opts.fsSpike;
spikeClusters = readNPY(spikeClustersPath);
end

function clusterIds = get_cluster_ids(ci)
% GET_CLUSTER_IDS - Unit cluster ids from cluster_info table
%
% Variables:
%   ci - cluster_info table
%
% Goal:
%   Return vector of cluster ids aligned with ci rows.

if ismember('cluster_id', ci.Properties.VariableNames)
    clusterIds = ci.cluster_id;
elseif ismember('id', ci.Properties.VariableNames)
    clusterIds = ci.id;
else
    error('cluster_info.tsv must contain cluster_id or id column.');
end
end

function firingRateHz = compute_unit_firing_rates(ci, spikeTimes, spikeClusters)
% COMPUTE_UNIT_FIRING_RATES - Mean spike rate (Hz) per unit in cluster_info
%
% Variables:
%   ci            - cluster_info table (one row per unit)
%   spikeTimes    - all spike times (seconds)
%   spikeClusters - cluster id per spike
%
% Goal:
%   Return firing rate in spikes/s for each row of ci over the full recording.

clusterIds = get_cluster_ids(ci);
recordingDurationSec = spikeTimes(end) - spikeTimes(1);
if recordingDurationSec <= 0
    recordingDurationSec = spikeTimes(end);
end
if recordingDurationSec <= 0
    error('Cannot compute firing rates: recording duration is zero.');
end

nUnits = height(ci);
firingRateHz = zeros(nUnits, 1);
for iUnit = 1:nUnits
    nSpikes = sum(spikeClusters == clusterIds(iUnit));
    firingRateHz(iUnit) = nSpikes / recordingDurationSec;
end
end

function markerSizes = firing_rate_marker_sizes(firingRateHz)
% FIRING_RATE_MARKER_SIZES - Map firing rates to scatter marker sizes
%
% Variables:
%   firingRateHz - vector of firing rates in spikes/s
%
% Goal:
%   Linearly scale marker area between min and max for visualization.

minMarkerSize = 8;
maxMarkerSize = 120;
firingRateHz = firingRateHz(:);
if all(firingRateHz <= 0) || ~any(isfinite(firingRateHz))
    markerSizes = repmat(mean([minMarkerSize, maxMarkerSize]), numel(firingRateHz), 1);
    return;
end
rateMin = min(firingRateHz(isfinite(firingRateHz) & firingRateHz > 0));
rateMax = max(firingRateHz);
if rateMax <= rateMin
    markerSizes = repmat(mean([minMarkerSize, maxMarkerSize]), numel(firingRateHz), 1);
else
    markerSizes = minMarkerSize + (maxMarkerSize - minMarkerSize) * ...
        (firingRateHz - rateMin) / (rateMax - rateMin);
end
markerSizes(~isfinite(firingRateHz) | firingRateHz <= 0) = minMarkerSize;
end

function figHandle = plot_unit_depths(ci, areaNames, depthRanges, sessionType, subjectName, sessionName, figHandle)
% PLOT_UNIT_DEPTHS - Jittered scatter of unit depths (surface at top)
%
% Variables:
%   ci           - cluster_info table with depth column
%   areaNames    - cell of area labels (used for area assignment elsewhere)
%   depthRanges  - numAreas x 2 depth bounds
%   sessionType  - session type string
%   subjectName  - subject folder name
%   sessionName  - session name string
%   figHandle    - optional existing figure handle
%
% Goal:
%   Plot good / mua / real units in one jittered vertical column vs depth;
%   marker size is proportional to ci.firingRateHz when present.

if nargin < 7 || isempty(figHandle) || ~isvalid(figHandle)
    figHandle = figure('Color', 'w', 'Name', 'Brain area depths');
else
    figure(figHandle);
end
clf(figHandle);

unitDepths = ci.depth;
nUnits = numel(unitDepths);
xJitter = 1 + 0.30 * (rand(nUnits, 1) - 0.5);

if ismember('firingRateHz', ci.Properties.VariableNames)
    markerSizes = firing_rate_marker_sizes(ci.firingRateHz);
else
    markerSizes = repmat(18, nUnits, 1);
end

scatter(xJitter, unitDepths, markerSizes, 'k', 'filled', 'MarkerFaceAlpha', 0.65);

xlim([0.5, 1.5]);
ylim([-50, 3890]);
set(gca, 'YDir', 'reverse', 'XTick', []);
ylabel('Depth from surface (\mum)');
title(sprintf('Unit depths: %s / %s (%s)', subjectName, sessionName, sessionType), 'Interpreter', 'none');
grid on;
end

function draw_area_border_lines(depthRanges)
% DRAW_AREA_BORDER_LINES - Horizontal lines at area boundaries
%
% Variables:
%   depthRanges - numAreas x 2 depth bounds
%
% Goal:
%   Draw y-lines at the upper bound of each shallow area (except deepest).

borderDepths = depthRanges(1:end-1, 2);
yl = yline(borderDepths, '--', 'Color', [0.15 0.15 0.15], 'LineWidth', 1.1);
set(yl, 'HandleVisibility', 'off');
end

function print_depth_ranges(areaNames, depthRanges)
% PRINT_DEPTH_RANGES - Print min/max depth per area
%
% Variables:
%   areaNames   - cell of area labels
%   depthRanges - numAreas x 2 depth bounds

for iArea = 1:numel(areaNames)
    fprintf('  %s: [%d, %d] um\n', areaNames{iArea}, depthRanges(iArea, 1), depthRanges(iArea, 2));
end
end

function depthShift = parse_depth_shift_input(userInput)
% PARSE_DEPTH_SHIFT_INPUT - Parse a single um offset (positive or negative)
%
% Variables:
%   userInput - character vector from input()
%
% Goal:
%   Return a rounded scalar shift, or [] if the input is invalid.

depthShift = sscanf(strtrim(userInput), '%f');
if numel(depthShift) ~= 1 || ~isfinite(depthShift)
    warning('Could not parse "%s". Enter one number (e.g. 80 or -50).', userInput);
    depthShift = [];
    return
end
depthShift = round(depthShift);
end
