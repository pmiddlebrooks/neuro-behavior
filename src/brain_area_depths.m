function brainAreaDepths = brain_area_depths(sessionType, subjectName, sessionName)
% BRAIN_AREA_DEPTHS - Interactive brain-area depth boundaries for a session
%
% Variables:
%   sessionType - 'spontaneous' or 'interval' (sessions with cluster_info.tsv)
%   subjectName - Subject folder under the task data root (e.g. 'ag', 'ey9166')
%   sessionName - Session folder under subject (e.g. 'ag112321_1')
%
% Goal:
%   Load cluster_info.tsv, plot a jittered scatter of unit depths, overlay
%   default area boundaries from load_data.m, then prompt per area to accept
%   or edit depth ranges. Save results to brain_area_depths.mat in the
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
ci = load_cluster_info(sessionFolder);
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
fprintf('Units in cluster_info: %d\n\n', height(ci));

for iArea = 1:numel(areaNames)
    areaName = areaNames{iArea};
    currentRange = depthRanges(iArea, :);
    highlight_area_on_plot(figHandle, ci, areaNames, depthRanges, iArea, sessionType, subjectName, sessionName);

    fprintf('--- %s: depth range [%d, %d] um ---\n', areaName, currentRange(1), currentRange(2));
    userInput = input('Press Enter to accept, or enter new range as "min max": ', 's');

    if ~isempty(userInput)
        newRange = parse_depth_range_input(userInput, currentRange);
        if ~isequal(newRange, currentRange)
            depthRanges(iArea, :) = newRange;
            fprintf('Updated %s to [%d, %d] um.\n', areaName, newRange(1), newRange(2));
            plot_unit_depths(ci, areaNames, depthRanges, sessionType, subjectName, sessionName, figHandle);
            draw_area_border_lines(depthRanges);
        end
    else
        fprintf('Keeping %s at [%d, %d] um.\n', areaName, currentRange(1), currentRange(2));
    end
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
% GET_CLUSTER_SESSION_FOLDER - Resolve session folder for cluster_info.tsv
%
% Variables:
%   sessionType - 'spontaneous' or 'interval'
%   paths       - struct from get_paths
%   subjectName - subject folder under the task data root
%   sessionName - session folder under subject
%
% Goal:
%   Return the folder containing cluster_info.tsv for this session.

if isempty(subjectName)
    error('subjectName is required for %s sessions.', sessionType);
end

switch lower(sessionType)
    case 'spontaneous'
        basePath = paths.spontaneousDataPath;
    case 'interval'
        basePath = paths.intervalDataPath;
    otherwise
        error(['brain_area_depths supports spontaneous and interval sessions ', ...
            '(cluster_info.tsv). Got sessionType = %s.'], sessionType);
end

sessionFolder = fullfile(basePath, subjectName, sessionName);

clusterInfoPath = fullfile(sessionFolder, 'cluster_info.tsv');
if ~isfolder(sessionFolder)
    error('Session folder not found: %s', sessionFolder);
end
if ~isfile(clusterInfoPath)
    error('cluster_info.tsv not found in %s', sessionFolder);
end
end

function ci = load_cluster_info(sessionFolder)
% LOAD_CLUSTER_INFO - Load and orient cluster depths like load_data.m
%
% Variables:
%   sessionFolder - path to session directory
%
% Goal:
%   Return cluster_info table with depth 0 = superficial (M23) and 3840 = deep (VS).

clusterInfoPath = fullfile(sessionFolder, 'cluster_info.tsv');
ci = readtable(clusterInfoPath, 'FileType', 'text', 'Delimiter', '\t');
ci = sortrows(ci, 'depth');
ci.depth = 3840 - ci.depth;
ci = flipud(ci);
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
%   Plot all units in one jittered vertical column vs depth; marker size
%   is proportional to ci.firingRateHz when present.

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

function highlight_area_on_plot(figHandle, ci, areaNames, depthRanges, iArea, sessionType, subjectName, sessionName)
% HIGHLIGHT_AREA_ON_PLOT - Emphasize units in the current area on the figure
%
% Variables:
%   figHandle    - figure handle
%   ci           - cluster_info table
%   areaNames    - area labels
%   depthRanges  - depth bounds
%   iArea        - index of area being reviewed
%   sessionType  - session type string
%   subjectName  - subject folder name
%   sessionName  - session name string
%
% Goal:
%   Retitle the figure while the user reviews the current area.

figure(figHandle);
areaName = areaNames{iArea};
currentRange = depthRanges(iArea, :);
areaAssignments = assign_areas(ci.depth, areaNames, depthRanges);
nUnits = sum(strcmp(areaAssignments, areaName));
title(sprintf('%s / %s (%s) | reviewing %s [%d, %d] um (%d units)', ...
    subjectName, sessionName, sessionType, areaName, currentRange(1), currentRange(2), nUnits), ...
    'Interpreter', 'none');
end

function areaAssignments = assign_areas(unitDepths, areaNames, depthRanges)
% ASSIGN_AREAS - Label each unit with a brain area from depth ranges
%
% Variables:
%   unitDepths  - vector of unit depths
%   areaNames   - area labels
%   depthRanges - numAreas x 2 bounds
%
% Goal:
%   Return cellstr area label per unit (empty if outside all ranges).

nUnits = numel(unitDepths);
areaAssignments = repmat({''}, nUnits, 1);
for iArea = 1:numel(areaNames)
    inRange = unitDepths >= depthRanges(iArea, 1) & unitDepths <= depthRanges(iArea, 2);
    areaAssignments(inRange) = areaNames(iArea);
end
end

function newRange = parse_depth_range_input(userInput, currentRange)
% PARSE_DEPTH_RANGE_INPUT - Parse user-entered depth range
%
% Variables:
%   userInput     - character vector from input()
%   currentRange  - fallback 1x2 range
%
% Goal:
%   Return validated [minDepth maxDepth] or currentRange on parse failure.

newRange = sscanf(strtrim(userInput), '%f %f');
if numel(newRange) ~= 2 || any(~isfinite(newRange))
    warning('Could not parse "%s". Keeping [%d, %d].', userInput, currentRange(1), currentRange(2));
    newRange = currentRange;
    return;
end
newRange = round(newRange(:))';
if newRange(2) < newRange(1)
    newRange = fliplr(newRange);
end
end
