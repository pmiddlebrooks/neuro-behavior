function areaChan = select_area_lfp_channels(sessionFolder, brainAreas)
% SELECT_AREA_LFP_CHANNELS - Two NP1 LFP channels per brain area (1/3 and 2/3 depth)
%
% Variables:
%   sessionFolder - Session directory (brain_area_depths.mat and channel map)
%   brainAreas    - Cell of area names, e.g. {'M23','M56','DS','VS'}
%
% Goal:
%   Use the same depth ranges as load_session_cluster_info.m /
%   get_brain_area_depth_ranges.m. For each requested area, pick the raw
%   LFP channels nearest 1/3 and 2/3 of that depth span and return their
%   IDs and oriented depths (0 = surface, 3840 = deepest).

if nargin < 2 || isempty(brainAreas)
    brainAreas = {'M23', 'M56', 'DS', 'VS'};
end
brainAreas = normalize_area_names(brainAreas);

[channelIds, orientedDepths, chanMeta] = load_np1_lfp_channel_depths(sessionFolder);
[m23, m56, cc, ds, vs, depthSource] = get_brain_area_depth_ranges(sessionFolder);
rangeMap = struct('M23', m23, 'M56', m56, 'CC', cc, 'DS', ds, 'VS', vs);

nAreas = numel(brainAreas);
areaChan = struct('areaName', {}, 'depthRange', {}, 'targetDepths', {}, ...
    'channelIds', {}, 'channelDepths', {}, 'depthSource', {}, 'chanMeta', {});

fprintf('Channel geometry: %s (%d sites). Depth source: %s\n', ...
    chanMeta.source, numel(channelIds), depthSource);

for iArea = 1:nAreas
    areaName = brainAreas{iArea};
    if ~isfield(rangeMap, areaName)
        error('select_area_lfp_channels:BadArea', ...
            'Unknown brain area "%s". Use M23, M56, CC, DS, or VS.', areaName);
    end
    depthRange = rangeMap.(areaName);
    dMin = depthRange(1);
    dMax = depthRange(2);
    inRange = orientedDepths >= dMin & orientedDepths <= dMax;
    ids = channelIds(inRange);
    deps = orientedDepths(inRange);
    if numel(ids) < 2
        error('select_area_lfp_channels:TooFew', ...
            '%s [%d %d] um has %d LFP channels; need at least 2.', ...
            areaName, dMin, dMax, numel(ids));
    end

    span = dMax - dMin;
    targetDepths = [dMin + span / 3, dMin + 2 * span / 3];
    pickedIds = zeros(1, 2);
    pickedDepths = zeros(1, 2);
    used = false(size(ids));
    for iTarget = 1:2
        dist = abs(deps - targetDepths(iTarget));
        dist(used) = inf;
        [~, idx] = min(dist);
        pickedIds(iTarget) = ids(idx);
        pickedDepths(iTarget) = deps(idx);
        used(idx) = true;
    end

    rec = struct();
    rec.areaName = areaName;
    rec.depthRange = depthRange;
    rec.targetDepths = targetDepths;
    rec.channelIds = pickedIds;
    rec.channelDepths = pickedDepths;
    rec.depthSource = depthSource;
    rec.chanMeta = chanMeta;
    areaChan(end+1) = rec; %#ok<AGROW>

    fprintf('  %s [%d %d] um  targets %.0f / %.0f  -> ch %d (%.0f um), ch %d (%.0f um)\n', ...
        areaName, dMin, dMax, targetDepths(1), targetDepths(2), ...
        pickedIds(1), pickedDepths(1), pickedIds(2), pickedDepths(2));
end
end

function brainAreas = normalize_area_names(brainAreas)
% NORMALIZE_AREA_NAMES - Cell of uppercase area labels

if ischar(brainAreas) || isstring(brainAreas)
    brainAreas = cellstr(brainAreas);
end
brainAreas = brainAreas(:)';
for iArea = 1:numel(brainAreas)
    brainAreas{iArea} = char(upper(brainAreas{iArea}));
end
end
