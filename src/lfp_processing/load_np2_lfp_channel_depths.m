function [channelIds, orientedDepths, chanMeta] = load_np2_lfp_channel_depths(sessionFolder, nChannels)
% LOAD_NP2_LFP_CHANNEL_DEPTHS - Raw NP2 channel IDs and surface-referenced depths
%
% Variables:
%   sessionFolder - Spike session folder (optional). Reads channel_map.npy
%                   and channel_positions.npy when both are present.
%   nChannels     - Channels in the raw frame (default 384)
%
% Goal:
%   Return 0-based raw-file channel IDs and oriented depths with the same
%   convention as the spike pipeline: depth = 3840 - phy_y, so 0 is the
%   surface (M23) and 3840 is deepest (VS). Without a channel map, assume a
%   single-shank bank whose rows start at the tip: pair ch 0-1 is deepest,
%   vertical pitch is 15 um, and the two columns share a y.

if nargin < 1
    sessionFolder = '';
end
constants = np2_lfp_constants();
if nargin < 2 || isempty(nChannels)
    nChannels = constants.nChannels;
end

chanMeta = struct();
chanMeta.depthConvention = 'oriented: 0 = surface, 3840 = deepest (3840 - phy_y)';
chanMeta.probePitchUm = constants.probePitchUm;
chanMeta.nChannelsRaw = nChannels;
chanMeta.referenceChannelId = [];
chanMeta.probeSpanUm = constants.probeSpanUm;

mapPath = fullfile(sessionFolder, 'channel_map.npy');
posPath = fullfile(sessionFolder, 'channel_positions.npy');
if ~isempty(sessionFolder) && isfile(mapPath) && isfile(posPath)
    if exist('readNPY', 'file') ~= 2
        paths = get_paths;
        addpath(fullfile(paths.homePath, 'toolboxes', 'npy-matlab', 'npy-matlab'));
    end
    channelIds = double(readNPY(mapPath));
    channelIds = channelIds(:);
    xy = double(readNPY(posPath));
    phyY = xy(:, 2);
    phyY = phyY(:);
    nKeep = min(numel(channelIds), numel(phyY));
    channelIds = channelIds(1:nKeep);
    phyY = phyY(1:nKeep);
    chanMeta.source = 'channel_positions.npy';
else
    if nChannels ~= 384
        error('load_np2_lfp_channel_depths:NeedMap', ...
            ['A %d-channel NP2 file needs channel_map.npy and channel_positions.npy ', ...
            'in the session folder.'], nChannels);
    end
    channelIds = (0:nChannels - 1)';
    phyY = constants.probePitchUm * floor(channelIds / 2);
    chanMeta.source = 'np2_bank_from_tip';
end

orientedDepths = constants.probeSpanUm - phyY;
chanMeta.phyY = phyY;
fprintf('NP2 depths from %s: %.0f to %.0f um (0 = surface).\n', ...
    chanMeta.source, min(orientedDepths), max(orientedDepths));
end
