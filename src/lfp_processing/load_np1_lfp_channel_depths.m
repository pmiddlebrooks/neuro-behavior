function [channelIds, orientedDepths, chanMeta] = load_np1_lfp_channel_depths(sessionFolder)
% LOAD_NP1_LFP_CHANNEL_DEPTHS - Raw LFP channel IDs and surface-referenced depths
%
% Variables:
%   sessionFolder - Spike session folder (optional). Used to read
%                   channel_map.npy / channel_positions.npy when present.
%
% Goal:
%   Return 0-based raw-file channel IDs and oriented depths matching
%   load_session_cluster_info.m: depth = 3840 - phy_y, so 0 is the top
%   surface (M23) and 3840 is deepest (VS). Skip the NP1 reference site
%   (channel 191) when it is absent from the kilosort map.

if nargin < 1
    sessionFolder = '';
end

chanMeta = struct();
chanMeta.depthConvention = 'oriented: 0 = surface, 3840 = deepest (3840 - phy_y)';
chanMeta.probePitchUm = 20;
chanMeta.nChannelsRaw = 384;
chanMeta.referenceChannelId = 191;

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
    orientedDepths = 3840 - phyY;
    chanMeta.source = 'channel_positions.npy';
else
    channelIds = (0:383)';
    channelIds(channelIds == chanMeta.referenceChannelId) = [];
    phyY = 20 * (floor(channelIds / 2) + 1);
    orientedDepths = 3840 - phyY;
    chanMeta.source = 'np1_geometry';
end

chanMeta.phyY = phyY;
end
