function [m23, m56, cc, ds, vs, depthSource] = get_brain_area_depth_ranges(sessionFolder)
% GET_BRAIN_AREA_DEPTH_RANGES - Depth bounds per brain area for a session
%
% Variables:
%   sessionFolder - path to session directory (optional)
%
% Returns:
%   m23, m56, cc, ds, vs - 1x2 depth ranges [min max] in um from surface
%   depthSource        - 'session' if brain_area_depths.mat exists, else 'default'
%
% Goal:
%   Use session-specific depths when brain_area_depths.mat is present;
%   otherwise return the standard depth guide from load_data.m.

if nargin < 1
    sessionFolder = '';
end

depthMatPath = fullfile(sessionFolder, 'brain_area_depths.mat');
if ~isempty(sessionFolder) && isfile(depthMatPath)
    depthVars = load(depthMatPath, 'm23', 'm56', 'cc', 'ds', 'vs');
    m23 = depthVars.m23;
    m56 = depthVars.m56;
    cc = depthVars.cc;
    ds = depthVars.ds;
    vs = depthVars.vs;
    depthSource = 'session';
    return;
end

m23 = [0 500];
m56 = [501 1240];
cc = [1241 1540];
ds = [1541 2700];
vs = [2701 3840];
depthSource = 'default';
end
