function [optimalBinSize, optimalWindowSize] = find_optimal_bin_and_window(firingRate, minSpikesPerBin, minBinsPerWindow)

% FIND_OPTIMAL_BIN_AND_WINDOW  Find optimal bin/frame and window size for population activity analysis
%
%   [optimalBinSize, optimalWindowSize] = find_optimal_bin_and_window(firingRate, minSpikesPerBin, minBinsPerWindow)
%
%   This function calculates the optimal bin (frame) size and window size for population activity analysis
%   based on a given firing rate. The bin size is chosen to guarantee an average of at least minSpikesPerBin
%   spikes per bin, rounded up to the nearest 5 ms. The window size is then calculated to contain at least
%   minBinsPerWindow bins.
%
%   Inputs:
%       firingRate          - Firing rate in spikes per second
%       minSpikesPerBin     - Minimum average number of spikes per bin required
%       minBinsPerWindow    - Minimum number of bins per window
%
%   Outputs:
%       optimalBinSize      - Optimal bin/frame size (in seconds), rounded up to nearest 5 ms
%       optimalWindowSize   - Optimal window size (in seconds), rounded to nearest 5 ms
%
%   Algorithm:
%       - Calculates minimum bin size: minBinSize = minSpikesPerBin / firingRate
%       - Rounds up to nearest 5 ms: optimalBinSize = ceil(minBinSize / 0.005) * 0.005
%       - Calculates window size: optimalWindowSize = optimalBinSize * minBinsPerWindow
%       - Rounds window size to nearest 5 ms

% Calculate minimum bin size needed to guarantee minSpikesPerBin
% Goal: firingRate * binSize >= minSpikesPerBin
% Therefore: binSize >= minSpikesPerBin / firingRate
minBinSize = minSpikesPerBin / firingRate;

% Round up to nearest 5 ms (0.005 seconds)
optimalBinSize = ceil(minBinSize / 0.005) * 0.005;

% Calculate window size based on minimum bins per window
optimalWindowSize = optimalBinSize * minBinsPerWindow;

% Round window size to nearest 5 ms
optimalWindowSize = round(optimalWindowSize / 0.005) * 0.005;

% Ensure window size is at least equal to bin size
if optimalWindowSize < optimalBinSize
    optimalWindowSize = optimalBinSize;
end
end