function [startIdx, endIdx, centerTime, numWindows] = calculate_window_indices(w, numTimePoints, winSamples, stepSamples, binSize)
% CALCULATE_WINDOW_INDICES Calculate window indices and center time
%
% Variables:
%   w - Current window number (1-indexed)
%   numTimePoints - Total number of time points in data
%   winSamples - Window size in samples
%   stepSamples - Step size in samples
%   binSize - Bin size in seconds (for calculating center time)
%
% Goal:
%   Calculate the start and end indices for a sliding window, along with
%   the center time of the window. Used across all sliding window analyses.
%
% Returns:
%   startIdx - Start index of window (1-indexed)
%   endIdx - End index of window (1-indexed)
%   centerTime - Center time of window in seconds
%   numWindows - Total number of windows (calculated from inputs)

    % Calculate total number of windows
    numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
    
    % Calculate window indices
    startIdx = (w - 1) * stepSamples + 1;
    endIdx = startIdx + winSamples - 1;
    
    % Calculate center time
    centerTime = (startIdx + round(winSamples/2) - 1) * binSize;
end

