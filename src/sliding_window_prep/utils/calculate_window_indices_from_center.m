function [startIdx, endIdx] = calculate_window_indices_from_center(centerTime, slidingWindowSize, binSize, numTimePoints, timeOrigin)
% CALCULATE_WINDOW_INDICES_FROM_CENTER Calculate window indices from center time
%
% Variables:
%   centerTime - Center time of window in absolute session seconds
%   slidingWindowSize - Window size in seconds
%   binSize - Bin size in seconds (for converting time to sample index)
%   numTimePoints - Total number of time points in data
%   timeOrigin - Absolute session time of matrix bin 1 (default 0). Relative
%                indexing uses (centerTime - timeOrigin).
%
% Goal:
%   Calculate the start and end indices for a sliding window given a center
%   time. This ensures all areas have aligned windows regardless of their
%   bin sizes. Indices are relative to the binned matrix (bin 1 = timeOrigin).
%
% Returns:
%   startIdx - Start index of window (1-indexed)
%   endIdx - End index of window (1-indexed)

    if nargin < 5 || isempty(timeOrigin)
        timeOrigin = 0;
    end

    % Convert absolute centerTime to time relative to matrix start
    relativeCenter = centerTime - timeOrigin;

    % Convert relative center to a sample index (1-indexed)
    % If bin 1 covers [0, binSize), then bin n covers [(n-1)*binSize, n*binSize)
    % For centering, use round to the nearest bin
    centerIdx = round(relativeCenter / binSize) + 1;
    
    % Calculate window size in samples
    winSamples = round(slidingWindowSize / binSize);
    if winSamples < 1
        winSamples = 1;
    end
    % Full-session / oversized window: use entire recording
    if winSamples >= numTimePoints
        startIdx = 1;
        endIdx = numTimePoints;
        return;
    end

    % Calculate start and end indices centered around centerIdx
    halfWin = round(winSamples / 2);
    startIdx = centerIdx - halfWin + 1;
    endIdx = startIdx + winSamples - 1;
    
    % Ensure indices are within bounds (clamp to valid range)
    if startIdx < 1
        startIdx = 1;
        endIdx = winSamples;
    end
    if endIdx > numTimePoints
        endIdx = numTimePoints;
        startIdx = endIdx - winSamples + 1;
    end
    
    % Final bounds check
    if startIdx < 1
        startIdx = 1;
    end
    if endIdx > numTimePoints
        endIdx = numTimePoints;
    end
end
