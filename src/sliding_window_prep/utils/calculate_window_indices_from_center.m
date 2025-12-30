function [startIdx, endIdx] = calculate_window_indices_from_center(centerTime, slidingWindowSize, binSize, numTimePoints)
% CALCULATE_WINDOW_INDICES_FROM_CENTER Calculate window indices from center time
%
% Variables:
%   centerTime - Center time of window in seconds
%   slidingWindowSize - Window size in seconds
%   binSize - Bin size in seconds (for converting time to sample index)
%   numTimePoints - Total number of time points in data
%
% Goal:
%   Calculate the start and end indices for a sliding window given a center
%   time. This ensures all areas have aligned windows regardless of their
%   bin sizes. The centerTime is in absolute time (seconds), and indices are
%   calculated based on the area's specific binSize.
%
% Returns:
%   startIdx - Start index of window (1-indexed)
%   endIdx - End index of window (1-indexed)

    % Convert centerTime to a sample index (1-indexed)
    % centerTime is in seconds, binSize is in seconds
    % If bin 1 covers [0, binSize), then bin n covers [(n-1)*binSize, n*binSize)
    % So for time t, the bin index is: floor(t / binSize) + 1
    % But for centering, we want the bin closest to centerTime, so use round
    centerIdx = round(centerTime / binSize) + 1;
    
    % Calculate window size in samples
    winSamples = round(slidingWindowSize / binSize);
    if winSamples < 1
        winSamples = 1;
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

