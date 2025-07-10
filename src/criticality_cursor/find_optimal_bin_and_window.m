% FIND_OPTIMAL_BIN_AND_WINDOW  Find optimal bin/frame and window size for population activity analysis
%
%   [optimalBinSize, optimalWindowSize] = find_optimal_bin_and_window(dataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow)
%
%   This function searches for the optimal bin (frame) size and window size for population activity analysis.
%   The optimal parameters are those that yield a mean spike count per bin within the specified range and
%   a minimum number of bins per window.
%
%   Inputs:
%       dataMat             - [time x neurons] matrix of spike counts
%       candidateFrameSizes - array of candidate bin/frame sizes (in seconds)
%       candidateWindowSizes- array of candidate window sizes (in seconds)
%       minSpikesPerBin     - minimum mean spikes per bin
%       maxSpikesPerBin     - maximum mean spikes per bin
%       minBinsPerWindow    - minimum number of bins per window
%
%   Outputs:
%       optimalBinSize      - optimal bin/frame size (in seconds)
%       optimalWindowSize   - optimal window size (in seconds)
%
%   The function returns 0 for both outputs if no suitable parameters are found.

function [optimalBinSize, optimalWindowSize] = find_optimal_bin_and_window(dataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow)
    optimalBinSize = 0;
    optimalWindowSize = 0;
    found = false;
    for fs = candidateFrameSizes
        aDataMatR = neural_matrix_ms_to_frames(dataMat, fs);
        meanSpikesPerBin = mean(sum(aDataMatR, 2));
        if meanSpikesPerBin < minSpikesPerBin || meanSpikesPerBin > maxSpikesPerBin
            continue;
        end
        for ws = candidateWindowSizes
            numBins = ws / fs;
            if numBins >= minBinsPerWindow
                optimalBinSize = fs;
                optimalWindowSize = ws;
                found = true;
                break;
            end
        end
        if found
            break;
        end
    end
end