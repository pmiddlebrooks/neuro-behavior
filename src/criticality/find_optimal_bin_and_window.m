% ===================== Helper function for optimal bin/frame and window size =====================
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