function dataStruct = compute_lfp_binned_envelopes(dataStruct, opts, lfpCleanParams, bands)
% COMPUTE_LFP_BINNED_ENVELOPES Compute binned LFP envelopes for all bands
%
% Variables:
%   dataStruct - Data structure with lfpPerArea, areas, dataType
%   opts - Options structure with fsLfp
%   lfpCleanParams - LFP cleaning parameters (not used here but kept for consistency)
%   bands - Frequency bands cell array
%
% Goal:
%   Compute frequency-dependent binned envelopes for LFP data. Each band
%   gets its own bin size proportional to frequency.
%
% Returns:
%   dataStruct - Updated with binnedEnvelopes, bandBinSizes, etc.

    % Minimum bin size (for highest frequency band)
    minBinSize = 0.005;  % 5 ms
    
    % Compute frequency-dependent bin sizes for each band
    numBands = size(bands, 1);
    bandAvgFreqs = zeros(1, numBands);
    for b = 1:numBands
        bandAvgFreqs(b) = mean(bands{b, 2});  % Average of frequency range
    end
    
    % Find highest frequency band
    [~, highestFreqIdx] = max(bandAvgFreqs);
    highestFreqAvg = bandAvgFreqs(highestFreqIdx);
    
    % Calculate bin size for each band: proportional to frequency
    bandBinSizes = zeros(1, numBands);
    for b = 1:numBands
        calculatedBinSize = minBinSize * (highestFreqAvg / bandAvgFreqs(b));
        bandBinSizes(b) = ceil(calculatedBinSize / minBinSize) * minBinSize;
    end
    
    dataStruct.bandBinSizes = bandBinSizes;
    
    fprintf('Frequency-dependent bin sizes:\n');
    for b = 1:numBands
        fprintf('  %s (%.1f Hz avg): %.3f s (%.1f ms)\n', ...
            bands{b,1}, bandAvgFreqs(b), bandBinSizes(b), bandBinSizes(b)*1000);
    end
    fprintf('  Max bin size: %.3f s (%.1f ms)\n', max(bandBinSizes), max(bandBinSizes)*1000);
    
    % Compute binned envelopes for each area
    numAreas = size(dataStruct.lfpPerArea, 2);
    
    if strcmp(dataStruct.dataType, 'schall')
        % Schall: FEF is single area, but may have multiple channels
        if numAreas == 1
            % Single channel - use it directly
            binnedEnvelopes = {cell(1, numBands)};
            binnedPower = {cell(1, numBands)};
            timePoints = {cell(1, numBands)};
            for b = 1:numBands
                singleBand = bands(b, :);
                [iBinnedPower, iBinnedEnvelopes, iTimePoints] = lfp_bin_bandpower(...
                    dataStruct.lfpPerArea, opts.fsLfp, singleBand, bandBinSizes(b), 'cwt');
                binnedEnvelopes{1}{b} = iBinnedEnvelopes';
                binnedPower{1}{b} = iBinnedPower';
                timePoints{1}{b} = iTimePoints(:);
            end
        else
            % Multiple channels - average them
            lfpMean = mean(dataStruct.lfpPerArea, 2);
            binnedEnvelopes = {cell(1, numBands)};
            binnedPower = {cell(1, numBands)};
            timePoints = {cell(1, numBands)};
            for b = 1:numBands
                singleBand = bands(b, :);
                [iBinnedPower, iBinnedEnvelopes, iTimePoints] = lfp_bin_bandpower(...
                    lfpMean, opts.fsLfp, singleBand, bandBinSizes(b), 'cwt');
                binnedEnvelopes{1}{b} = iBinnedEnvelopes';
                binnedPower{1}{b} = iBinnedPower';
                timePoints{1}{b} = iTimePoints(:);
            end
        end
        fprintf('LFP data loaded: %d bands (each with different frame counts)\n', numBands);
    else
        % Naturalistic and Reach: multiple areas
        binnedEnvelopes = cell(1, numAreas);
        binnedPower = cell(1, numAreas);
        timePoints = cell(1, numAreas);
        for iArea = 1:numAreas
            binnedEnvelopes{iArea} = cell(1, numBands);
            binnedPower{iArea} = cell(1, numBands);
            timePoints{iArea} = cell(1, numBands);
            for b = 1:numBands
                singleBand = bands(b, :);
                [iBinnedPower, iBinnedEnvelopes, iTimePoints] = lfp_bin_bandpower(...
                    dataStruct.lfpPerArea(:, iArea), opts.fsLfp, singleBand, bandBinSizes(b), 'cwt');
                binnedEnvelopes{iArea}{b} = iBinnedEnvelopes';
                binnedPower{iArea}{b} = iBinnedPower';
                timePoints{iArea}{b} = iTimePoints(:);
            end
        end
        fprintf('LFP data loaded: %d areas, %d bands/area (each band with different frame counts)\n', numAreas, numBands);
    end
    
    dataStruct.binnedEnvelopes = binnedEnvelopes;
    dataStruct.binnedPower = binnedPower;
    dataStruct.timePoints = timePoints;
    dataStruct.lfpBinSize = 0.005;  % 200 Hz
end
