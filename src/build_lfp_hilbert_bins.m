function [binnedMat, binCenterSec] = build_lfp_hilbert_bins(signalColumn, fs, bandsCell, binSizeSec, filterOrder)
% BUILD_LFP_HILBERT_BINS Band-pass LFP, Hilbert envelope, mean in non-overlapping bins.
%
% Variables:
%   - signalColumn: LFP samples [nSamples x 1] or row vector
%   - fs: sample rate [Hz]
%   - bandsCell: Nx2 cell array {name, [fLow fHigh]} per row [Hz]
%   - binSizeSec: bin width [s] (non-overlapping)
%   - filterOrder: Butterworth bandpass order (e.g. 11, Akella et al.)
% Goal:
%   Build Akella-style HMM inputs: envelope power in fixed spectral bands at bin resolution.
% Returns:
%   - binnedMat: [nBins x numBands] mean envelope per bin per band
%   - binCenterSec: [nBins x 1] bin center times from start of signal [s]

signalColumn = double(signalColumn(:));
if any(~isfinite(signalColumn))
    validIdx = isfinite(signalColumn);
    if nnz(validIdx) < 2
        binnedMat = zeros(0, size(bandsCell, 1));
        binCenterSec = zeros(0, 1);
        warning('build_lfp_hilbert_bins:invalidSignal', ...
            'Signal has fewer than 2 finite samples; returning empty output.');
        return;
    end
    % Fill rare non-finite points so filtering/Hilbert remain stable.
    signalColumn = interp1(find(validIdx), signalColumn(validIdx), ...
        (1:numel(signalColumn))', 'linear', 'extrap');
end

signalColumn = detrend(signalColumn, 'constant');
numBands = size(bandsCell, 1);
binLen = max(1, round(binSizeSec * fs));

bandCols = cell(numBands, 1);
minBins = inf;
for bandIdx = 1:numBands
    freqRange = bandsCell{bandIdx, 2};
    lowHz = freqRange(1);
    highHz = freqRange(2);
    nyq = fs / 2;
    assert(highHz < nyq, 'Band high frequency must be below Nyquist (%.1f Hz)', nyq);

    wn = [lowHz highHz] / nyq;
    assert(wn(1) > 0 && wn(2) < 1 && wn(1) < wn(2), ...
        'Invalid normalized band edges: [%.5f %.5f].', wn(1), wn(2));

    % Use SOS form for numerical stability on narrow/high-order bands.
    filteredLfp = [];
    orderTry = unique([filterOrder, 9, 7, 5], 'stable');
    for ord = orderTry
        try
            [z, p, k] = butter(ord, wn, 'bandpass');
            [sos, g] = zp2sos(z, p, k);
            iFiltered = filtfilt(sos, g, signalColumn);
            if all(isfinite(iFiltered))
                filteredLfp = iFiltered;
                break;
            end
        catch
            % Keep trying lower order.
        end
    end

    if isempty(filteredLfp)
        warning('build_lfp_hilbert_bins:filterFailure', ...
            ['Band %s [%g %g] Hz produced non-finite output for all tried orders. ' ...
             'Returning zeros for this band.'], bandsCell{bandIdx, 1}, lowHz, highHz);
        nBin = floor(numel(signalColumn) / binLen);
        bandCols{bandIdx} = zeros(max(nBin, 0), 1);
        minBins = min(minBins, nBin);
        continue;
    end

    envelope = abs(hilbert(filteredLfp));
    if any(~isfinite(envelope))
        warning('build_lfp_hilbert_bins:hilbertNonFinite', ...
            'Non-finite Hilbert envelope in band %s; replacing with 0.', bandsCell{bandIdx, 1});
        envelope(~isfinite(envelope)) = 0;
    end

    nSamp = numel(envelope);
    nBin = floor(nSamp / binLen);
    if nBin < 1
        bandCols{bandIdx} = zeros(0, 1);
        minBins = 0;
        continue;
    end
    trimmed = envelope(1:nBin * binLen);
    binnedBand = mean(reshape(trimmed, binLen, nBin), 1)';
    bandCols{bandIdx} = binnedBand;
    minBins = min(minBins, nBin);
end

if ~isfinite(minBins) || minBins < 1
    binnedMat = zeros(0, numBands);
    binCenterSec = zeros(0, 1);
    return;
end

binnedMat = zeros(minBins, numBands);
for bandIdx = 1:numBands
    binnedMat(:, bandIdx) = bandCols{bandIdx}(1:minBins);
end

binCenterSec = ((1:minBins)' - 0.5) * binSizeSec;
end
