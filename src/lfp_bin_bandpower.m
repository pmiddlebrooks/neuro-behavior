function [binnedPower, binnedEnvelopes, timeBins] = lfp_bin_bandpower(signal, fs, bands, frameSize, method)
% lfp_bin_bandpower: Computes band powers and envelopes of LFP signals,
% bins them by frames, and returns results.
% Envelopes are computed directly from bandpass-filtered LFP signals.
% The signal is padded before filtering to minimize edge effects at the
% beginning and end of the signal.
%
% Inputs:
%   - signal: LFP signal (1D array).
%   - fs: Sampling rate (Hz).
%   - bands: Cell array of frequency bands (e.g., {'alpha', [8 13]; 'beta', [13 30]}).
%   - frameSize: Frame size in seconds for binning.
%   - method: (Optional, deprecated) Method parameter kept for backward compatibility.
%
% Outputs:
%   - binnedPower: Band power binned by frames (computed from envelope^2).
%     Size: [numBands x numFrames]. Note: numFrames may be less than
%     signal duration/frameSize if signal length is not perfectly divisible.
%   - binnedEnvelopes: Band-specific envelopes binned by frames (computed from filtered LFP).
%     Size: [numBands x numFrames].
%   - timeBins: Time vector (in seconds) for bin midpoints. Length: numFrames.
%     Use this to align binned outputs with the original signal timeline.

if nargin < 5
    method = 'cwt';
end
% Frame parameters
frameSamples = round(frameSize * fs); % Samples per frame
numFrames = floor(length(signal) / frameSamples);
% Include partial final frame if there are remaining samples
remainingSamples = length(signal) - numFrames * frameSamples;
if remainingSamples > 0
    numFrames = numFrames + 1;
end
numBands = size(bands, 1);

signal = zscore(signal, [], 1);
% Preallocate binned outputs
binnedPower = zeros(numBands, numFrames);
binnedEnvelopes = zeros(numBands, numFrames);

% Preallocate matrices for band power and envelopes
bandPowerSignals = zeros(numBands, length(signal));
bandEnvelopes = zeros(numBands, length(signal));

% Step 1: For each band, bandpass filter the signal and compute envelopes
% Estimate filter order to determine padding needed (to minimize edge effects)
% Use a conservative estimate based on the lowest frequency band
minFreq = min(cellfun(@(x) x(1), bands(:, 2)));
% Estimate filter transition bandwidth (conservative: 10% of lower frequency)
transitionBW = minFreq * 0.1;
% Estimate filter order (rough approximation for IIR filters)
filterOrder = round(fs / transitionBW);
% Padding length: ~3-5 filter periods to allow filter to settle
% Cap at 10% of signal length and ensure we don't pad more than signal length
padLength = min([round(5 * filterOrder), round(length(signal) * 0.1), floor(length(signal) / 3)]);

for i = 1:numBands
    % Get the frequency range for the current band
    freqRange = bands{i, 2};
    
    % Pad signal at beginning and end to minimize edge effects
    % Use reflection padding (mirror) which is better than zero-padding for filtering
    if padLength > 0 && padLength < length(signal)
        % Ensure signal is column vector for consistent indexing
        signalCol = signal(:);
        signalPadded = [flipud(signalCol(1:padLength)); signalCol; flipud(signalCol(end-padLength+1:end))];
        
        % Bandpass filter the padded signal
        filteredSignalPadded = bandpass(signalPadded, freqRange, fs);
        
        % Remove padding to get filtered signal of original length
        filteredSignal = filteredSignalPadded(padLength+1:end-padLength);
    else
        % If padding would be too large relative to signal, filter without padding
        filteredSignal = bandpass(signal, freqRange, fs);
    end
    
    % Compute envelope directly from the filtered signal using Hilbert transform
    % Ensure filteredSignal is row vector to match preallocated matrix orientation
    filteredSignal = filteredSignal(:)'; % Convert to row vector
    bandEnvelopes(i, :) = abs(hilbert(filteredSignal));
    
    % Compute power from the envelope (envelope^2) or from filtered signal squared
    bandPowerSignals(i, :) = bandEnvelopes(i, :).^2;
end

% Step 2: Bin power and envelopes by frames
% Preallocate time bins
timeBins = zeros(1, numFrames);

for frameIdx = 1:numFrames
    % Frame indices
    startIdx = (frameIdx - 1) * frameSamples + 1;
    endIdx = min(startIdx + frameSamples - 1, length(signal)); % Handle partial final frame
    
    % Extract frame
    framePower = bandPowerSignals(:, startIdx:endIdx);
    frameEnvelope = bandEnvelopes(:, startIdx:endIdx);
    
    % Average across the frame
    binnedPower(:, frameIdx) = mean(framePower, 2);
    binnedEnvelopes(:, frameIdx) = mean(frameEnvelope, 2);
    
    % Compute bin midpoint time
    timeBins(frameIdx) = (startIdx + endIdx) / (2 * fs);
end

end
