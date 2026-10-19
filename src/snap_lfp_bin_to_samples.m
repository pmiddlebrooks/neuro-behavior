function binSec = snap_lfp_bin_to_samples(targetSec, fs)
% SNAP_LFP_BIN_TO_SAMPLES - Bin width that is an integer number of samples
%
% Variables:
%   targetSec - Requested bin width (seconds)
%   fs        - Sampling rate (Hz)
%
% Goal:
%   Mean-binning uses whole samples. A 5 ms target is 5 samples at 1000 Hz
%   and 6.25 samples at 1250 Hz, so the realized width is the nearest
%   whole-sample duration.
%
% Returns:
%   binSec - Realized bin width (seconds)

nPerBin = max(1, round(targetSec * fs));
binSec = nPerBin / fs;
end
