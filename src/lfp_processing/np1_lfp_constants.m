function constants = np1_lfp_constants()
% NP1_LFP_CONSTANTS - Neuropixels 1.0 Open Ephys LFP recording defaults
%
% Variables:
%   (none)
%
% Goal:
%   Return probe/file constants for np1-lfp_*.raw files written by the
%   Open Ephys Neuropixels plugin (384 ch, 2500 Hz, 10-bit ADC in uint16).

constants = struct();
constants.nChannels = 384;
constants.fsRaw = 2500;
constants.fsOut = 1000;
constants.lowpassFreq = 300;
constants.bytesPerSample = 2;
constants.adcBits = 10;
constants.adcOffset = 512;
constants.lfpGain = 250;
constants.vPeakToPeak = 1.2;
constants.uVPerBit = (constants.vPeakToPeak / (2^constants.adcBits) / constants.lfpGain) * 1e6;
constants.filePattern = 'np1-lfp_*.raw';
constants.outputFileName = 'lfp.mat';
constants.defaultBrainAreas = {'M23', 'M56', 'DS', 'VS'};
end
