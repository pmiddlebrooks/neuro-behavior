function constants = np2_lfp_constants()
% NP2_LFP_CONSTANTS - Neuropixels 2.0 LFP file and ADC defaults
%
% Variables:
%   (none)
%
% Goal:
%   Describe the 2500 Hz int16 files written by extract_lfp_np2.sh.
%   Those files keep the wideband ADC codes. NP2.0 single- and multi-shank
%   probes use a 12-bit offset-binary ADC (midscale 2048) and a fixed gain
%   of 80. Electrode microvolts are 3.0517578125 * (sample - 2048), from
%   1.0 V peak-to-peak at the ADC divided by 4096 counts and by the gain.
%   Quad-base probes use a nominal gain of 100; change apGain for that probe.

constants = struct();
constants.nChannels = 384;
constants.fsRaw = 2500;
constants.fsOut = 1250;
constants.lowpassFreq = 300;
constants.extractLowpassHz = 500;
constants.bytesPerSample = 2;
constants.adcBits = 12;
constants.adcOffset = 2048;
constants.vPeakToPeak = 1.0;
constants.apGain = 80;
constants.uVPerBit = (constants.vPeakToPeak / (2^constants.adcBits) / constants.apGain) * 1e6;
constants.probePitchUm = 15;
constants.probeSpanUm = 3840;
constants.filePattern = 'np2-*-ephys_*_lfp.raw';
constants.outputFileName = 'lfp.mat';
constants.defaultBrainAreas = {'M23', 'M56', 'DS', 'VS'};
constants.defaultProbeId = 'a';
end
