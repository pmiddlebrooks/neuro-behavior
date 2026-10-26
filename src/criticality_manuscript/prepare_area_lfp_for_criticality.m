function [dataStruct, areaLabel, collectStart, collectEnd] = prepare_area_lfp_for_criticality( ...
    dataStruct, brainArea, combinations, collectStart, collectEnd, bands, lfpCleanParams, rawD2BinSize, analyzeBandPower)
% PREPARE_AREA_LFP_FOR_CRITICALITY - One area, bandpass, zero-mean, re-binned
%
% Variables:
%   dataStruct     - LFP session from load_session_data. Artifact cleaning is
%                    skipped when lfpCleanParams.skipArtifactClean is true.
%   brainArea      - Requested area or combined-area name
%   combinations   - Cell of struct('name', ..., 'areas', {{...}})
%   collectStart   - Requested start (s)
%   collectEnd     - Requested end (s); [] = end of the trace after trim rules
%   bands          - Frequency bands for lfp_bin_bandpower
%   lfpCleanParams - highpassFreq and lowpassFreq for the analysis bandpass
%   rawD2BinSize   - Bin width (s) stored as dataStruct.lfpBinSize for raw d2
%   analyzeBandPower - If false, skip binned band-power envelopes (default true)
%
% Goal:
%   Restrict LFP to the requested area (average channels for a combined area),
%   keep the collect window, bandpass, subtract the mean so the Klaus threshold
%   is about zero, and recompute binned band power on that trace.

if ~isfield(dataStruct, 'lfpPerArea') || isempty(dataStruct.lfpPerArea)
    error('session_lfp_criticality:NoLfp', 'Loaded session has no lfpPerArea.');
end
if ~isfield(dataStruct, 'opts') || ~isfield(dataStruct.opts, 'fsLfp') || isempty(dataStruct.opts.fsLfp)
    error('session_lfp_criticality:NoFs', 'opts.fsLfp is missing.');
end

if nargin < 9 || isempty(analyzeBandPower)
    analyzeBandPower = true;
end
analyzeBandPower = logical(analyzeBandPower);

fs = dataStruct.opts.fsLfp;
rawD2BinSize = snap_lfp_bin_to_samples(rawD2BinSize, fs);
fprintf('Raw LFP d2 bin: %.2f ms (%d samples at %.0f Hz)\n', ...
    rawD2BinSize * 1000, round(rawD2BinSize * fs), fs);
[trace, areaLabel] = select_lfp_area_trace(dataStruct.lfpPerArea, dataStruct.areas, ...
    brainArea, combinations);
[trace, collectStart, collectEnd] = trim_lfp_collect_window(trace, fs, ...
    dataStruct.sessionType, collectStart, collectEnd);
trace = bandpass_analysis_lfp(trace, fs, lfpCleanParams);
trace = zero_mean_lfp(trace);

dataStruct.lfpPerArea = trace;
dataStruct.areas = {areaLabel};
dataStruct.areasToTest = 1;
dataStruct.opts.collectStart = collectStart;
dataStruct.opts.collectEnd = collectEnd;
dataStruct.opts.fsLfp = fs;
if ~isfield(dataStruct, 'dataType') || isempty(dataStruct.dataType)
    dataStruct.dataType = dataStruct.sessionType;
end
if analyzeBandPower
    dataStruct.bands = bands;
    dataStruct = compute_lfp_binned_envelopes(dataStruct, dataStruct.opts, lfpCleanParams, bands);
else
    dataStruct.bands = cell(0, 2);
    dataStruct.bandBinSizes = [];
    dataStruct.binnedPower = {};
    dataStruct.binnedEnvelopes = {};
    dataStruct.timePoints = {};
end
dataStruct.lfpBinSize = rawD2BinSize;
end

function [trace, areaLabel] = select_lfp_area_trace(lfpPerArea, areaNames, brainArea, combinations)
% SELECT_LFP_AREA_TRACE - One column, or the mean of a combined area
%
% Variables:
%   lfpPerArea   - [samples x areas] LFP matrix
%   areaNames    - Cell of area names, one per column
%   brainArea    - Requested name
%   combinations - Combined-area definitions
%
% Goal:
%   Return the LFP used for this analysis. Combined names (for example M23M56)
%   average the source columns.

brainArea = char(brainArea);
combo = lookup_lfp_area_combination(brainArea, combinations);
if ~isempty(combo)
    sourceIdx = zeros(1, numel(combo.areas));
    for iArea = 1:numel(combo.areas)
        sourceIdx(iArea) = find(strcmp(areaNames, combo.areas{iArea}), 1);
        if isempty(sourceIdx(iArea))
            error('session_lfp_criticality:MissingSourceArea', ...
                'Combined area %s needs %s, which is not in this session.', ...
                combo.name, combo.areas{iArea});
        end
    end
    trace = mean(lfpPerArea(:, sourceIdx), 2);
    areaLabel = combo.name;
    fprintf('  LFP area %s = mean of %s\n', areaLabel, strjoin(combo.areas, '+'));
    return;
end

areaIdx = find(strcmp(areaNames, brainArea), 1);
if isempty(areaIdx)
    error('session_lfp_criticality:MissingArea', ...
        'Brain area "%s" is not in this session (%s).', ...
        brainArea, strjoin(areaNames, ', '));
end
trace = lfpPerArea(:, areaIdx);
areaLabel = areaNames{areaIdx};
fprintf('  Restricting LFP to area: %s\n', areaLabel);
end

function combo = lookup_lfp_area_combination(brainArea, combinations)
% LOOKUP_LFP_AREA_COMBINATION - Match brainArea to a merged-area definition

combo = [];
if isempty(combinations)
    return;
end
for iCombo = 1:numel(combinations)
    entry = combinations{iCombo};
    if isstruct(entry) && isfield(entry, 'name') && strcmpi(entry.name, brainArea)
        combo = entry;
        combo.areas = cellstr(combo.areas);
        return;
    end
end
end

function [trace, collectStart, collectEnd] = trim_lfp_collect_window( ...
    trace, fs, sessionType, collectStart, collectEnd)
% TRIM_LFP_COLLECT_WINDOW - Keep samples inside the requested collect window
%
% Variables:
%   trace        - LFP column
%   fs           - Sampling rate (Hz)
%   sessionType  - Reach LFP is loaded from t = 0; other loaders already
%                  slice load_data from collectStart, so sample 1 is collectStart
%   collectStart - Requested start (s)
%   collectEnd   - Requested end (s); [] = last sample
%
% Goal:
%   Return the analysis segment and the absolute times of its first and last
%   samples. collectEnd is filled in when the caller left it empty.

if nargin < 4 || isempty(collectStart)
    collectStart = 0;
end
nSamples = size(trace, 1);
if strcmpi(sessionType, 'reach')
    signalStart = 0;
else
    signalStart = collectStart;
end

i0 = 1 + round((collectStart - signalStart) * fs);
if isempty(collectEnd)
    i1 = nSamples;
else
    i1 = round((collectEnd - signalStart) * fs);
end
i0 = max(1, min(nSamples, i0));
i1 = max(i0, min(nSamples, i1));
trace = trace(i0:i1, :);
collectStart = signalStart + (i0 - 1) / fs;
collectEnd = signalStart + i1 / fs;
end

function trace = bandpass_analysis_lfp(trace, fs, lfpCleanParams)
% BANDPASS_ANALYSIS_LFP - Zero-phase bandpass of one analysis trace
%
% Variables:
%   trace          - LFP column after the collect window is applied
%   fs             - Sampling rate (Hz)
%   lfpCleanParams - .highpassFreq and .lowpassFreq (Hz)
%
% Goal:
%   Limit the trace to the Klaus band (default 1-100 Hz). process_np1_lfp
%   already lowpassed at 300 Hz. This does not blank peaks or interpolate.

highpassHz = 1;
lowpassHz = 100;
if isstruct(lfpCleanParams)
    if isfield(lfpCleanParams, 'highpassFreq') && ~isempty(lfpCleanParams.highpassFreq)
        highpassHz = lfpCleanParams.highpassFreq(1);
    end
    if isfield(lfpCleanParams, 'lowpassFreq') && ~isempty(lfpCleanParams.lowpassFreq)
        lowpassHz = lfpCleanParams.lowpassFreq(1);
    end
end
if ~(highpassHz > 0 && lowpassHz > highpassHz && lowpassHz < fs / 2)
    error('session_lfp_criticality:BadBand', ...
        'Need 0 < highpass (%.3g) < lowpass (%.3g) < Nyquist (%.3g).', ...
        highpassHz, lowpassHz, fs / 2);
end
trace = bandpass(double(trace(:)), [highpassHz, lowpassHz], fs);
fprintf('  Bandpass %.0f-%.0f Hz\n', highpassHz, lowpassHz);
end

function trace = zero_mean_lfp(trace)
% ZERO_MEAN_LFP - Drop non-finite samples' contribution and subtract the mean
%
% Variables:
%   trace - LFP column after the analysis bandpass
%
% Goal:
%   The analysis window is zero-mean so the 2.5 SD avalanche cutoff is
%   symmetric about zero.

trace = trace(:, 1);
bad = ~isfinite(trace);
if any(bad)
    trace(bad) = 0;
end
trace = trace - mean(trace);
end
