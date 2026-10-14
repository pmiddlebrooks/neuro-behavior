%%
% Session LFP power and population spiking
%
% For one spontaneous session and one brain area, plot a sliding window of
% population spike rate above the LFP band powers. This is the first view
% for marking sleep before those epochs are dropped from criticality.
%
% Band power uses the same binned envelopes as the LFP criticality loader
% (lfp_bin_bandpower: broadband z-score, bandpass, Hilbert envelope squared).
% Each plotted point is the mean of those bins inside a window. Windows
% share one set of center times, as in criticality_lfp_analysis.
%
% Variables (session identity comes from the workspace):
%   sessionType  - 'spontaneous'
%   sessionName  - Session folder
%   subjectName  - Subject folder
%   collectStart - Window start (s from session onset)
%   collectEnd   - Window end (s); [] = through the spikes, then the LFP
%   brainArea    - Area name shared by spikes and LFP (e.g. 'M56')
%   windowSec    - Sliding window length (s)
%   stepSec      - Step between window centers (s)
%   bands        - Band name and [fLow fHigh] rows passed to the LFP loader
%
% Goal:
%   2x1 figure. Top: population spike rate. Bottom: log10 band power, one
%   line per band, on the same time axis.

%% Configuration
collectStart = 0;
collectEnd = [];

brainArea = 'M56';
windowSec = 3;
stepSec = .5;

bands = {'delta', [0.5 4]; ...
    'theta', [4 8]; ...
    'alpha', [8 13]; ...
    'beta', [13 30]; ...
    'lowGamma', [30 50]; ...
    'highGamma', [50 80]};

if ~exist('sessionType', 'var') || isempty(sessionType) ...
        || ~exist('sessionName', 'var') || isempty(sessionName)
    error('session_lfp_power:MissingSession', ...
        'Set sessionType and sessionName in the workspace before running.');
end
if windowSec <= 0 || stepSec <= 0 || stepSec > windowSec
    error('session_lfp_power:BadWindow', ...
        'Need 0 < stepSec <= windowSec.');
end

subjectNameForLoad = '';
if exist('subjectName', 'var') && ~isempty(subjectName)
    subjectNameForLoad = subjectName;
end

opts = neuro_behavior_options();
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.firingRateCheckTime = [];
opts.useSpikeTimes = true;

fprintf('\n=== Session LFP power ===\n');
fprintf('Session [%s]: %s\n', sessionType, sessionName);
fprintf('Area: %s\n', brainArea);
fprintf('Window %.1f s, step %.1f s\n', windowSec, stepSec);

loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
spikeStruct = load_session_data(sessionType, 'spikes', loadArgs{:});
if isfield(spikeStruct, 'opts')
    if isfield(spikeStruct.opts, 'collectStart') && ~isempty(spikeStruct.opts.collectStart)
        opts.collectStart = spikeStruct.opts.collectStart;
    end
    if isfield(spikeStruct.opts, 'collectEnd') && ~isempty(spikeStruct.opts.collectEnd)
        opts.collectEnd = spikeStruct.opts.collectEnd;
    end
end

lfpCleanParams = struct();
lfpCleanParams.skipArtifactClean = true;
loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
loadArgs = [loadArgs, {'lfpCleanParams', lfpCleanParams, 'bands', bands}];
lfpStruct = load_session_data(sessionType, 'lfp', loadArgs{:});

spikeAreaIdx = find(strcmp(spikeStruct.areas, brainArea), 1);
lfpAreaIdx = find(strcmp(lfpStruct.areas, brainArea), 1);
if isempty(spikeAreaIdx) || isempty(lfpAreaIdx)
    error('session_lfp_power:MissingArea', ...
        'Area "%s" must be present in both spikes (%s) and LFP (%s).', ...
        brainArea, strjoin(spikeStruct.areas, ', '), strjoin(lfpStruct.areas, ', '));
end

[centerTimes, popRate, bandPower] = sliding_power_and_pop( ...
    spikeStruct, spikeAreaIdx, lfpStruct, lfpAreaIdx, windowSec, stepSec);

fprintf('Windows: %d from %.1f s to %.1f s\n', ...
    numel(centerTimes), centerTimes(1), centerTimes(end));
fprintf('Neurons in %s: %d\n', brainArea, numel(spikeStruct.idLabel{spikeAreaIdx}));

plot_power_and_pop(centerTimes, popRate, bandPower, bands(:, 1), ...
    sessionName, brainArea, windowSec, stepSec);

%% Local functions

function [centerTimes, popRate, bandPower] = sliding_power_and_pop( ...
    spikeStruct, spikeAreaIdx, lfpStruct, lfpAreaIdx, windowSec, stepSec)
% SLIDING_POWER_AND_POP - Window means of population rate and band power
%
% Variables:
%   spikeStruct   - Spikes from load_session_data
%   spikeAreaIdx  - Column of spikeStruct.areas for this area
%   lfpStruct     - LFP from load_session_data (binnedPower, timePoints)
%   lfpAreaIdx    - Column of lfpStruct.areas for this area
%   windowSec     - Full window length (s)
%   stepSec       - Step between window centers (s)
%
% Goal:
%   One center time per step. Population rate is total spikes in the window
%   divided by window length. Band power is the mean of that band's binned
%   power inside the same window.
%
% Returns:
%   centerTimes - Window centers (s, absolute session time)
%   popRate     - Population spike rate (spikes/s), one value per center
%   bandPower   - [nCenters x nBands] mean power

timeOrigin = 0;
if isfield(lfpStruct, 'opts') && isfield(lfpStruct.opts, 'collectStart') ...
        && ~isempty(lfpStruct.opts.collectStart)
    timeOrigin = lfpStruct.opts.collectStart;
end
nSamples = size(lfpStruct.lfpPerArea, 1);
fsLfp = lfpStruct.opts.fsLfp;
tEnd = timeOrigin + nSamples / fsLfp;
firstCenter = timeOrigin + windowSec / 2;
lastCenter = tEnd - windowSec / 2;
centerTimes = firstCenter:stepSec:lastCenter;
centerTimes = centerTimes(:);
if isempty(centerTimes)
    error('session_lfp_power:NoWindows', ...
        'No windows fit in %.1f s. Shorten windowSec.', tEnd - timeOrigin);
end

neuronIds = spikeStruct.idLabel{spikeAreaIdx};
neuronIds = neuronIds(:);
if isempty(neuronIds)
    error('session_lfp_power:NoNeurons', ...
        'No neurons in area %s.', spikeStruct.areas{spikeAreaIdx});
end
inArea = ismember(spikeStruct.spikeClusters, neuronIds);
areaSpikeTimes = spikeStruct.spikeTimes(inArea);
popRate = population_rate_in_windows(areaSpikeTimes, centerTimes, windowSec);

nBands = size(lfpStruct.bands, 1);
bandPower = nan(numel(centerTimes), nBands);
halfWin = windowSec / 2;
for b = 1:nBands
    binTimes = timeOrigin + lfpStruct.timePoints{lfpAreaIdx}{b}(:);
    powerVals = lfpStruct.binnedPower{lfpAreaIdx}{b}(:);
    bandPower(:, b) = window_mean_at_centers(binTimes, powerVals, centerTimes, halfWin);
end
end

function popRate = population_rate_in_windows(spikeTimes, centerTimes, windowSec)
% POPULATION_RATE_IN_WINDOWS - Total spikes/s in each centered window
%
% Variables:
%   spikeTimes  - Spike times (s) for every neuron in the area
%   centerTimes - Window centers (s)
%   windowSec   - Window length (s)
%
% Goal:
%   Count spikes in [center - window/2, center + window/2) and divide by
%   window length.

if isempty(spikeTimes)
    popRate = zeros(numel(centerTimes), 1);
    return
end
spikeTimes = sort(spikeTimes(:));
iLeft = lookup_first_bin_at_or_after(spikeTimes, centerTimes - windowSec / 2);
iRight = lookup_first_bin_at_or_after(spikeTimes, centerTimes + windowSec / 2);
popRate = max(0, iRight - iLeft) / windowSec;
end

function yMean = window_mean_at_centers(binTimes, values, centerTimes, halfWin)
% WINDOW_MEAN_AT_CENTERS - Mean of samples whose times fall in each window
%
% Variables:
%   binTimes    - Sample times (s), ascending
%   values      - Sample values, same length as binTimes
%   centerTimes - Window centers (s)
%   halfWin     - Half window (s)
%
% Goal:
%   Average band-power bins inside [center - halfWin, center + halfWin).

yMean = nan(numel(centerTimes), 1);
if isempty(binTimes) || isempty(values)
    return
end
nKeep = min(numel(binTimes), numel(values));
binTimes = binTimes(1:nKeep);
values = values(1:nKeep);
[binTimes, sortIdx] = sort(binTimes);
values = values(sortIdx);
cs = [0; cumsum(values)];
iLeft = lookup_first_bin_at_or_after(binTimes, centerTimes - halfWin);
iRight = lookup_first_bin_at_or_after(binTimes, centerTimes + halfWin);
nIn = iRight - iLeft;
valid = nIn > 0;
sumIn = cs(iRight(valid)) - cs(iLeft(valid));
yMean(valid) = sumIn ./ nIn(valid);
end

function idx = lookup_first_bin_at_or_after(sampleTimes, queryTimes)
% LOOKUP_FIRST_BIN_AT_OR_AFTER - First index with time >= each query
%
% Variables:
%   sampleTimes - Ascending sample times (duplicates allowed)
%   queryTimes  - Query times, one per window
%
% Goal:
%   Left edge of a half-open window. Queries after the last sample return
%   numel(sampleTimes)+1 so a difference of edges is a count.

nTimes = numel(sampleTimes);
queryTimes = queryTimes(:);
idx = repmat(nTimes + 1, numel(queryTimes), 1);
if nTimes == 0
    return
end
sampleTimes = sampleTimes(:);
for iQuery = 1:numel(queryTimes)
    q = queryTimes(iQuery);
    if sampleTimes(1) >= q
        idx(iQuery) = 1;
        continue
    end
    if sampleTimes(nTimes) < q
        continue
    end
    lo = 1;
    hi = nTimes;
    while lo < hi
        mid = floor((lo + hi) / 2);
        if sampleTimes(mid) < q
            lo = mid + 1;
        else
            hi = mid;
        end
    end
    idx(iQuery) = lo;
end
end

function plot_power_and_pop(centerTimes, popRate, bandPower, bandNames, ...
    sessionName, brainArea, windowSec, stepSec)
% PLOT_POWER_AND_POP - Population rate above log10 band power
%
% Variables:
%   centerTimes - Window centers (s)
%   popRate     - Population spike rate (spikes/s)
%   bandPower   - [nCenters x nBands] mean power
%   bandNames   - Cell of band labels
%   sessionName, brainArea, windowSec, stepSec - Title text
%
% Goal:
%   2x1 figure with a shared time axis. Top is spiking. Bottom overlays
%   every band.

bandNames = cellstr(bandNames);
nBands = numel(bandNames);
colors = band_colors(nBands);
logPower = log10(bandPower);
logPower(~isfinite(logPower)) = nan;

fig = figure('Color', 'w', 'Position', [120 80 960 640], ...
    'Name', sprintf('LFP power | %s | %s', sessionName, brainArea));
tileLayout = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

axPop = nexttile(tileLayout);
plot(axPop, centerTimes, popRate, 'Color', [0.1 0.1 0.1], 'LineWidth', 1.2);
ylabel(axPop, 'Population rate (spikes/s)');
title(axPop, brainArea, 'Interpreter', 'none');
set(axPop, 'XTickLabel', [], 'Box', 'off', 'TickDir', 'out', 'FontSize', 11);
grid(axPop, 'on');

axPower = nexttile(tileLayout);
hold(axPower, 'on');
for b = 1:nBands
    plot(axPower, centerTimes, logPower(:, b), 'Color', colors(b, :), ...
        'LineWidth', 1.2, 'DisplayName', bandNames{b});
end
hold(axPower, 'off');
xlabel(axPower, 'Time (s)');
ylabel(axPower, 'log_{10}(band power)', 'Interpreter', 'tex');
legend(axPower, 'Location', 'northeast', 'Interpreter', 'none');
set(axPower, 'Box', 'off', 'TickDir', 'out', 'FontSize', 11);
grid(axPower, 'on');

linkaxes([axPop, axPower], 'x');
sgtitle(tileLayout, sprintf('%s | %s | %.0f s windows, %.0f s step', ...
    sessionName, brainArea, windowSec, stepSec), 'Interpreter', 'none');
end

function colors = band_colors(nBands)
% BAND_COLORS - One RGB row per band, delta through high gamma first
%
% Variables:
%   nBands - Number of bands to color
%
% Goal:
%   Stable colors when the default band list is used. Extra bands use lines.

palette = [ ...
    0.45 0.20 0.70; ...
    0.20 0.40 0.85; ...
    0.15 0.62 0.32; ...
    0.90 0.50 0.10; ...
    0.85 0.22 0.22; ...
    0.15 0.15 0.15];
if nBands <= size(palette, 1)
    colors = palette(1:nBands, :);
    return
end
colors = lines(nBands);
end
