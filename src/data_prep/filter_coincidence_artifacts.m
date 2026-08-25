function [spikeTimes, spikeClusters] = filter_coincidence_artifacts( ...
    spikeTimes, spikeClusters, neuronIDs, neuronAreas, opts)
% FILTER_COINCIDENCE_ARTIFACTS - Drop local multi-unit coincident noise bursts
%
% Variables:
%   spikeTimes    - Spike times in seconds
%   spikeClusters - Unit ID per spike
%   neuronIDs     - Unit IDs (same order as neuronAreas)
%   neuronAreas   - Area label per unit (e.g. M23, M56, DS, VS)
%   opts          - Optional; fields:
%       .coincidenceBinSec      - Bin width (default 0.001 s)
%       .coincidenceMinUnits    - Min unique units in a bin (default 6)
%       .coincidenceMinFraction - Min fraction of the area's units (default 0.25)
%
% Goal:
%   Within each area separately, flag 1 ms bins where too many unique units
%   spike at once (electrical artifacts). Delete only those spikes in that
%   area. Other areas in the same bin are left unchanged, so artifacts
%   confined to M23/M56 do not require DS/VS coincidence.

    binSizeSec = 0.001;
    minUnits = 6;
    minFraction = 0.25;
    if nargin >= 5 && isstruct(opts)
        if isfield(opts, 'coincidenceBinSec') && ~isempty(opts.coincidenceBinSec)
            binSizeSec = opts.coincidenceBinSec;
        end
        if isfield(opts, 'coincidenceMinUnits') && ~isempty(opts.coincidenceMinUnits)
            minUnits = opts.coincidenceMinUnits;
        end
        if isfield(opts, 'coincidenceMinFraction') && ~isempty(opts.coincidenceMinFraction)
            minFraction = opts.coincidenceMinFraction;
        end
    end

    if isempty(spikeTimes) || isempty(neuronIDs)
        fprintf(['Coincidence filter (%.1f ms, local): removed 0 of 0 spikes ', ...
            'from 0 bins\n'], binSizeSec * 1000);
        return
    end

    wasRowTimes = isrow(spikeTimes);
    wasRowClusters = isrow(spikeClusters);
    spikeTimes = spikeTimes(:);
    spikeClusters = spikeClusters(:);
    neuronIDs = neuronIDs(:);
    areaStr = strtrim(string(neuronAreas(:)));
    areaStr(ismissing(areaStr)) = "";
    neuronAreas = cellstr(areaStr);

    nSpikes = numel(spikeTimes);
    keepMask = true(nSpikes, 1);
    [isMapped, unitLoc] = ismember(spikeClusters, neuronIDs);
    spikeAreaNames = repmat({''}, nSpikes, 1);
    spikeAreaNames(isMapped) = neuronAreas(unitLoc(isMapped));

    areaNames = unique(spikeAreaNames(isMapped & ~cellfun(@isempty, spikeAreaNames)));
    nRemoved = 0;
    nBinsFlagged = 0;
    areaReports = {};

    for iArea = 1:numel(areaNames)
        areaName = areaNames{iArea};
        nAreaUnits = sum(strcmp(neuronAreas, areaName));
        areaSpikeMask = strcmp(spikeAreaNames, areaName);
        [areaKeep, nAreaRemoved, nAreaBins, thresh] = filter_area_coincidence( ...
            spikeTimes(areaSpikeMask), spikeClusters(areaSpikeMask), ...
            binSizeSec, minUnits, minFraction, nAreaUnits);
        keepMask(areaSpikeMask) = areaKeep;
        nRemoved = nRemoved + nAreaRemoved;
        nBinsFlagged = nBinsFlagged + nAreaBins;
        areaReports{end+1} = sprintf('%s: %d spikes / %d bins (thresh %d of %d units)', ...
            areaName, nAreaRemoved, nAreaBins, thresh, nAreaUnits); %#ok<AGROW>
    end

    spikeTimes = spikeTimes(keepMask);
    spikeClusters = spikeClusters(keepMask);

    if wasRowTimes
        spikeTimes = spikeTimes.';
    end
    if wasRowClusters
        spikeClusters = spikeClusters.';
    end

    fprintf('Coincidence filter (%.1f ms, local): removed %d of %d spikes from %d bins\n', ...
        binSizeSec * 1000, nRemoved, nSpikes, nBinsFlagged);
    for iArea = 1:numel(areaReports)
        fprintf('  %s\n', areaReports{iArea});
    end
end

function [keepMask, nRemoved, nBinsFlagged, thresh] = filter_area_coincidence( ...
    areaTimes, areaClusters, binSizeSec, minUnits, minFraction, nAreaUnits)
% FILTER_AREA_COINCIDENCE - Flag high unique-unit bins within one area
%
% Variables:
%   areaTimes     - Spike times for this area (seconds)
%   areaClusters  - Unit IDs for those spikes
%   binSizeSec    - Coincidence bin width
%   minUnits      - Minimum unique units to flag a bin
%   minFraction   - Minimum fraction of the area's units
%   nAreaUnits    - Number of units assigned to this area
%
% Goal:
%   Count unique units per bin and drop spikes in bins at or above threshold.

    keepMask = true(size(areaTimes));
    nRemoved = 0;
    nBinsFlagged = 0;
    thresh = max(minUnits, ceil(minFraction * nAreaUnits));
    if isempty(areaTimes) || nAreaUnits <= 0 || thresh > nAreaUnits
        return
    end

    binIdx = floor(areaTimes / binSizeSec);
    binIdx = binIdx - min(binIdx) + 1;
    pairMat = unique([binIdx, double(areaClusters)], 'rows');
    nBins = max(binIdx);
    uniqueCount = accumarray(pairMat(:, 1), 1, [nBins, 1]);
    flaggedBins = uniqueCount >= thresh;
    nBinsFlagged = sum(flaggedBins);
    if nBinsFlagged == 0
        return
    end

    keepMask = ~flaggedBins(binIdx);
    nRemoved = sum(~keepMask);
end
