function [spikeTimes, spikeClusters] = filter_isi_violations(spikeTimes, spikeClusters, minIsiSec)
% FILTER_ISI_VIOLATIONS - Keep the first spike when a unit violates min ISI
%
% Variables:
%   spikeTimes    - Spike times in seconds (vector)
%   spikeClusters - Unit / cluster ID for each spike (same size as spikeTimes)
%   minIsiSec     - Minimum allowed ISI in seconds (default 0.0015 = 1.5 ms)
%
% Goal:
%   For every unit (good, mua, or real), if more than one spike occurs within
%   minIsiSec, delete all but the first. Each kept spike opens a new
%   refractory window, so later spikes are compared to the last kept time.

    if nargin < 3 || isempty(minIsiSec)
        minIsiSec = 0.0015;
        minIsiSec = 0.0015;
    end

    if isempty(spikeTimes)
        fprintf('ISI filter (%.1f ms): removed 0 of 0 spikes from 0 of 0 units\n', ...
            minIsiSec * 1000);
        return
    end

    wasRowTimes = isrow(spikeTimes);
    wasRowClusters = isrow(spikeClusters);
    spikeTimes = spikeTimes(:);
    spikeClusters = spikeClusters(:);

    nSpikes = numel(spikeTimes);
    [~, sortIdx] = sortrows([double(spikeClusters), spikeTimes]);
    sortedTimes = spikeTimes(sortIdx);
    sortedClusters = spikeClusters(sortIdx);

    keepSorted = true(nSpikes, 1);
    lastKeptTime = sortedTimes(1);
    lastCluster = sortedClusters(1);
    nRemoved = 0;
    for iSpike = 2:nSpikes
        if sortedClusters(iSpike) ~= lastCluster
            lastCluster = sortedClusters(iSpike);
            lastKeptTime = sortedTimes(iSpike);
            continue
        end
        if sortedTimes(iSpike) - lastKeptTime < minIsiSec
            keepSorted(iSpike) = false;
            nRemoved = nRemoved + 1;
        else
            lastKeptTime = sortedTimes(iSpike);
        end
    end

    keepMask = false(nSpikes, 1);
    keepMask(sortIdx) = keepSorted;
    spikeTimes = spikeTimes(keepMask);
    spikeClusters = spikeClusters(keepMask);

    if wasRowTimes
        spikeTimes = spikeTimes.';
    end
    if wasRowClusters
        spikeClusters = spikeClusters.';
    end

    nUnits = numel(unique(sortedClusters));
    nUnitsRemoved = numel(unique(sortedClusters(~keepSorted)));
    fprintf('ISI filter (%.1f ms): removed %d of %d spikes from %d of %d units\n', ...
        minIsiSec * 1000, nRemoved, nSpikes, nUnitsRemoved, nUnits);
end
