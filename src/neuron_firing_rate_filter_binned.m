function keepMask = neuron_firing_rate_filter_binned(dataMat, opts, dataMatDuration)
% NEURON_FIRING_RATE_FILTER_BINNED - Keep mask from firing rate on binned data
%
% Variables:
%   dataMat          - [timeBins x neurons] binned spike counts or rates
%   opts             - options with method, frameSize, min/max firing rate
%   dataMatDuration  - total duration of dataMat in seconds
%
% Returns:
%   keepMask - logical row vector, true for neurons that pass criteria
%
% Goal:
%   If opts.firingRateCheckTime is empty, require min/max rate over the full
%   session. Otherwise check rates at the start and end windows.

useWholeSession = ~isfield(opts, 'firingRateCheckTime') || isempty(opts.firingRateCheckTime);

if useWholeSession
    if strcmp(opts.method, 'standard')
        meanRate = sum(dataMat, 1) ./ dataMatDuration;
    else
        meanRate = mean(dataMat, 1);
    end
    keepMask = meanRate >= opts.minFiringRate & meanRate <= opts.maxFiringRate;
    return;
end

checkTime = opts.firingRateCheckTime;
if checkTime > dataMatDuration
    checkTime = dataMatDuration;
end
checkFrames = floor(checkTime / opts.frameSize);
if ~strcmp(opts.method, 'standard')
    meanStart = mean(dataMat(1:checkFrames, :), 1);
    meanEnd = mean(dataMat(end - checkFrames + 1:end, :), 1);
else
    meanStart = sum(dataMat(1:checkFrames, :), 1) ./ checkTime;
    meanEnd = sum(dataMat(end - checkFrames + 1:end, :), 1) ./ checkTime;
end
keepMask = meanStart >= opts.minFiringRate & meanStart <= opts.maxFiringRate & ...
    meanEnd >= opts.minFiringRate & meanEnd <= opts.maxFiringRate;
end
