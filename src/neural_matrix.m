function [dataMat, idLabels, areaLabels, rmvNeurons] = neural_matrix(data, opts)


    if opts.shiftAlignFactor ~= 0
    warning('neural_matrix.m:  You shifted the spike bin time w.r.t. the behavior start time: make sure you wanted to do this.')
end

% checkTime = 5 * 60;
durFrames = ceil(sum(data.bhvDur) / opts.frameSize);

if ismember('id',data.ci.Properties.VariableNames)
    idLabels = data.ci.id(opts.useNeurons);
else
    idLabels = data.ci.cluster_id(opts.useNeurons);
end

% Preallocate data matrix: If it's huge, make it a int matrix (reduce memory)
if durFrames > 30*60/.001
    dataMat = int8(zeros(durFrames, length(opts.useNeurons)));
else
    dataMat = zeros(durFrames, length(opts.useNeurons));
end

% dataMat = int8([]);
areaLabels = {};
% rmvNeurons = [];
% Loop through each neuron and put spike numbers within each frame
for i = 1 : length(idLabels)

    % iSpikeTimes = data.spikeTimes(data.spikeClusters == opts.useNeurons(i));
    iSpikeTime = data.spikeTimes(data.spikeClusters == idLabels(i));


    % % Don't add neurons out of min-max firing rate
    % meanStart = sum(iSpikeTime >= checkTime) ./ checkTime;
    % meanEnd = sum(iSpikeTime > opts.collectFor - checkTime) ./ checkTime;
    %
    % keepStart = meanStart >= opts.minFiringRate & meanStart <= opts.maxFiringRate;
    % keepEnd = meanEnd >= opts.minFiringRate & meanEnd <= opts.maxFiringRate;
    % if keepStart && keepEnd

    timeEdges = 0 : opts.frameSize : sum(data.bhvDur); % Create edges of bins from 0 to max time
if opts.shiftAlignFactor ~= 0
    timeEdges = timeEdges + opts.shiftAlignFactor;
end

    % Count the number of spikes in each bin
    [iSpikeCount, ~] = histcounts(iSpikeTime, timeEdges);
% If we shifted where the spike count bin is, account for that here
    if opts.shiftAlignFactor ~= 0
    iSpikeCount = [0 iSpikeCount(1:end-1)];
end

    dataMat(:, i) = iSpikeCount';
    % dataMat = [dataMat, iSpikeCount'];

    % Keep track which neurons (columns) are in which brain area
    areaLabels = [areaLabels, data.ci.area(opts.useNeurons(i))];
    % else
    %     rmvNeurons = [rmvNeurons, i];
    % end
end



% return
%% Remove neurons that are out of min-max firing rates
%
% check mean rates at the beginning and end of data to ensure somewhat
% stable isolation and neurons are within firirng rate range
rmvNeurons = [];
if opts.removeSome

    checkTime = 5 * 60;
    checkFrames = floor(checkTime / opts.frameSize);

    meanStart = sum(dataMat(1:checkFrames, :), 1) ./ checkTime;
    meanEnd = sum(dataMat(end-checkFrames+1:end, :), 1) ./ checkTime;

    keepStart = meanStart >= opts.minFiringRate & meanStart <= opts.maxFiringRate;
    keepEnd = meanEnd >= opts.minFiringRate & meanEnd <= opts.maxFiringRate;

    rmvNeurons = ~(keepStart & keepEnd);
    fprintf('\nkeeping %d of %d neurons\n', sum(~rmvNeurons), length(rmvNeurons))

    % nrnIdx = find(rmvNeurons);
    % for i = 1 : length(nrnIdx)
    % plot(dataMat(:,nrnIdx(i)))
    % [nrnIdx(i) meanStart(nrnIdx(i)) sum(dataMat(8000:10000,nrnIdx(i)) / checkTime) meanEnd(nrnIdx(i))]
    % end

    % meanRates = sum(dataMat, 1) / (size(dataMat, 1) * opts.frameSize);
    % sum(meanRates >= opts.minFiringRate & meanRates <= opts.maxFiringRate)

    dataMat(:,rmvNeurons) = [];
    idLabels(rmvNeurons) = [];
    areaLabels(rmvNeurons) = [];
    % imagesc(dataMat')

end








