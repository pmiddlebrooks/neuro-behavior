function [dataMat, idLabels, rmvNeurons] = neural_matrix(data, opts)

durFrames = floor(sum(data.bhvDur) / opts.frameSize);

if ismember('id',data.ci.Properties.VariableNames)
    idLabels = data.ci.id(opts.useNeurons);
else
    idLabels = data.ci.cluster_id(opts.useNeurons);
end

% Preallocate data matrix
dataMat = zeros(durFrames, length(opts.useNeurons));

% Loop through each neuron and put spike numbers within each frame
for i = 1 : length(opts.useNeurons)

    iSpikeTimes = data.spikeTimes(data.spikeClusters == opts.useNeurons(i));

    % Define the time interval (100 ms in seconds)
    interval = opts.frameSize;

    % Initialize a vector to store the counts
    iSpikeCount = zeros(size(dataMat, 1), 1);

    % Count the number of time stamps (spikes) within each interval
    for j = 1:size(dataMat, 1)
        lower_bound = ((j - 1) * interval) + (interval * opts.shiftAlignFactor);
        upper_bound = (j * interval) + (interval * opts.shiftAlignFactor);
        iSpikeCount(j) = sum(iSpikeTimes >= lower_bound & iSpikeTimes < upper_bound);
    end
    % [sum(iSpikeCount) find(iSpikeCount, 1, 'last')]
    dataMat(:, i) = iSpikeCount;
    % Display the count vector
    % disp(sSpikeCount);

end

%% Remove neurons that are out of min-max firing rates
%
% check mean rates at the beginning and end of data to ensure somewhat
% stable isolation and neurons are within firirng rate range

checkTime = 5 * 60;
checkFrames = checkTime / opts.frameSize;

meanStart = sum(dataMat(1:checkFrames, :), 1) / checkTime;
meanEnd = sum(dataMat(end-checkFrames+1:end, :), 1) / checkTime;

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
% imagesc(dataMat')








