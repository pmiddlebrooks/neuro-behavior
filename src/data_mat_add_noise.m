function dataMat = data_mat_add_noise(dataMat, noisePct)

% Add or subtract a percentage of spikes randomly to a spike data matrix of
% samples X neurons

% Adds spikes (+1) randomly if addSpikes(col) is positive.
% Removes spikes (-1) randomly if addSpikes(col) is negative, ensuring it only removes existing spikes.
% Uses vectorized operations (accumarray) where possible for efficiency.

% % Total spikes per neuron 
% totalSpikes = sum(dataMat, 1);
% 
% addSpikes = round(totalSpikes .* noisePct ./ 100);
% 
% % Process each column independently
% for col = 1:size(dataMat, 2)
%     numSpikes = addSpikes(col);
% 
%     if numSpikes > 0
%         % Add spikes (+1) to random positions
%         randIndices = randi(size(dataMat, 1), numSpikes, 1);
%         dataMat(:, col) = dataMat(:, col) + accumarray(randIndices, 1, [size(dataMat, 1), 1]);
% 
%     elseif numSpikes < 0
%         % Remove spikes (-1) from existing spikes (without going below zero)
%         existingSpikeIndices = find(dataMat(:, col) > 0); % Find where spikes exist
% 
%         if numel(existingSpikeIndices) < abs(numSpikes)
%             warning('Not enough spikes to remove in column %d. Removing all available.', col);
%             numSpikes = -numel(existingSpikeIndices); % Only remove available spikes
%         end
% 
%         % Randomly select spikes to remove
%         removeIndices = existingSpikeIndices(randperm(numel(existingSpikeIndices), abs(numSpikes)));
% 
%         % Subtract spikes at selected locations
%         dataMat(removeIndices, col) = dataMat(removeIndices, col) - 1;
%     end
% end


% Total spikes per neuron 
totalSpikes = sum(dataMat, 1);

% Compute how many spikes to add or remove per neuron
addSpikes = round(totalSpikes .* noisePct ./ 100);

% Process each neuron independently
for col = 1:size(dataMat, 2)
    numSpikes = addSpikes(col);
    
    if numSpikes > 0
        % Add spikes (+1) to random positions
        randIndices = randsample(size(dataMat, 1), numSpikes, true); % Sample with replacement
        dataMat(:, col) = dataMat(:, col) + accumarray(randIndices, 1, [size(dataMat, 1), 1]);

    elseif numSpikes < 0
        % Find ALL spikes across time bins for this neuron
        existingSpikeIndices = find(dataMat(:, col) > 0); % Locations where spikes exist

        % Expand spike indices based on count (accounts for multiple spikes per bin)
        expandedSpikeIndices = repelem(existingSpikeIndices, dataMat(existingSpikeIndices, col));

        % Ensure we don't remove more spikes than available
        numSpikes = min(abs(numSpikes), length(expandedSpikeIndices));

        if numSpikes > 0
            % Randomly select indices to remove spikes from
            removeIndices = randsample(expandedSpikeIndices, numSpikes, false);

            % Create removal array and update dataMat
            removalMatrix = accumarray(removeIndices, -1, [size(dataMat, 1), 1]);
            dataMat(:, col) = max(dataMat(:, col) + removalMatrix, 0); % Ensure non-negative spike counts
        end
    end
end
