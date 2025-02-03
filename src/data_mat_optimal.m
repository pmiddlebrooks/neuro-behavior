function dataMatOpt = data_mat_optimal(dataMat, labels, noisePercentage, opts)

% Inputs: dataMat (original spike matrix), labels (vector of behavior labels), opts struct
[numTimeBins, numNeurons] = size(dataMat);
uniqueBhv = unique(labels); % Unique behavior labels
numBehaviors = length(uniqueBhv);

% Compute the proportion of samples occupied by each behavior
bhvCounts = histcounts(labels, [uniqueBhv; max(uniqueBhv)+1]); % Count occurrences of each behavior
bhvProportions = bhvCounts / numTimeBins; % Fraction of time bins occupied by each behavior

% Compute the proportion of spikes for each neuron
neuronSpikeProportions = sum(dataMat, 1) / numTimeBins; % 1 x numNeurons

% Initialize optimized data matrix
dataMatOpt = zeros(numTimeBins, numNeurons);

% Alternate accumulation strategy
alternateFlag = true; % Start with one accumulation strategy

% Assign behaviors to neurons based on best-matching proportions with random sampling
for n = 1:numNeurons
    targetSpikeCount = neuronSpikeProportions(n) * numTimeBins; % Convert proportion to absolute spike count
    
    % Initialize tracking variables
    selectedBhv = [];
    cumulativeSpikes = 0;
    
    % Random selection with replacement
    while true
        randBhv = uniqueBhv(randi(numBehaviors)); % Randomly select a behavior with replacement
        selectedBhv = [selectedBhv; randBhv]; % Store selected behavior
        
        % Get indices for this behavior
        bhvIndices = find(labels == randBhv);
        
        % Add spikes at the selected behavior's indices
        numSpikesToAdd = length(bhvIndices);
        
        % Stopping condition based on alternating over/under strategy
        if alternateFlag
            if cumulativeSpikes + numSpikesToAdd > targetSpikeCount
                break;
            end
        end

        dataMatOpt(bhvIndices, n) = dataMatOpt(bhvIndices, n) + 1; % Allow multiple spikes
        
        % Update the cumulative spike count
        cumulativeSpikes = cumulativeSpikes + numSpikesToAdd;
        
        % Stopping condition based on alternating over/under strategy
        if ~alternateFlag
            if cumulativeSpikes >= targetSpikeCount
                break;
            end
        end
    end
    
    % Ensure minimum firing rate constraint is satisfied
    while (sum(dataMatOpt(:, n)) / (numTimeBins * opts.frameSize)) < opts.minFiringRate
        % Select a random behavior to add more spikes
        randBhv = uniqueBhv(randi(numBehaviors));
        bhvIndices = find(labels == randBhv);
        
        % Add spikes to this behavior
        numSpikesToAdd = length(bhvIndices);
        dataMatOpt(bhvIndices, n) = dataMatOpt(bhvIndices, n) + 1;
    end

    % Toggle the alternate flag for the next neuron
    alternateFlag = ~alternateFlag;
end

% **Final Correction Step: Match Total Spikes Between dataMat and dataMatOpt**
spikeDifference = sum(dataMatOpt, 1) - sum(dataMat, 1); % Difference per neuron

for n = 1:numNeurons
    if spikeDifference(n) > 0
        % Too many spikes → remove excess
        spikeIndices = find(dataMatOpt(:, n) > 0); % Find nonzero indices
        removeIdx = randperm(length(spikeIndices), min(spikeDifference(n), length(spikeIndices))); 
        dataMatOpt(spikeIndices(removeIdx), n) = dataMatOpt(spikeIndices(removeIdx), n) - 1;
        
    elseif spikeDifference(n) < 0
        % Too few spikes → add missing ones
        zeroIndices = find(dataMatOpt(:, n) == 0); % Find zero locations
        addIdx = randperm(length(zeroIndices), min(abs(spikeDifference(n)), length(zeroIndices)));
        dataMatOpt(zeroIndices(addIdx), n) = dataMatOpt(zeroIndices(addIdx), n) + 1;
    end
end

% **Add Noise to Spike Timing (Fixed Version)**
if noisePercentage > 0
    numNoisySpikes = round(noisePercentage / 100 * sum(dataMatOpt, 1)); % Compute number of bins to modify

    for n = 1:numNeurons
        % % Find indices where spikes exist
        % spikeIndices = find(dataMatOpt(:, n) > 0);
        % 
        % % Ensure we have enough spikes to shuffle
        % if length(spikeIndices) < numNoisyBins
        %     numNoisyBins = length(spikeIndices);
        % end
        % 
        % % Select a random subset of spikes to move
        % removeIdx = randsample(spikeIndices, numNoisyBins);
        % 
        % % Find indices where we can add spikes (i.e., not the same as removeIdx)
        % availableIndices = setdiff(1:numTimeBins, removeIdx);
        % addIdx = randsample(availableIndices, numNoisyBins);
        % 
        % % Perform spike shifting
        % dataMatOpt(removeIdx, n) = dataMatOpt(removeIdx, n) - 1; % Remove spike
        % dataMatOpt(addIdx, n) = dataMatOpt(addIdx, n) + 1; % Add spike


                    % Get all spike locations and expand according to spike count
            spikeLocations = find(dataMatOpt(:, n) > 0);
            expandedSpikeIndices = repelem(spikeLocations, dataMatOpt(spikeLocations, n));

            % Select the indices of spikes to move (allow duplicates)
            numNoisySpikes(n) = min(numNoisySpikes(n), length(expandedSpikeIndices)); % Ensure valid size
            removeIdx = randsample(expandedSpikeIndices, numNoisySpikes(n), false); % Allow duplicate selection

            % Select indices where to re-add spikes (allowing replacement)
            % addIdx = randsample(1:numTimeBins, numNoisySpikes(n), true); % Replacement allowed

            % Select indices where to re-add spikes
            if opts.shuffleWithReplacement
                % Allow spikes to be placed into bins that already have spikes
                addIdx = randsample(1:numTimeBins, numNoisySpikes(n), true); 
            else
                % Ensure spikes are only added to bins that are currently empty
                availableIndices = find(dataMatOpt(:, n) == 0); % Only consider empty bins
                addIdx = randsample(availableIndices, min(numNoisySpikes(n), length(availableIndices)), false);
            end

            removalMatrix = accumarray(removeIdx(:), -1, [numTimeBins, 1], @sum, 0);
            additionMatrix = accumarray(addIdx(:), 1, [numTimeBins, 1], @sum, 0);

            % Apply changes in one step
            dataMatOpt(:, n) = dataMatOpt(:, n) + removalMatrix + additionMatrix;

    end
end
