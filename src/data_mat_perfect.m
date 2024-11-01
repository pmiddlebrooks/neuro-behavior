% Inputs:
% dataMat: Original matrix of spike counts (samples x neurons)
% bhvIDMat: Vector classifying each sample in dataMat into categories

% Outputs:
% dataMatPerfect: New matrix with specified properties

% Initialize dataMatPerfect with zeros, matching the size of dataMat
[sampleCount, neuronCount] = size(dataMat);
dataMatPerfect = zeros(sampleCount, neuronCount);

% Compute average spikes per neuron from dataMat
avgSpikesPerNeuron = mean(dataMat, 1);

% Define unique spike count patterns for each category
categories = unique(bhvIDMat);
numCategories = length(categories);
categoryPatterns = rand(numCategories, neuronCount) * 10; % Customize the range for distinct patterns

% Populate dataMatPerfect:
% 1. Set baseline spikes for each neuron based on original averages
for neuronIdx = 1:neuronCount
    dataMatPerfect(:, neuronIdx) = poissrnd(avgSpikesPerNeuron(neuronIdx), sampleCount, 1);
end

% 2. Insert distinct patterns before each category transition
for sampleIdx = 2:sampleCount
    if bhvIDMat(sampleIdx) ~= bhvIDMat(sampleIdx - 1)
        % Transition detected; apply the pattern for the new category
        categoryIdx = find(categories == bhvIDMat(sampleIdx));
        dataMatPerfect(sampleIdx - 1, :) = categoryPatterns(categoryIdx, :);
    end
end

% dataMatPerfect is now generated with the specified properties.
