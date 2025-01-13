%%  Run PCA from  pca_neuron_contributions, then begin here
%
% E:\Projects\neuro-behavior\src\pca_neuron_contributions.m
%%

% pca_neuron_contributions



%% Which data to model:
preInd = [diff(bhvIDMat) ~= 0; 0]; % 1 frame prior to all behavior transitions


%% within-bout
modelInd = ~preInd & ~[preInd(2:end); 0] & ~(bhvID == -1); % Everything but 1 frame prior and 1 frame post behavior onset

%% all data
modelInd = 1:length(bhvID);
modelID = bhvID;



%% On average, how much does each neuron contribute to each behavior?

% Inputs:
% svmID: Behavior labels for each sample

% Get the unique behavior labels
uniqueLabels = unique(bhvID(modelInd));

% Initialize a matrix to store neuron contributions for each behavior label
[numNeurons, numComponents] = size(coeff);
neuronContributions = zeros(numNeurons, length(uniqueLabels));

% Loop through each behavior label
for i = 1:length(uniqueLabels)
    % Get indices for the current behavior label
    labelIdx = (bhvID(modelInd) == uniqueLabels(i));
    
    % Average projections (score) for the current label
    avgLabelScore = mean(score(labelIdx, :), 1); % Mean across all samples with the label
    
    % Calculate the contribution of each neuron to the behavior label
    % Multiply PCA loadings by the mean score for the current label
    neuronContributions(:, i) = coeff * avgLabelScore';
end

% Store neuron contributions for each behavior label
disp('Neuron contributions for each behavior label:');
disp(neuronContributions);
