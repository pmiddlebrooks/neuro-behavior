%%  Run PCA from  pca_neuron_contributions, then begin here
%
% E:\Projects\neuro-behavior\src\pca_neuron_contributions.m
%%

% pca_neuron_contributions



%% Which data to model:
preInd = [diff(bhvID) ~= 0; 0]; % 1 frame prior to all behavior transitions


%% within-bout
modelInd = ~preInd & ~[preInd(2:end); 0] & ~(bhvID == -1); % Everything but 1 frame prior and 1 frame post behavior onset

%% all data
modelInd = 1:length(bhvID);
modelID = bhvID;



%% PCA for single behaviors. How is the explained varience over components?
% modelInd = modelInd & bhvID == 11;
%%
    [coeff, score, ~, ~, explained] = pca(dataMat(modelInd, idSelect));


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


%% Compute each neuron's contribution to each behavior label across all instances and obtain distributions of contributions

% Inputs:
% svmID: Behavior labels for each sample


% Get unique behavior labels
uniqueLabels = unique(bhvID(modelInd));

% Initialize a cell array to store contributions
% contributions{neuron, behavior} will hold the contributions of a neuron to all instances of a specific behavior
numNeurons = length(idSelect);
numBehaviors = length(uniqueLabels);
contributions = cell(numNeurons, numBehaviors);

% Loop through each behavior
for b = 1:numBehaviors
    % Get the indices of the samples corresponding to the current behavior
    labelIdx = (bhvID(modelInd) == uniqueLabels(b));
    
    % Extract the projections (score) for the current behavior
    behaviorScores = score(labelIdx, :);
    
    % Compute contributions for each neuron and store them
    for n = 1:numNeurons
        % Contribution is the sum of projections weighted by the loadings for this neuron
        contributions{n, b} = behaviorScores * coeff(n, :)';
    end
end

% Example: Display the contributions for the first neuron across all behaviors
disp('Contributions of Neuron 1 for each behavior:');
for b = 1:numBehaviors
    fprintf('Behavior %d: ', uniqueLabels(b));
    disp(contributions{1, b}');
end

%% Example: Plot distributions of contributions for the first neuron
iNeuron = 42;
figure(2222); clf;
[ax, pos] = tight_subplot(1, numBehaviors, [.04 .01], .1);
ymax = 0;
for b = 1:numBehaviors
            axes(ax(b))
            if b == 1
    ylabel('Density');
            end
    % subplot(1, numBehaviors, b);
    histogram(contributions{iNeuron, b}, 'Normalization', 'pdf');
    title(['Nrn ', num2str(iNeuron), ' Bhv ' num2str(uniqueLabels(b))]);
    xlabel('Contribution');
    ylimits = ylim;
    ymax = max(ymax, ylimits(2));
end
for b = 1:numBehaviors
            axes(ax(b))
            ylim([0 ymax])
end