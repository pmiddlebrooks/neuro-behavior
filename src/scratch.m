% SaniOG_etal_2021_NatNeuro: Modeling behaviorally relevant neural dynamics enabled by preferential subspace identification
%%
cd '/Users/paulmiddlebrooks/Projects/toolboxes/PSID/'
init
%%
opts = neuro_behavior_options;
get_standard_data
%%
y = dataMat;
% y = dataMat(:, idDS);
z = bhvIDMat;
%%
nx = 5;
n1 = 7;
i = 7;
idSys = PSID(y, z, nx, n1, i);

%% predict
[zPred, yPred, xPred] = PSIDPredict(idSys, y);


%%
R2 = evalPrediction(z, zPred, 'R2')






%%                  PYTHON in MATLAB: Do some python stuff in matlab
% ===========================================================================
p = pyenv
%%
py.sys.version
py.math.sqrt(42)

py.list([1,2])




%%
locBouts = strcmp(dataBhv.Name, 'locomotion');
figure(555);
histogram(dataBhv.Dur(locBouts))

%%
for i = 2 : size(dataBhv) - 1
    if dataBhv.ID(i-1) == dataBhv.ID(i+1) && dataBhv.ID(i-1) ~= dataBhv.ID(i)
        disp(dataBhv(i-5:i+5,:))
        pause
    end
end


%%
opts = neuro_behavior_options;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 2 * 60 * 60; % seconds
opts.frameSize = .05;

getDataType = 'behavior';
get_standard_data

%%
bhvID = bhvIDMat;
opts.possibleBhv = [5:12];
opts.possibleBhv = [0 1 2 13 14 15];
opts.minLength = ceil(2 / opts.frameSize); % Mininum length in sec for a sequence to count
opts.minBhvPct = 90; % Minimum percent in the sequence that has to be within the requested behaviors
opts.maxNonBhv = ceil(.5 / opts.frameSize); % Max consecutive time of a non-requested behavior allowed within the sequence 
opts.minBtwnSeq = 3 / opts.frameSize; % Minimum time between qualifying sequences

[matchingSequences, startIndices, bhvSequences] = find_matching_sequences(bhvID, opts);

% Display the results
fprintf('Number of sequences that count: %d\n', length(matchingSequences))
% disp('Matching sequences and their starting indices:');
% for i = 1:length(matchingSequences)
%     fprintf('Sequence %d (Start Index: %d): \n', i, startIndices(i));
%     disp(matchingSequences{i}');
% end

%%
figure(1); clf
seqL = cellfun(@length, matchingSequences);
edges = 0 : 1 : 60;
histogram(seqL * opts.frameSize, 20)


%%

% Define a sample vector
inputVector = bhvID;
windowSize = floor(1 / opts.frameSize);

% Get the sorted frequencies around each unique integer occurrence
sortedFreqsAll = getSortedFrequenciesForAll(inputVector, windowSize);

%%
% Display the result for each unique integer
uniqueIntegers = fieldnames(sortedFreqsAll);
for i = 1:length(uniqueIntegers)
    uniqueInt = uniqueIntegers{i};
    fprintf('Results for %s:\n', uniqueInt);
    occurrences = sortedFreqsAll.(uniqueInt);
    for j = 1:length(occurrences)
        fprintf('Occurrence %d: ', j);
        disp(occurrences{j}');
    end
end






















% Example Input
% bhvID = randi([1, 5], 1, 1000); % Example behavior labels (1 to 5) over 1000 time points

%%
allLabels = unique(bhvID);
time = 1:length(bhvID);        % Time vector

% Parameters
windowSizes = [10, 50, 100];   % Multi-resolution window sizes
overlapFraction = 0.5;         % Fraction of overlap between windows

% Preallocate Results
segmentationResults = cell(length(windowSizes), 1);

%% Multi-Resolution Segmentation
for wIdx = 1:length(windowSizes)
    windowSize = windowSizes(wIdx);
    stepSize = round(windowSize * (1 - overlapFraction));
    boundaries = []; % Store segment boundaries
    prevDistribution = []; % Initialize previous distribution
    
    % Loop through the sequence
    for startIdx = 1:stepSize:(length(bhvID) - windowSize + 1)
        % Define the current window
        window = bhvID(startIdx:(startIdx + windowSize - 1));
        
        % Compute the distribution of categories in the window
        distribution = histcounts(window, [allLabels', max(allLabels)], 'Normalization', 'probability');
        
        % Compare to previous window (if not the first)
        if ~isempty(prevDistribution)
            % Compute the difference between this and the previous distribution
            diffDist = abs(distribution - prevDistribution);
            
            % Set a threshold to detect boundaries
            if max(diffDist) > 0.5 % Change this threshold as needed
                boundaries = [boundaries, startIdx]; %#ok<AGROW>
            end
        end
        
        % Update the previous distribution
        prevDistribution = distribution;
    end
    
    % Save segmentation results
    segmentationResults{wIdx} = boundaries;
end

%% Visualization
figure;

% Plot the original behavior sequence
subplot(length(windowSizes)+1, 1, 1);
plot(time, double(bhvID), 'k');
title('Original Behavior Sequence');
xlabel('Time');
ylabel('Behavior Labels');

% Plot segmentation results at each resolution
for wIdx = 1:length(windowSizes)
    subplot(length(windowSizes)+1, 1, wIdx+1);
    hold on;
    plot(time, double(bhvID), 'k');
    for b = segmentationResults{wIdx}
        xline(b, '--r', 'LineWidth', 1.5); % Mark boundaries
    end
    title(['Segmentation with Window Size ', num2str(windowSizes(wIdx))]);
    xlabel('Time');
    ylabel('Behavior Labels');
end















%% Example Input
% bhvID = categorical(randi([1, 5], 1, 1000)); % Categorical behavior labels (1 to 5)

% Parameters
allLabels = unique(bhvID);  % All unique behaviors (nodes in the graph)
numLabels = numel(allLabels);

% Transition Matrix Construction
transitionMatrix = zeros(numLabels); % Initialize transition matrix
for i = 1:(length(bhvID) - 1)
    fromIdx = find(allLabels == bhvID(i));       % Index of current behavior
    toIdx = find(allLabels == bhvID(i + 1));    % Index of next behavior
    transitionMatrix(fromIdx, toIdx) = transitionMatrix(fromIdx, toIdx) + 1;
end

% Normalize the Transition Matrix
rowSums = sum(transitionMatrix, 2); % Row sums for normalization
validRows = rowSums > 0;            % Identify rows with valid transitions
normalizedMatrix = transitionMatrix; % Initialize normalized matrix
normalizedMatrix(validRows, :) = normalizedMatrix(validRows, :) ./ rowSums(validRows);

% Graph Construction
% Extract edges from the normalized transition matrix
[s, t, weights] = find(normalizedMatrix); % Nonzero entries as edges
behaviorGraph = digraph(s, t, weights, string(allLabels)); % Create directed graph

% Community Detection (Louvain Method)
% Convert the graph to an undirected version for community detection
undirectedGraph = graph(normalizedMatrix + normalizedMatrix'); % Symmetric adjacency
communities = conncomp(undirectedGraph); % Connected components (simplified community detection)

% Plot the Graph with Communities
figure;
plot(behaviorGraph, 'Layout', 'force', ...
    'NodeLabel', string(allLabels), ...
    'EdgeColor', [0.5, 0.5, 0.5], ...
    'NodeCData', communities, ...
    'LineWidth', 2);
colorbar;
title('Behavior Transition Graph with Community Detection');
xlabel('Communities are indicated by node colors');

% Output Results
disp('Transition Matrix:');
disp(array2table(transitionMatrix, 'VariableNames', string(allLabels), 'RowNames', string(allLabels)));

disp('Community Assignments:');
for i = 1:numLabels
    fprintf('Behavior %s is in Community %d\n', string(allLabels(i)), communities(i));
end











%% Raw high-d spiking data

acc = [.252 .197 .440 .221 .289 .187 .467 .251];
x = [1 2 4 5 7 8 10 11];
%% UMAP low-D (4-dim)

acc = [.273 .221 .365 .284 .291 .213 .463 .233];
acc = [.273 .221 .365 .284 .291 .213 .463 .233];
x = [1 2 4 5 7 8 10 11];


figure(39); clf
b = bar(x, acc, 'FaceColor', 'flat');

colors = [...
    0 0 1;...
    1 0 0;...
    0 0 1;...
    1 0 0;...
    0 0 1;...
    1 0 0;...
    0 0 1;...
    1 0 0;...
    ];
for i = 1:length(colors)
b.CData(i,:) = colors(i,:);
end
xticklabels(' ')
ylabel('Accuracy')
% title('SVM decoding accuracy on raw neural spiking')
t = [selectFrom, ' SVM decoding accuracy on 4D UMAP projections'];
title(t)
% t = [t, 'pdf'];
t = [t, 'png'];
figure_pretty_things
        % print('-dpdf', fullfile(paths.dropPath, t), '-bestfit')
        % print('-dpdf', fullfile(paths.dropPath, t))
        print('-dpng', fullfile(paths.dropPath, t))


%% chatgpt code to save transparent backgorund images

 % Create a sample figure (Replace with your own figure)
fig = figure;
plot(rand(10,1), 'LineWidth', 2);
xlabel('X-axis');
ylabel('Y-axis');
title('Sample Plot');
grid on;

% Set figure background to transparent
set(fig, 'Color', 'none');
ax = gca;
set(ax, 'Color', 'none');

% Define file names
pdfFileName = 'figure_transparent.pdf';
pngFileName = 'figure_transparent.png';

% Save as PDF with transparent background
print(fig, pdfFileName, '-dpdf', '-r300', '-vector');

% Save as PNG with transparent background
print(fig, pngFileName, '-dpng', '-r300', '-transparent');

disp(['PDF saved as ', pdfFileName]);
disp(['PNG saved as ', pngFileName]);















%% 
fromCodes = codes(contains(behaviors, 'groom'));
fromCodes = codes(contains(behaviors, 'ate_2'));
opts.transTo = 15;
opts.transFrom = fromCodes;
opts.minBoutDur = .25;
opts.minTransFromDur = .25;
goodTransitions = find_good_transitions(bhvID, opts);
length(goodTransitions)








%%
function sortedFrequenciesAroundAll = getSortedFrequenciesForAll(inputVector, windowSize)
    % Initialize the output structure
    sortedFrequenciesAroundAll = struct();
    
    % Get the unique integers in the input vector
    uniqueIntegers = unique(inputVector);
    
    % Adjust window size to be symmetric
    halfWindow = floor(windowSize / 2);
    
    % Loop over each unique integer
    for uniqueInt = transpose(uniqueIntegers) % Ensure column vector for iteration
        % Identify indices of the current unique integer
        targetIndices = find(inputVector == uniqueInt);
        
        % Initialize a cell array for storing sorted frequencies for each occurrence
        sortedFrequenciesAroundTarget = cell(1, length(targetIndices));
        
        % Loop over each occurrence of the current unique integer
        for i = 1:length(targetIndices)
            idx = targetIndices(i);
            
            % Determine the start and end indices of the window
            startIdx = max(1, idx - halfWindow);
            endIdx = min(length(inputVector), idx + halfWindow);
            
            % Extract window elements including the target itself
            windowElements = inputVector(startIdx:endIdx);
            
            % Count the frequency of each unique element in the window
            uniqueElements = unique(windowElements);
            counts = zeros(length(uniqueElements), 1);
            for j = 1:length(uniqueElements)
                counts(j) = sum(windowElements == uniqueElements(j));
            end
            
            % Sort the elements by frequency (descending) and then by value (ascending for ties)
            [~, sortIdx] = sortrows([counts, -uniqueElements], [-1, 2]);
            sortedUniqueElements = uniqueElements(sortIdx);
            
            % Store the sorted elements by frequency in the current cell
            sortedFrequenciesAroundTarget{i} = sortedUniqueElements;
        end
        
        % Assign to output structure
        sortedFrequenciesAroundAll.(['Int' num2str(uniqueInt)]) = sortedFrequenciesAroundTarget;
    end
end


