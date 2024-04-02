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



