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
opts = neuro_behavior_options;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 2 * 60 * 60; % seconds
opts.frameSize = .05;

getDataType = 'behavior';
get_standard_data

%%
bhvID = bhvIDMat;
opts.possibleBhv = [5:12];
% opts.possibleBhv = [0 1 2 3 13 14 15];
opts.minLength = ceil(2.5 / opts.frameSize); % Mininum length in sec for a sequence to count
opts.minBhvPct = 90; % Minimum percent in the sequence that has to be within the requested behaviors
opts.maxNonBhv = ceil(.5 / opts.frameSize); % Max consecutive time of a non-requested behavior allowed within the sequence 
opts.minBtwnSeq = 5 / opts.frameSize; % Minimum time between qualifying sequences

[matchingSequences, startIndices, bhvSequences] = find_matching_sequences(bhvID, opts);

% Display the results
fprintf('Number of sequences that count: %d\n', length(matchingSequences))
% disp('Matching sequences and their starting indices:');
% for i = 1:length(matchingSequences)
%     fprintf('Sequence %d (Start Index: %d): \n', i, startIndices(i));
%     disp(matchingSequences{i}');
% end

%%
seqL = cellfun(@length, matchingSequences)
histogram(seqL * opts.frameSize, 20)
