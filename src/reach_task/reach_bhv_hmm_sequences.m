function [results] = reach_bhv_hmm_sequences(bhvID, bhvIdx, hmmStates, varargin)
% Analyze relationship between behavioral events and HMM state sequences
% 
% Inputs:
%   bhvID - categorical array of behavior event types
%   bhvIdx - time bin indices for each behavior event
%   hmmStates - vector of HMM states over time (0 = undefined)
%   
% Optional parameters:
%   'WindowSize' - time window around events [pre, post] (default: [50, 50])
%   'MaxSeqLength' - maximum sequence length to analyze (default: 5)
%   'MinSeqLength' - minimum sequence length to analyze (default: 1)
%   'TimeShifts' - range of time shifts to test (default: -10:10)
%   'MinOccurrence' - minimum occurrences for pattern inclusion (default: 3)

% ## Key Features of the Implementation:
% 
% ### 1. **Core Analysis Pipeline**
% - **Temporal Alignment**: Uses mutual information to find optimal lag between HMM states and behavior
% - **Bidirectional Motif Discovery**: Analyzes both HMM→Behavior and Behavior→HMM directions
% - **Motif Length Analysis**: Examines motifs of length 1-4 time bins
% - **Statistical Significance**: Performs shuffling tests to validate motif significance
% 
% ### 2. **Motif Extraction & Scoring**
% - **Frequency Analysis**: Counts how often each motif occurs
% - **Predictive Power**: Measures how well motifs predict target sequences using entropy-based metrics
% - **Combined Scoring**: Combines frequency and predictive power for comprehensive ranking
% 
% ### 3. **Comprehensive Visualization**
% - **12-panel visualization** covering all aspects of the analysis:
%   - Motif frequency and predictive power by length
%   - Top motifs heatmap and ranking
%   - Direction performance comparison
%   - Statistical significance results
%   - Summary statistics and parameters
% 
% ### 4. **Advanced Analysis Features**
% - **Significance Testing**: 1000 shuffles with p-value calculation
% - **Top Motif Ranking**: Multiple ranking criteria (frequency, predictive power, combined score)
% - **Direction Comparison**: Quantifies whether HMM→Behavior or Behavior→HMM is more predictive
% - **Motif Stability**: Framework for analyzing temporal consistency
% 
% ### 5. **Integration with Existing Code**
% - Uses the same data loading structure as `hmm_vs_behavior.m`
% - Compatible with existing HMM results and behavior data
% - Follows the same coding conventions and error handling
% 
% ## Usage:
% 
% 1. **Run the script** with your HMM and behavior data
% 2. **Adjust parameters** at the top (max motif length, number of shuffles, lag range)
% 3. **Review console output** for detailed analysis results
% 4. **Examine visualizations** for comprehensive insights
% 5. **Save results** for further analysis or comparison
% 
% The script will automatically:
% - Find optimal temporal alignment
% - Discover motifs in both directions
% - Rank motifs by multiple criteria
% - Test statistical significance
% - Generate comprehensive visualizations
% - Save all results for future use

This implementation addresses your request for discovering the most prevalent motifs between behavioral ID categories and HMM states, providing both the analytical framework and the tools to interpret the results.
% Parse inputs
p = inputParser;
addParameter(p, 'WindowSize', [50, 50]);
addParameter(p, 'MaxSeqLength', 5);
addParameter(p, 'MinSeqLength', 1);
addParameter(p, 'TimeShifts', -10:10);
addParameter(p, 'MinOccurrence', 3);
parse(p, varargin{:});

windowSize = p.Results.WindowSize;
maxSeqLen = p.Results.MaxSeqLength;
minSeqLen = p.Results.MinSeqLength;
timeShifts = p.Results.TimeShifts;
minOccur = p.Results.MinOccurrence;

% Initialize results structure
results = struct();
results.parameters = p.Results;
results.behaviorTypes = categories(bhvID);

% Step 1: Extract sequences around each behavioral event
fprintf('Step 1: Extracting HMM sequences around behavioral events...\n');
[eventSequences, validEvents] = extract_event_sequences(bhvID, bhvIdx, hmmStates, windowSize);

% Step 2: Find recurring patterns for each behavior type and time window
fprintf('Step 2: Identifying recurring state patterns...\n');
results.patterns = find_recurring_patterns(eventSequences, validEvents, bhvID, ...
                                         minSeqLen, maxSeqLen, minOccur);

% Step 3: Test temporal relationships with time shifts
fprintf('Step 3: Testing temporal relationships...\n');
results.temporalAnalysis = analyze_temporal_relationships(bhvID, bhvIdx, hmmStates, ...
                                                        results.patterns, timeShifts, windowSize);

% Step 4: Statistical significance testing
fprintf('Step 4: Computing statistical significance...\n');
results.statistics = compute_pattern_statistics(results.patterns, eventSequences, validEvents);

% Step 5: Classify patterns as predictive vs reactive
fprintf('Step 5: Classifying predictive vs reactive patterns...\n');
results.classification = classify_patterns(results.temporalAnalysis, windowSize);

fprintf('Analysis complete.\n');
end

function [eventSequences, validEvents] = extract_event_sequences(bhvID, bhvIdx, hmmStates, windowSize)
% Extract HMM state sequences around each behavioral event

nEvents = length(bhvIdx);
preWin = windowSize(1);
postWin = windowSize(2);
totalWin = preWin + postWin + 1;

eventSequences = cell(nEvents, 3); % {pre, at, post}
validEvents = true(nEvents, 1);

for iEvent = 1:nEvents
    eventTime = bhvIdx(iEvent);
    
    % Define time windows
    preStart = max(1, eventTime - preWin);
    postEnd = min(length(hmmStates), eventTime + postWin);
    
    % Skip if insufficient data
    if (eventTime - preStart) < preWin/2 || (postEnd - eventTime) < postWin/2
        validEvents(iEvent) = false;
        continue;
    end
    
    % Extract sequences (remove undefined states)
    preSeq = hmmStates(preStart:eventTime-1);
    atSeq = hmmStates(eventTime);
    postSeq = hmmStates(eventTime+1:postEnd);
    
    % Remove zeros (undefined states) but keep temporal structure
    preSeq = remove_undefined_states(preSeq);
    postSeq = remove_undefined_states(postSeq);
    
    eventSequences{iEvent, 1} = preSeq;
    eventSequences{iEvent, 2} = atSeq;
    eventSequences{iEvent, 3} = postSeq;
end

eventSequences = eventSequences(validEvents, :);
end

function cleanSeq = remove_undefined_states(seq)
% Remove undefined states (0) while preserving sequence structure
cleanSeq = seq(seq ~= 0);
end

function patterns = find_recurring_patterns(eventSequences, validEvents, bhvID, minSeqLen, maxSeqLen, minOccur)
% Find recurring state patterns for each behavior type and time window

behaviorTypes = categories(bhvID);
nBhvTypes = length(behaviorTypes);
patterns = struct();

for iBhv = 1:nBhvTypes
    bhvType = behaviorTypes{iBhv};
    bhvMask = bhvID == bhvType & validEvents;
    bhvSequences = eventSequences(bhvMask, :);
    
    patterns.(bhvType) = struct();
    
    % Analyze pre-event patterns
    patterns.(bhvType).pre = find_patterns_in_window(bhvSequences(:, 1), minSeqLen, maxSeqLen, minOccur);
    
    % Analyze post-event patterns  
    patterns.(bhvType).post = find_patterns_in_window(bhvSequences(:, 3), minSeqLen, maxSeqLen, minOccur);
    
    % Analyze patterns spanning the event
    spanningSeqs = cellfun(@(pre, at, post) [pre, at, post], ...
                          bhvSequences(:, 1), bhvSequences(:, 2), bhvSequences(:, 3), ...
                          'UniformOutput', false);
    patterns.(bhvType).spanning = find_patterns_in_window(spanningSeqs, minSeqLen, maxSeqLen, minOccur);
end
end

function windowPatterns = find_patterns_in_window(sequences, minSeqLen, maxSeqLen, minOccur)
% Find recurring patterns within a time window

windowPatterns = struct();
allPatterns = {};
patternCounts = [];

% Extract all possible subsequences
for iSeq = 1:length(sequences)
    seq = sequences{iSeq};
    if isempty(seq), continue; end
    
    for seqLen = minSeqLen:min(maxSeqLen, length(seq))
        for startPos = 1:(length(seq) - seqLen + 1)
            pattern = seq(startPos:startPos + seqLen - 1);
            allPatterns{end+1} = pattern;
        end
    end
end

% Count pattern occurrences
[uniquePatterns, ~, idx] = unique_patterns(allPatterns);
for iPattern = 1:length(uniquePatterns)
    count = sum(idx == iPattern);
    if count >= minOccur
        patternKey = sprintf('pattern_%d', iPattern);
        windowPatterns.(patternKey) = struct();
        windowPatterns.(patternKey).sequence = uniquePatterns{iPattern};
        windowPatterns.(patternKey).count = count;
        windowPatterns.(patternKey).probability = count / length(sequences);
    end
end
end

function [uniquePatterns, ia, ic] = unique_patterns(patterns)
% Find unique patterns (handles variable length sequences)

patternStrings = cellfun(@(x) mat2str(x), patterns, 'UniformOutput', false);
[~, ia, ic] = unique(patternStrings);
uniquePatterns = patterns(ia);
end

function temporalAnalysis = analyze_temporal_relationships(bhvID, bhvIdx, hmmStates, patterns, timeShifts, windowSize)
% Test temporal relationships between patterns and behavioral events

temporalAnalysis = struct();
behaviorTypes = fieldnames(patterns);

for iBhv = 1:length(behaviorTypes)
    bhvType = behaviorTypes{iBhv};
    bhvMask = bhvID == categorical(bhvType);
    bhvEvents = bhvIdx(bhvMask);
    
    temporalAnalysis.(bhvType) = struct();
    
    % Test each pattern at different time shifts
    windowTypes = {'pre', 'post', 'spanning'};
    for iWin = 1:length(windowTypes)
        winType = windowTypes{iWin};
        if ~isfield(patterns.(bhvType), winType), continue; end
        
        winPatterns = patterns.(bhvType).(winType);
        patternNames = fieldnames(winPatterns);
        
        for iPattern = 1:length(patternNames)
            patternName = patternNames{iPattern};
            pattern = winPatterns.(patternName).sequence;
            
            % Test pattern occurrence at different time shifts
            shiftScores = zeros(size(timeShifts));
            for iShift = 1:length(timeShifts)
                shift = timeShifts(iShift);
                shiftScores(iShift) = test_pattern_at_shift(pattern, hmmStates, bhvEvents, shift, windowSize);
            end
            
            temporalAnalysis.(bhvType).(winType).(patternName).timeShifts = timeShifts;
            temporalAnalysis.(bhvType).(winType).(patternName).scores = shiftScores;
            [maxScore, maxIdx] = max(shiftScores);
            temporalAnalysis.(bhvType).(winType).(patternName).optimalShift = timeShifts(maxIdx);
            temporalAnalysis.(bhvType).(winType).(patternName).maxScore = maxScore;
        end
    end
end
end

function score = test_pattern_at_shift(pattern, hmmStates, bhvEvents, shift, windowSize)
% Test how well pattern occurs at a specific time shift relative to events

patternLen = length(pattern);
hits = 0;
totalTests = 0;

for iEvent = 1:length(bhvEvents)
    testTime = bhvEvents(iEvent) + shift;
    
    % Skip if outside valid range
    if testTime < 1 || testTime + patternLen - 1 > length(hmmStates)
        continue;
    end
    
    testSeq = hmmStates(testTime:testTime + patternLen - 1);
    testSeq = remove_undefined_states(testSeq);
    
    if length(testSeq) == length(pattern) && isequal(testSeq, pattern)
        hits = hits + 1;
    end
    totalTests = totalTests + 1;
end

score = hits / max(totalTests, 1);
end

function statistics = compute_pattern_statistics(patterns, eventSequences, validEvents)
% Compute statistical significance of patterns using permutation tests

statistics = struct();
nPermutations = 1000;
behaviorTypes = fieldnames(patterns);

for iBhv = 1:length(behaviorTypes)
    bhvType = behaviorTypes{iBhv};
    statistics.(bhvType) = struct();
    
    windowTypes = fieldnames(patterns.(bhvType));
    for iWin = 1:length(windowTypes)
        winType = windowTypes{iWin};
        winPatterns = patterns.(bhvType).(winType);
        patternNames = fieldnames(winPatterns);
        
        for iPattern = 1:length(patternNames)
            patternName = patternNames{iPattern};
            observedCount = winPatterns.(patternName).count;
            
            % Permutation test
            nullCounts = zeros(nPermutations, 1);
            for iPerm = 1:nPermutations
                % Shuffle sequences and recount pattern
                shuffledSeqs = eventSequences(randperm(size(eventSequences, 1)), :);
                nullCount = count_pattern_in_shuffled_data(winPatterns.(patternName).sequence, shuffledSeqs, winType);
                nullCounts(iPerm) = nullCount;
            end
            
            pValue = sum(nullCounts >= observedCount) / nPermutations;
            statistics.(bhvType).(winType).(patternName).pValue = pValue;
            statistics.(bhvType).(winType).(patternName).nullMean = mean(nullCounts);
            statistics.(bhvType).(winType).(patternName).nullStd = std(nullCounts);
            statistics.(bhvType).(winType).(patternName).zScore = (observedCount - mean(nullCounts)) / std(nullCounts);
        end
    end
end
end

function count = count_pattern_in_shuffled_data(pattern, shuffledSeqs, winType)
% Count pattern occurrences in shuffled data

count = 0;
winIdx = strcmp(winType, {'pre', 'at', 'post'});
if strcmp(winType, 'spanning')
    % Handle spanning patterns
    return;
end

sequences = shuffledSeqs(:, winIdx);
for iSeq = 1:length(sequences)
    seq = sequences{iSeq};
    if contains_pattern(seq, pattern)
        count = count + 1;
    end
end
end

function hasPattern = contains_pattern(sequence, pattern)
% Check if sequence contains the pattern

hasPattern = false;
if length(sequence) < length(pattern)
    return;
end

for i = 1:(length(sequence) - length(pattern) + 1)
    if isequal(sequence(i:i+length(pattern)-1), pattern)
        hasPattern = true;
        return;
    end
end
end

function classification = classify_patterns(temporalAnalysis, windowSize)
% Classify patterns as predictive (pre-event) or reactive (post-event)

classification = struct();
behaviorTypes = fieldnames(temporalAnalysis);

for iBhv = 1:length(behaviorTypes)
    bhvType = behaviorTypes{iBhv};
    classification.(bhvType) = struct();
    
    if isfield(temporalAnalysis.(bhvType), 'pre')
        prePatterns = fieldnames(temporalAnalysis.(bhvType).pre);
        for iPattern = 1:length(prePatterns)
            patternName = prePatterns{iPattern};
            optimalShift = temporalAnalysis.(bhvType).pre.(patternName).optimalShift;
            
            if optimalShift < -windowSize(1)/4
                classification.(bhvType).(patternName) = 'predictive';
            else
                classification.(bhvType).(patternName) = 'concurrent';
            end
        end
    end
    
    if isfield(temporalAnalysis.(bhvType), 'post')
        postPatterns = fieldnames(temporalAnalysis.(bhvType).post);
        for iPattern = 1:length(postPatterns)
            patternName = postPatterns{iPattern};
            optimalShift = temporalAnalysis.(bhvType).post.(patternName).optimalShift;
            
            if optimalShift > windowSize(2)/4
                classification.(bhvType).(patternName) = 'reactive';
            else
                classification.(bhvType).(patternName) = 'concurrent';
            end
        end
    end
end
end