%%
% Test script for criticality_decoding_accuracy.m
% This script tests the basic functionality without running the full analysis

%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 5 * 60; % seconds - shorter for testing
opts.minFiringRate = .05;
opts.frameSize = .001;

paths = get_paths;

% Test data loading
fprintf('Testing data loading...\n');
getDataType = 'spikes';
opts.firingRateCheckTime = 5 * 60;
get_standard_data

% Test behavior data loading
fprintf('Testing behavior data loading...\n');
getDataType = 'behavior';
get_standard_data
[dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);

fprintf('Data loading successful!\n');
fprintf('Neural data size: %d x %d\n', size(dataMat));
fprintf('Behavior data size: %d x %d\n', size(bhvID));

% Test criticality functions
fprintf('\nTesting criticality functions...\n');
testData = randn(1000, 1); % Random test data

try
    [varphi, varNoise] = myYuleWalker3(testData, 10);
    fprintf('myYuleWalker3 successful\n');
catch e
    fprintf('myYuleWalker3 failed: %s\n', e.message);
end

try
    d2Val = getFixedPointDistance2(10, 2, varphi);
    fprintf('getFixedPointDistance2 successful\n');
catch e
    fprintf('getFixedPointDistance2 failed: %s\n', e.message);
end

% Test optimal bin/window finding
fprintf('\nTesting optimal bin/window finding...\n');
try
    [optimalBinSize, optimalWindowSize] = find_optimal_bin_and_window(dataMat(:, 1:10), [0.01, 0.02, 0.05], [30, 60, 90], 5, 20, 1000);
    fprintf('find_optimal_bin_and_window successful\n');
catch e
    fprintf('find_optimal_bin_and_window failed: %s\n', e.message);
end

% Test SVM functionality
fprintf('\nTesting SVM functionality...\n');
try
    testFeatures = randn(100, 6);
    testLabels = randi([1, 5], 100, 1);
    
    cv = cvpartition(testLabels, 'HoldOut', 0.2);
    trainData = testFeatures(training(cv), :);
    testData = testFeatures(test(cv), :);
    trainLabels = testLabels(training(cv));
    testLabels = testLabels(test(cv));
    
    t = templateSVM('Standardize', true, 'KernelFunction', 'polynomial');
    svmModel = fitcecoc(trainData, trainLabels, 'Learners', t);
    
    predictedLabels = predict(svmModel, testData);
    accuracy = sum(predictedLabels == testLabels) / length(testLabels);
    fprintf('SVM test successful, accuracy: %.3f\n', accuracy);
catch e
    fprintf('SVM test failed: %s\n', e.message);
end

fprintf('\nTest completed!\n'); 