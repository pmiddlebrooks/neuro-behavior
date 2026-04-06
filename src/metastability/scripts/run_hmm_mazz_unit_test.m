% RUN_HMM_MAZZ_UNIT_TEST Grid-search unit test for HMM recovery quality.
%
% Variables:
%   hmmBinSizeGrid       - Candidate HmmParam.BinSize values (seconds).
%   minDurGrid           - Candidate HmmParam.MinDur values (seconds).
%   modelSelectionMethodList - Candidate model-selection modes to compare.
%   repeatPerSetting     - Number of synthetic datasets per setting.
%
% Goal:
%   Evaluate combinations of hmmBinSize and minDur (and optionally selection
%   method) to find the best recovery of synthetic ground truth.

paths = get_paths;
basePath = fileparts(mfilename('fullpath'));
analysesPath = fullfile(basePath, '..', 'analyses');
if exist(analysesPath, 'dir')
    addpath(analysesPath);
end

dropboxMetastabilityData = fullfile(paths.dropPath, 'metastability');
if ~exist(dropboxMetastabilityData, 'dir')
    mkdir(dropboxMetastabilityData);
end

% -----------------------------
% User-editable search settings
% -----------------------------
hmmBinSizeGrid = [0.005, 0.01, .015];
minDurGrid = [0.04, 0.05];
modelSelectionMethodList = {'XVAL', 'BIC', 'AIC'};
repeatPerSetting = 1;
useParallel = true;
saveData = false;
makePlotsForBest = true;

baseSynthParams = struct();
baseSynthParams.maxStateDur = 3;
baseSynthParams.totalDurationSec = 90;
baseSynthParams.rngSeed = 42;

numComb = numel(hmmBinSizeGrid) * numel(minDurGrid) * ...
    numel(modelSelectionMethodList) * repeatPerSetting;
resultRows = repmat(struct( ...
    'hmmBinSize', NaN, ...
    'minDur', NaN, ...
    'modelSelectionMethod', '', ...
    'repeatIdx', NaN, ...
    'argmaxAccuracy', NaN, ...
    'hardAccuracy', NaN, ...
    'fittedNumStates', NaN, ...
    'numBinsCompared', NaN, ...
    'score', NaN), 1, numComb);

rowIdx = 0;
for methodIdx = 1:numel(modelSelectionMethodList)
    modelSelectionMethod = modelSelectionMethodList{methodIdx};
    for binIdx = 1:numel(hmmBinSizeGrid)
        hmmBinSizeVal = hmmBinSizeGrid(binIdx);
        for durIdx = 1:numel(minDurGrid)
            minDurVal = minDurGrid(durIdx);
            for repIdx = 1:repeatPerSetting
                rowIdx = rowIdx + 1;

                synthParams = baseSynthParams;
                synthParams.hmmBinSize = hmmBinSizeVal;
                synthParams.minDur = minDurVal;
                synthParams.rngSeed = baseSynthParams.rngSeed + repIdx - 1;

                fprintf(['\n[%d/%d] Running unit-test combo: method=%s, ' ...
                    'hmmBinSize=%.4f, minDur=%.4f, repeat=%d\n'], ...
                    rowIdx, numComb, modelSelectionMethod, ...
                    hmmBinSizeVal, minDurVal, repIdx);

                synthBundle = hmm_mazz_unit_test_generate(dropboxMetastabilityData, synthParams);

                dataStruct = struct();
                dataStruct.sessionType = 'spontaneous';
                dataStruct.paths = paths;
                dataStruct.opts = synthBundle.opts;
                dataStruct.areas = synthBundle.areas;
                dataStruct.idList = synthBundle.idList;
                dataStruct.spikeData = synthBundle.spikeData;
                dataStruct.trialDur = synthBundle.trialDur;
                dataStruct.sessionName = sprintf('%s_%s_bin%.3f_minDur%.3f_rep%d', ...
                    synthBundle.sessionName, modelSelectionMethod, hmmBinSizeVal, minDurVal, repIdx);

                config = struct();
                config.modelSelectionMethod = modelSelectionMethod;
                config.minNumNeurons = 15;
                config.saveData = saveData;
                config.useParallel = useParallel;
                config.HmmParam = synthBundle.opts.HmmParam;
                config.HmmParam.BinSize = hmmBinSizeVal;
                config.HmmParam.MinDur = minDurVal;

                results = hmm_mazz_analysis(dataStruct, config);
                unitTestReport = hmm_mazz_unit_test_validate(results, synthBundle.groundTruthStateSeq);

                % Composite score: prioritize MAP accuracy, then hard decode, then K proximity.
                hardAccVal = unitTestReport.meanAccuracyHard;
                if isnan(hardAccVal)
                    hardAccVal = 0;
                end
                scoreVal = unitTestReport.meanAccuracyArgmaxP + ...
                    0.5 * hardAccVal - ...
                    0.02 * abs(unitTestReport.fittedNumStates - synthBundle.numStates);

                resultRows(rowIdx).hmmBinSize = hmmBinSizeVal;
                resultRows(rowIdx).minDur = minDurVal;
                resultRows(rowIdx).modelSelectionMethod = modelSelectionMethod;
                resultRows(rowIdx).repeatIdx = repIdx;
                resultRows(rowIdx).argmaxAccuracy = unitTestReport.meanAccuracyArgmaxP;
                resultRows(rowIdx).hardAccuracy = unitTestReport.meanAccuracyHard;
                resultRows(rowIdx).fittedNumStates = unitTestReport.fittedNumStates;
                resultRows(rowIdx).numBinsCompared = unitTestReport.numBinsCompared;
                resultRows(rowIdx).score = scoreVal;
            end
        end
    end
end

resultTable = struct2table(resultRows);
resultTable = sortrows(resultTable, 'score', 'descend');

fprintf('\n=== HMM Unit-Test Grid Search (Top 10 by score) ===\n');
disp(resultTable(1:min(10, height(resultTable)), :));

[~, bestArgmaxIdx] = max(resultTable.argmaxAccuracy);
bestArgmaxRow = resultTable(bestArgmaxIdx, :);
fprintf(['Best argmax accuracy: method=%s, hmmBinSize=%.4f, minDur=%.4f, ' ...
    'argmax=%.3f, hard=%.3f, fittedK=%d\n'], ...
    bestArgmaxRow.modelSelectionMethod{1}, bestArgmaxRow.hmmBinSize, ...
    bestArgmaxRow.minDur, bestArgmaxRow.argmaxAccuracy, ...
    bestArgmaxRow.hardAccuracy, bestArgmaxRow.fittedNumStates);

if makePlotsForBest
    fprintf('\nRe-running best-score model to generate plots...\n');
    bestRow = resultTable(1, :);
    synthParams = baseSynthParams;
    synthParams.hmmBinSize = bestRow.hmmBinSize;
    synthParams.minDur = bestRow.minDur;
    synthParams.rngSeed = baseSynthParams.rngSeed + bestRow.repeatIdx - 1;
    synthBundle = hmm_mazz_unit_test_generate(dropboxMetastabilityData, synthParams);

    dataStruct = struct();
    dataStruct.sessionType = 'spontaneous';
    dataStruct.paths = paths;
    dataStruct.opts = synthBundle.opts;
    dataStruct.areas = synthBundle.areas;
    dataStruct.idList = synthBundle.idList;
    dataStruct.spikeData = synthBundle.spikeData;
    dataStruct.trialDur = synthBundle.trialDur;
    dataStruct.sessionName = sprintf('%s_best', synthBundle.sessionName);

    config = struct();
    config.modelSelectionMethod = bestRow.modelSelectionMethod{1};
    config.minNumNeurons = 15;
    config.saveData = false;
    config.useParallel = useParallel;
    config.HmmParam = synthBundle.opts.HmmParam;
    config.HmmParam.BinSize = bestRow.hmmBinSize;
    config.HmmParam.MinDur = bestRow.minDur;

    bestResults = hmm_mazz_analysis(dataStruct, config);
    hmm_mazz_unit_test_validate(bestResults, synthBundle.groundTruthStateSeq);
    hmm_mazz_plot(bestResults, struct('brainArea', 'M23'));
end
