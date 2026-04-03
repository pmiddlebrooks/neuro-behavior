function synthBundle = hmm_mazz_unit_test_generate(dataOutputDir)
% HMM_MAZZ_UNIT_TEST_GENERATE Synthetic Poisson spikes for HMM pipeline testing.
%
% Variables:
%   dataOutputDir - Directory where hmm_mazz_unit_test_synth.mat is written.
%
% Goal:
%   Simulate 20 neurons with piecewise-constant Poisson rates in 5 hidden states,
%   each state lasting blockDurSec (3 s), cycling 1..5. Rates are distinct per
%   (state, neuron) in [0.5, 30] Hz. Emissions match the toolbox pipeline:
%   spikes -> hmm.Spikes2Seq -> Bernoulli-style single-symbol bins (see
%   metastability/hmm_mazz_toolbox/+hmm/low_bernoulli.m).
%
% Output .mat contains spikeData, idList, areas, opts, groundTruthStateSeq,
% ratesHz, and metadata for hmm_mazz_analysis / hmm_mazz_unit_test_validate.

if nargin < 1 || isempty(dataOutputDir)
    error('dataOutputDir is required.');
end
if ~exist(dataOutputDir, 'dir')
    mkdir(dataOutputDir);
end

rng(42, 'twister');

numStates = 5;
numNeurons = 20;
blockDurSec = 3;
totalDurationSec = 120;
binSizeTruth = 0.01;

% Distinct rates per state-neuron, spanning [0.5, 30] Hz
baseRatesHz = linspace(0.5, 30, numNeurons);
ratesHz = zeros(numStates, numNeurons);
for stateIdx = 1:numStates
    ratesHz(stateIdx, :) = circshift(baseRatesHz, (stateIdx - 1) * 3);
end

% Ground-truth discrete state at each HMM bin (for validation)
numBinsTruth = floor(totalDurationSec / binSizeTruth);
groundTruthStateSeq = zeros(1, numBinsTruth, 'uint8');
for binIdx = 1:numBinsTruth
    tEdge = (binIdx - 1) * binSizeTruth;
    blockNum = floor(tEdge / blockDurSec);
    groundTruthStateSeq(binIdx) = uint8(mod(blockNum, numStates) + 1);
end

% Poisson spike times: independent neurons, piecewise constant rate
spikeLists = cell(1, numNeurons);
for neuronIdx = 1:numNeurons
    spikeLists{neuronIdx} = [];
end

timeCursor = 0;
blockNum = 0;
while timeCursor < totalDurationSec
    stateIdx = mod(blockNum, numStates) + 1;
    segmentEnd = min(timeCursor + blockDurSec, totalDurationSec);
    segmentDur = segmentEnd - timeCursor;
    if segmentDur <= 0
        break;
    end
    for neuronIdx = 1:numNeurons
        lambdaHz = ratesHz(stateIdx, neuronIdx);
        numSpikes = poissrnd(lambdaHz * segmentDur);
        if numSpikes > 0
            relTimes = sort(rand(numSpikes, 1) * segmentDur);
            spikeLists{neuronIdx} = [spikeLists{neuronIdx}, timeCursor + relTimes']; %#ok<AGROW>
        end
    end
    timeCursor = segmentEnd;
    blockNum = blockNum + 1;
end

% spikeData columns: [time_s, neuronId, areaIdx] (single area M23, ids 1..20)
spikeRows = zeros(0, 3);
neuronIds = 1:numNeurons;
for neuronIdx = 1:numNeurons
    spk = spikeLists{neuronIdx}(:);
    if isempty(spk)
        continue;
    end
    spikeRows = [spikeRows; [spk, neuronIdx * ones(numel(spk), 1), ones(numel(spk), 1)]]; %#ok<AGROW>
end
[~, sortOrd] = sort(spikeRows(:, 1));
spikeData = spikeRows(sortOrd, :);

areas = {'M23', 'M56', 'DS', 'VS'};
idList = {neuronIds, [], [], []};
sessionName = 'hmm_unit_test_synth';
trialDur = numStates * blockDurSec;

opts = neuro_behavior_options;
opts.collectStart = 0;
opts.collectEnd = totalDurationSec;
opts.frameSize = 0.001;
opts.minActTime = 0.16;
opts.minFiringRate = 0.1;

hmmParam = struct();
hmmParam.AdjustT = 0.0;
hmmParam.BinSize = binSizeTruth;
hmmParam.MinDur = 0.04;
hmmParam.MinP = 0.7;
hmmParam.NumSteps = 6;
hmmParam.NumRuns = 25;
hmmParam.singleSeqXval.K = 7;
opts.HmmParam = hmmParam;

synthBundle = struct();
synthBundle.spikeData = spikeData;
synthBundle.idList = idList;
synthBundle.areas = areas;
synthBundle.sessionName = sessionName;
synthBundle.trialDur = trialDur;
synthBundle.opts = opts;
synthBundle.groundTruthStateSeq = groundTruthStateSeq;
synthBundle.ratesHz = ratesHz;
synthBundle.blockDurSec = blockDurSec;
synthBundle.totalDurationSec = totalDurationSec;
synthBundle.binSizeTruth = binSizeTruth;
synthBundle.numStates = numStates;
synthBundle.numNeurons = numNeurons;

outFile = fullfile(dataOutputDir, 'hmm_mazz_unit_test_synth.mat');
save(outFile, ...
    'spikeData', 'idList', 'areas', 'sessionName', 'trialDur', 'opts', ...
    'groundTruthStateSeq', 'ratesHz', 'blockDurSec', 'totalDurationSec', ...
    'binSizeTruth', 'numStates', 'numNeurons', '-v7.3');
fprintf('Saved synthetic HMM unit-test dataset:\n%s\n', outFile);

end
