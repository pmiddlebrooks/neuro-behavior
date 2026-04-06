function synthBundle = hmm_mazz_unit_test_generate(dataOutputDir, params)
% HMM_MAZZ_UNIT_TEST_GENERATE Synthetic Poisson spikes for HMM pipeline testing.
%
% Variables:
%   dataOutputDir - Directory where hmm_mazz_unit_test_synth.mat is written.
%   params        - Optional struct; fields (all optional):
%       .minDur        - HmmParam.MinDur (s) and minimum segment duration for
%                       ground-truth / Poisson blocks (default 0.04).
%       .maxStateDur   - Upper bound of uniform random duration per state block (s);
%                       each block length ~ U(minDur, maxStateDur) (default 3).
%       .hmmBinSize    - HMM bin size (s) (default 0.01).
%       .totalDurationSec - Session length (default 120).
%       .rngSeed       - Scalar; if set, rng(rngSeed,'twister') before schedule.
%
% Goal:
%   Simulate 20 neurons with piecewise-constant continuous-time Poisson spikes
%   in numStates hidden states, cycling 1..numStates in order. Each state block
%   lasts a random duration in [minDur, maxStateDur] (uniform). Rates are
%   distinct per (state, neuron) in [0.5, 30] Hz. hmmBinSize sets the HMM grid;
%   groundTruthStateSeq uses that grid. opts.HmmParam.MinDur = params.minDur.
%
% Output .mat contains spikeData, idList, areas, opts, groundTruthStateSeq,
% ratesHz, segment schedule, and metadata for hmm_mazz_analysis /
% hmm_mazz_unit_test_validate.

if nargin < 1 || isempty(dataOutputDir)
    error('dataOutputDir is required.');
end
if nargin < 2 || isempty(params)
    params = struct();
end
if ~exist(dataOutputDir, 'dir')
    mkdir(dataOutputDir);
end

minDurSec = local_get_param(params, 'minDur', 0.04);
maxStateDurSec = local_get_param(params, 'maxStateDur', 5);
hmmBinSize = local_get_param(params, 'hmmBinSize', 0.01);
totalDurationSec = local_get_param(params, 'totalDurationSec', 90);
rngSeed = [];
if isfield(params, 'rngSeed') && ~isempty(params.rngSeed)
    rngSeed = params.rngSeed;
end

if maxStateDurSec < minDurSec
    error('maxStateDur (%.4g) must be >= minDur (%.4g).', maxStateDurSec, minDurSec);
end
if minDurSec <= 0 || maxStateDurSec <= 0
    error('minDur and maxStateDur must be positive.');
end

if isempty(rngSeed)
    rng(42, 'twister');
else
    rng(rngSeed, 'twister');
end

numStates = 5;
numNeurons = 20;

% Distinct rates per state-neuron, spanning [0.5, 30] Hz
baseRatesHz = linspace(0.5, 30, numNeurons);
ratesHz = zeros(numStates, numNeurons);
for stateIdx = 1:numStates
    ratesHz(stateIdx, :) = circshift(baseRatesHz, (stateIdx - 1) * 3);
end

% Segment schedule: cycle states 1..numStates in order; random duration per block
[segmentStartSec, segmentDurSec, segmentStateLabel] = local_build_segment_schedule( ...
    numStates, totalDurationSec, minDurSec, maxStateDurSec);

% Ground-truth state at each HMM bin (left edge time falls in [start, start+dur))
numBinsHmm = floor(totalDurationSec / hmmBinSize);
groundTruthStateSeq = zeros(1, numBinsHmm, 'uint8');
for binIdx = 1:numBinsHmm
    tEdge = (binIdx - 1) * hmmBinSize;
    segIdx = local_find_segment_for_time(tEdge, segmentStartSec, segmentDurSec);
    groundTruthStateSeq(binIdx) = uint8(segmentStateLabel(segIdx));
end

% Continuous-time Poisson per segment
spikeLists = cell(1, numNeurons);
for neuronIdx = 1:numNeurons
    spikeLists{neuronIdx} = [];
end

numSeg = numel(segmentDurSec);
for segIdx = 1:numSeg
    stateIdx = segmentStateLabel(segIdx);
    timeCursor = segmentStartSec(segIdx);
    segmentDur = segmentDurSec(segIdx);
    if segmentDur <= 0
        continue;
    end
    for neuronIdx = 1:numNeurons
        lambdaHz = ratesHz(stateIdx, neuronIdx);
        numSpikes = poissrnd(lambdaHz * segmentDur);
        if numSpikes > 0
            relTimes = sort(rand(numSpikes, 1) * segmentDur);
            spikeLists{neuronIdx} = [spikeLists{neuronIdx}, timeCursor + relTimes']; %#ok<AGROW>
        end
    end
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
trialDur = numStates * (minDurSec + maxStateDurSec) / 2;

opts = neuro_behavior_options;
opts.collectStart = 0;
opts.collectEnd = totalDurationSec;
opts.frameSize = 0.001;
opts.minActTime = 0.16;
opts.minFiringRate = 0.1;

hmmParam = struct();
hmmParam.AdjustT = 0.0;
hmmParam.BinSize = hmmBinSize;
hmmParam.MinDur = minDurSec;
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
synthBundle.minDur = minDurSec;
synthBundle.maxStateDur = maxStateDurSec;
synthBundle.segmentStartSec = segmentStartSec;
synthBundle.segmentDurSec = segmentDurSec;
synthBundle.segmentStateLabel = segmentStateLabel;
synthBundle.totalDurationSec = totalDurationSec;
synthBundle.hmmBinSize = hmmBinSize;
synthBundle.numStates = numStates;
synthBundle.numNeurons = numNeurons;

outFile = fullfile(dataOutputDir, 'hmm_mazz_unit_test_synth.mat');
save(outFile, ...
    'spikeData', 'idList', 'areas', 'sessionName', 'trialDur', 'opts', ...
    'groundTruthStateSeq', 'ratesHz', 'minDurSec', 'maxStateDurSec', ...
    'segmentStartSec', 'segmentDurSec', 'segmentStateLabel', ...
    'totalDurationSec', 'hmmBinSize', 'numStates', 'numNeurons', '-v7.3');
fprintf('Saved synthetic HMM unit-test dataset:\n%s\n', outFile);

end

function val = local_get_param(params, fieldName, defaultVal)
% LOCAL_GET_PARAM Read params.fieldName or return defaultVal.

if isfield(params, fieldName) && ~isempty(params.(fieldName))
    val = params.(fieldName);
else
    val = defaultVal;
end
end

function [segmentStartSec, segmentDurSec, segmentStateLabel] = ...
    local_build_segment_schedule(numStates, totalDurationSec, minDurSec, maxStateDurSec)
% LOCAL_BUILD_SEGMENT_SCHEDULE Random block durations, ordered state cycle.

segmentStartSec = [];
segmentDurSec = [];
segmentStateLabel = [];
timeCursor = 0;
cycleIdx = 0;
while timeCursor < totalDurationSec - 1e-12
    stateIdx = mod(cycleIdx, numStates) + 1;
    span = maxStateDurSec - minDurSec;
    dur = minDurSec + span * rand;
    dur = min(dur, totalDurationSec - timeCursor);
    if dur <= 1e-12
        break;
    end
    segmentStartSec(end + 1) = timeCursor; %#ok<AGROW>
    segmentDurSec(end + 1) = dur; %#ok<AGROW>
    segmentStateLabel(end + 1) = stateIdx; %#ok<AGROW>
    timeCursor = timeCursor + dur;
    cycleIdx = cycleIdx + 1;
end
end

function segIdx = local_find_segment_for_time(tSec, segmentStartSec, segmentDurSec)
% LOCAL_FIND_SEGMENT_FOR_TIME Index of segment containing tSec (left-closed).

nSeg = numel(segmentStartSec);
segIdx = nSeg;
for k = 1:nSeg
    t0 = segmentStartSec(k);
    t1 = t0 + segmentDurSec(k);
    if tSec >= t0 && tSec < t1
        segIdx = k;
        return;
    end
end
% tSec on last boundary or numerical end: use last segment
if tSec >= segmentStartSec(end)
    segIdx = nSeg;
end
end
