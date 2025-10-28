function bhvID = define_reach_bhv_labels(fileName, opts)

% DEFINE_REACH_BHV_LABELS Defines behavioral labels based on reach times
%
% This function assigns behavioral labels to time bins based on reach start and stop times.
% Labels are assigned relative to each reach:
%   1: pre-reach        [-1.0, -0.2] seconds w.r.t. reachStart
%   2: reach            [reachStart-0.2, reachStop] seconds
%   3: pre-reward       [reachStop, reachStart+1] seconds (correct reaches only)
%   4: reward           [reachStart+1, reachStart+2] seconds (correct reaches only)
%   5: post-reward      [reachStart+2, reachStart+4] seconds (correct reaches only)
%   6: intertrial       all other time bins
%
% Note: Labels 3-5 (reward-related phases) are only assigned for correct reaches.
%       Error reaches only get labels 1 and 2.
%
% Inputs:
%   fileName  - Path to reach data file (.mat file containing R and block matrices)
%   frameSize - Size of each time bin in seconds
%   opts      - Structure with optional fields:
%       .collectStart  - Start time (s) of collection window (default: 0)
%       .collectFor    - Number of seconds to collect (default: use all data)
%
% Outputs:
%   bhvID     - Vector of behavioral labels for each time bin
%
% Example:
%   opts.frameSize = 0.05;
%   opts.collectStart = 0;
%   opts.collectFor = 100;
%   bhvID = define_reach_bhv_labels('Y4_100623_Spiketimes_idchan_BEH.mat', 0.05, opts);

% Default options
if nargin < 2
    opts = struct();
end
if ~isfield(opts, 'collectStart')
    opts.collectStart = 0;
end
if ~isfield(opts, 'collectFor')
    opts.collectFor = []; % Will use all data
end
frameSize = opts.frameSize;

% Load reach data
dataR = load(fileName);

% Get reach start and stop times (in milliseconds)
reachStart = dataR.R(:, 1);  % Reach start times in ms
reachStop = dataR.R(:, 2);     % Reach stop times in ms

% Get reach outcomes to determine correct vs error reaches
% reachClass: positive values = correct (reward), negative values = error
reachClass = dataR.Block(:, 2);  % Classification column
isCorrect = reachClass > 0;       % Logical array: true for correct reaches

% Convert from ms to seconds
reachStartSec = reachStart / 1000;
reachStopSec = reachStop / 1000;

% Create time bins for the entire session
% First, find the maximum time (don't filter reaches yet)
maxTime = max(reachStopSec) + 4; % Add buffer for post-reward period
numBins = ceil(maxTime / frameSize);
bhvID = 6 * ones(numBins, 1);  % Initialize all bins as intertrial (label 6)

% For each reach, assign labels to bins within the specified time windows
for r = 1:length(reachStartSec)
    startTime = reachStartSec(r);
    stopTime = reachStopSec(r);
    
    % 2: reach [reachStart-0.2, reachStop] seconds
    reachWinStart = startTime - 0.1;
    reachWinStop = stopTime;

    % Define time windows relative to this reach (in seconds)
    % 1: pre-reach [-1.0, -0.2] seconds w.r.t. reachStart
    preReachStart = startTime - 1.0;
    preReachStop = reachWinStart;
    
    
    % 3: pre-reward [reachStop, reachStart+1] seconds
    preRewardStart = stopTime;
    preRewardStop = startTime + 1.0;
    
    % 4: reward [reachStart+1, reachStart+2] seconds
    rewardStart = startTime + 1.0;
    rewardStop = startTime + 2.0;
    
    % 5: post-reward [reachStart+2, reachStart+4] seconds
    postRewardStart = startTime + 2.0;
    postRewardStop = startTime + 4.0;
    
    % Convert time windows to bin indices
    % Bin i (1-indexed) represents time [(i-1)*frameSize, i*frameSize)
    % For a time window [tStart, tEnd), we want bins that overlap with this interval
    % First bin: floor(tStart / frameSize) + 1
    % Last bin (inclusive): floor((tEnd - 1e-10) / frameSize) + 1
    
    % Label 1: pre-reach
    firstBin = floor(preReachStart / frameSize) + 1;
    lastBin = floor(preReachStop / frameSize);
    if firstBin <= numBins && lastBin >= 1 && firstBin <= lastBin
        binIndices = max(1, firstBin):min(numBins, lastBin);
        bhvID(binIndices) = 1;
    end
    
    % Label 2: reach
    firstBin = floor(reachWinStart / frameSize) + 1;
    lastBin = floor(reachWinStop / frameSize);
    if firstBin <= numBins && lastBin >= 1 && firstBin <= lastBin
        binIndices = max(1, firstBin):min(numBins, lastBin);
        bhvID(binIndices) = 2;
    end
    
    % Labels 3, 4, 5 (reward-related) only for correct reaches
    if isCorrect(r)
        % Label 3: pre-reward
        firstBin = floor(preRewardStart / frameSize) + 1;
        lastBin = floor(preRewardStop / frameSize);
        if firstBin <= numBins && lastBin >= 1 && firstBin <= lastBin
            binIndices = max(1, firstBin):min(numBins, lastBin);
            bhvID(binIndices) = 3;
        end
        
        % Label 4: reward
        firstBin = floor(rewardStart / frameSize) + 1;
        lastBin = floor(rewardStop / frameSize);
        if firstBin <= numBins && lastBin >= 1 && firstBin <= lastBin
            binIndices = max(1, firstBin):min(numBins, lastBin);
            bhvID(binIndices) = 4;
        end
        
        % Label 5: post-reward
        firstBin = floor(postRewardStart / frameSize) + 1;
        lastBin = floor(postRewardStop / frameSize);
        if firstBin <= numBins && lastBin >= 1 && firstBin <= lastBin
            binIndices = max(1, firstBin):min(numBins, lastBin);
            bhvID(binIndices) = 5;
        end
    end
    
    % Note: For error reaches, bins in the reward-related time windows remain as 6 (intertrial)
end

% Trim to match collectFor if specified
if ~isempty(opts.collectFor)
    numBinsToKeep = floor(opts.collectFor / frameSize);
    bhvID = bhvID(1:numBinsToKeep);
end

end

