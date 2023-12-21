function opts = neuro_behavior_options
%rrm_optsions 
% 
% Returns default opts structure for use as input to ridge regression
% analyses

% opts.behaviors = 'all';
% opts.behaviors = {'unlabeled', 'torso groom', 'investigate type 0', 'unsure 1', 'wall rear type 1', 'rear', 'unsure 2', 'wall rear type 2', 'investigate type 1', 'investigate type 2', 'contra-itch', 'investigate type 3', 'sleep/scrunch type 1', 'sleep/scrunch type 2', 'wall rear type 3', 'contra-body groom', 'face groom type 1', 'dive/scrunch', 'head groom', 'ipsi-orient', 'ipsi-investigate', 'face groom type 2', 'ipsi-body groom', 'ipsi-itch type 1', 'ipsi-itch type 2', 'face groom type 3', 'paw groom', 'locomotion', 'contra-forepaw orient', 'contra-orient'};
% opts.bhvLabel = (-1 : length(opts.behaviors) - 2);


% sampling frequency for various collected data
opts.fsBhv = 60;
opts.fsSpike = 30000;
opts.fsLfp = 1250;

opts.frameSize = .1; % how many s per time step (frame)
opts.windowSize = 2; % (sec): use a 2 second window (one second before and after behavior onset)
opts.framesPerTrial = floor(opts.windowSize / opts.frameSize);

% To capture a window centered on start of behavior, shift the time
% bins by half their width. Thus spike counts will be -100 to +100
% peri-behavior start
opts.shiftAlignFactor = 0; % -.5 will make windows centered on regressor events instead of flanking them before/after

% How many frames do we want to regress before and after events?
opts.sPostTime = ceil(1 / opts.frameSize);   % frames: follow stim events for 1000ms for sPostStim (used for eventType 2)
opts.mPreTime = ceil(1 / opts.frameSize);  % frames: precede motor events by 1000 ms to capture preparatory activity (used for eventType 3)
opts.mPostTime = ceil(1 / opts.frameSize) - 1;   % frames: follow motor eventS for 1000 ms for mPostStim (used for eventType 3). Remove one frame since behavior starts at time zero.

opts.folds = 10; %nr of folds for cross-validation


% Use 30 minutes of data, because 1) neurons move around and 2) animals change beahvioral states
opts.collectStart = 5 * 60; % (sec) When to start collecting the data
opts.collectFor = 30 * 60; % (sec) How long to collect data

%  Balance the data (take nTrial of each behavior)
opts.nTrialBalance = 200;



% Only count (use) a behavior if it...
opts.minActTime = .099999; % is at least minActTime long (in seconds): this is .099999 instead of .1 b/c matlab precision thinks .1 doesn't equal .1 (see rrm_behavior_selection)
opts.minNoRepeatTime = .99999; % Can't have occured this much time in the past (in seconds)
opts.minBhvNum = 20;  % must have occured at least this many times to analyze

% For one-back (previous behavior) analyses
opts.minOneBackNum = 40;
opts.nOneBackKeep = 6;


% Neural activity options
opts.removeSome = true;
opts.minFiringRate = 0.5;
opts.maxFiringRate = 40;

end