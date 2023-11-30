


%% Linear encoding model (Musall et al 2019; Runyan et al 2017 




%% Get some data

% Classified behaviors a la Hsu & Yttri 20
% actionData = 

% spikeData = 

% lfpData = 

% Spatial behavioral data (location in box, orientation, etc)
%% Design matrix

% Model Features
%   Actions
%   Last Action
%   LFP powers as a proxy of animal state (alpha, beta, gamma)
%   Physical location in box
%   Distance from nest
%   Orientation w.r.t. nest

%   Actions: 16 acts
actList = {'xxxxxxxx', 'xxxxxxxx', 'xxxxxxxx', 'xxxxxxxx', ...
    'xxxxxxxx', 'xxxxxxxx', 'xxxxxxxx', 'xxxxxxxx', ...
    'xxxxxxxx', 'xxxxxxxx', 'xxxxxxxx', 'xxxxxxxx', ...
    'xxxxxxxx', 'xxxxxxxx', 'xxxxxxxx', 'xxxxxxxx'};
nAct = length(actList);

%   Last Action


%   LFP powers as a proxy of animal state (alpha, beta, gamma)


%   Physcial location in box

%   Distance from nest

%   Orientation w.r.t. nest

%% Options for the model
optsModel.binSize = 100; % how many ms per time step (frame)
optsModel.windowSize = 2000; % use a 2 second window (one second before and after behavior onset)
optsModel.actOnset = ceil(optsModel.windowSize / 2); % align onset of action in the middle of the analysis window
% opts.actDuration = 1000; % how many ms per act
% opts.timeStepsPerAct = actDuration / timeStepWindow; % how many time steps in an act (i.e. time steps per "trial")


actKernel = eye(ceil(optsModel.windowSize / optsModel.binSize));

% Neural history


%% Get neural data

% load spiking data

%% Options for the data

% Use 30 minutes of data, because 1) neurons move around and 2) animals change beahvioral states
dataOpts.dataDuration = 30 * 60; % seconds

%  Balance the data (take nTrial of each behavior)
dataOpts.nTrial = 200;

% Only count (use) a behavior if it...
dataOpts.minActTime = 5; % is at least minActTime frames long
dataOpts.preAct = 60; % hasn't occured within the last preAct frames

%% Build data matrix for regression


