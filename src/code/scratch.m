


%% Linear encoding model (Musall et al 2019; Runyan et al 2017 

% Design matrix

% VARIABLES

% Actions: 16 acts
nAct = 16;
timeStepWindow = 20; % how many ms per time step
actDuration = 1000; % how many ms per act
timeStepsPerAct = actDuration / timeStepWindow; % how many time steps in an act (i.e. time steps per "trial")
% 

% Neural history


