%%

% In data files, overlap: 
%   - Each column is a mouse.
%   - Row 1: See constants below
%   - Row 2: Decision time
%   - Row 3: Mouse ID
%%

% Training data: - 'dSPN_training_7', 'dSPN_training_18', 'dSPN_training_14',  'dSPN_training_16', 'iSPN_training_3', 'iSPN_training_11',  'iSPN_training_2', 'iSPN_training_14', 'dopamine_training_1',  'dopamine_training_4' 
% 
% Trained data - iSPN_1, dSPN_2, dSPN_4, iSPN_5, dSPN_6, iSPN_7, dSPN_8, iSPN_9, dSPN_10, dopamine_11, dopamine_12, dopamine_100, dopamine_101
d1Subj = 2:2:8; % D1 subjects
% d1Subj = 8:2:10; % D1 subjects
% d1Subj = 2:2:8; % D1 subjects
d2Subj = [1 5 7 9]; % D2 subjects
daSubj = [11 12 100 101]; % Dopamine subjects



tIdx = 1;       % 1. current time in ms
emIdx = 2;      % 2. event marker (happy to send a breakdown of those as well if you need!)
nTrialIdx = 3;  % 3. current trial number (I believe within block)
corrIdx = 4;    % 4. correct direction. 0 means left; 1 means right
errIdx = 5;     % 5. incorrect direction reward pct
rewIdx = 6;     % 6. correct direction reward pct
minTrialIdx = 7;% 7. minimum trial # per block
maxTrialIdx = 8;% 8. maximum trial # per block
nBlockIdx = 9;  % 9. block count
rewNIdx = 10;   % 10. rewarded trial count: This number increases AFTER a turn (not in the same row)
nTurnIdx = 11;  % 11. total turn #: This number increases AFTER a turn (not in the same row)
                % 12. N/A
xIdx = 13;      % 13. X position
yIdx = 14;      % 14. Y position
                % 15. N/A
randIdx = 16;   % 16. random number coin flipped to decide whether a trial will be rewarded or not
hallIdx = 17;   % 17. hall now
prevHallIdx = 18;% 18. previous hall
                % 19-24. N/A



% event markers for when turns are detected
crl = 1; % 1 = correct, rewarded, left
cul = 2; % 2 = correct, unrewarded, left
irl = 3; % 3 = incorrect, rewarded, left
iul = 4; % 4 = incorrect, unrewarded, left
crr = 5; % 5 = correct, rewarded, right
cur = 6; % 6 = correct, unrewarded, right
irr = 7; % 7 = incorrect, rewarded, right
iur = 8; % 8 = incorrect, unrewarded, right