function bhvID = get_reach_bhv_labels(fileName, frameSize)

% All times are in ms


% I have data from a joystick reach task. The mouse, self-paced, learns to
% move the joystick into a target area. The mouse has to learn the target
% area without perceiving it, through trial and error. Each self-initiated trial can be
% correct, resulting in reward, or an error, resulting in an error light.
% Errors can be above or below the target area. A session is divided into
% blocks, in which the target area changes without notice.
% I want to generate a vector of categorical behavior labels for all time
% points throughout the session. 
%%
paths = get_paths;

traces = load(fullfile(paths.dropPath, 'reach_data/Y4_06-Oct-2023 14_14_53_NIBEH.mat'));
dataR = load(fullfile(paths.dropPath, 'reach_data/Copy_of_Y4_100623_Spiketimes_idchan_BEH.mat'));

%%

reachStart = dataR.R(:,1);
reachStop = dataR.R(:,2);
reachAmp = dataR.R(:,3); % Amplitude of each reach (distance from 0)

 % block or BlockWithErrors - columns are BlockNum Correct? ReachClassification1-4 ReachClassification-20-20
 %     -rows are reaches
 %     classification1-4:
 %     - 1 - block 1 error
 %     - 2 - block 1 Rew
 %     - 3 - block 2 error
 %     - 4 - block 2 rew
 %     - 5 - block 3 error
 %     - 6 - block 3 rew
 % 
 % 
 %     classification-20-20:
 %     - -10 - block 1 error below
 %     -  1   - block 1 Rew
 %     - 10 - block 1 error above
 %     - -20 - block 2 error below
 %     -  2   - block 2 Rew
 %     - 20 - block 2 error above

reachClass = dataR.block(:,4);

% Continuous traces normalized to max
jsTrace = zscore(traces.JSNIpos); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
% jsTrace = traces.JSNIpos ./ max(traces.JSNIpos); % Continuous absolute value of amplitude trace of joystick jsTrace = sqrt(x^2 + y^2)
errTrace = traces.ERtrace ./ max(traces.ERtrace); % Continuous voltage trace of error light (for error trials)
solTrace = traces.SOLtrace ./ max(traces.SOLtrace); % Continous voltage trace of solenoid (for correct rewarded trials)

% Event times
errTimes = traces.ERtimes; % Error light turns on (errors)
solTimes = traces.SOLtimes; % Reward solenoid opens (correct)

%%
figure(88); clf; hold on;

plot(jsTrace, 'k');
plot(errTrace, 'red');
plot(solTrace, 'color', [0 .6 0]);
scatter(reachStart, -.025, 'k')
scatter(errTimes, -.05, 'MarkerFaceColor', 'r');
scatter(solTimes, -.075*ones(length(solTimes)), 'MarkerFaceColor',[0 .6 0]);
