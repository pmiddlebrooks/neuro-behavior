%% Get data from get_standard_data

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0; % seconds
opts.collectEnd = 60*60*3.5; % seconds

get_standard_data

% From Matt Smith:
% Slow drift is PCA on the matrix of neurons by time bins, where each time
% bin is the average response over a pretty big window of time. Let's say
% at least 30 seconds, but could be several minutes. In our paper we didn't
% just average the spikes in that window, but rather took the evoked
% responses (let's say there were 20 trials in a minute, we took 20 400 ms
% epochs of data after stimulus onset and averaged those 20 values). Then
% it's just the first PC.
%
% Just to say again what this is, start with the trial-locked epochs of
% activity. So, in a day if there were 1000 trials each 400 ms long, we
% grab the spike count in that 400 ms window, and have a matrix that is
% neurons by 1000. Then, each trial has an exact time, an what we did is
% take time bins thorough the day (0-10 minutes, 5-15 minutes, 10-20
% minutes, etc) and for each bin grab all the trials that were in that time
% bin and average those spike counts. Now the matrix is neurons by time
% bins. We did PCA on that.

%% Define window constants
bigWinTime = 10 * 60; % window within which to avg all responses: seconds
bigWinFrame = bigWinTime / opts.frameSize;
winSlideTime = 5 * 60; % slide bigWindow this many seconds over the duration
winSlideFrame = winSlideTime / opts.frameSize;
nWind = (size(dataMat, 1) - bigWinFrame)/ winSlideFrame;
%%
% Which brain area to analyze?
idInd = idM56;
brainArea = 'M56';
% idInd = cell2mat(idAll);

% which behavior to analyze?
% behavior = 'contra_body_groom';
behavior = 'locomotion';
bhvID = analyzeCodes(strcmp(analyzeBhv, behavior));
bhvInd = find(strcmp(analyzeBhv, behavior));

periTime = -.2 : opts.frameSize : .2;
periWindow = periTime(1:end-1) / opts.frameSize; % frames around onset w.r.t. zWindow (remove last frame)

% startFrame = 1 + floor(dataBhv.StartTime(dataBhv.ID == bhvID & validBhv(:, codes == bhvID)) / opts.frameSize);
startFrame = 1 + floor(dataBhv.StartTime(dataBhv.ID == bhvID) / opts.frameSize);

% Get mean of spike counts in peri-onset windows, across session, to
% calculate residual spikes
periSpike = [];
for iStart = 1 : length(startFrame)
    periSpike = [periSpike; sum(dataMat(startFrame(iStart) + periWindow, :), 1)];
end
meanSpike = mean(periSpike, 1);
%
meanRes = zeros(nWind, size(dataMat, 2));
for iWin = 1 : nWind
    % Get the spike counts for each bout in this window
    iWinFrame = (iWin-1) * winSlideFrame + 1 : iWin * winSlideFrame + bigWinFrame;
    iStartFrame = startFrame(startFrame >= iWinFrame(1) & startFrame <= iWinFrame(end));
    iPeriRes = [];
    for j = 1 : length(iStartFrame)
        % Calculate residuals of peri-onset spike counts
        iPeriRes = [iPeriRes; sum(dataMat(iStartFrame(j) + periWindow, :), 1) - meanSpike];       
    end
    % Mean of the residuals for this window
    meanRes(iWin,:) = mean(iPeriRes, 1);
end
%% PCA on the residual running avgs
% [coeff, score, ~, ~, explained] = pca(meanRes(:,idInd)');
[coeff, score, ~, ~, explained] = pca(meanRes(:,idInd));
figure(45); clf
for i = 1 : 3
slowDriftAxis = coeff(:,i);
%
slowDriftProj = periSpike(:, idInd) * slowDriftAxis;
subplot(3,1,i)
hold on
scatter(1:length(slowDriftProj), slowDriftProj, '.')
    plot(1:length(slowDriftProj), movmean(slowDriftProj, 20), 'lineWidth', 3)

xlabel('Bouts')
ylabel(['Slow drift PC ', num2str(i)])
end
sgtitle([brainArea, ' Slow drift for ', behavior], 'interpreter', 'none')
fprintf('First 3 components explained variance:\n %.2f\n %.2f\n %.2f\n', explained(1),  explained(2),  explained(3));




%%
idInd = cell2mat(idAll);
nBout = size(eventMat{bhvInd}, 3);
periSpikeCt = sum(eventMat{bhvInd}(fullStartInd + periWindow, :, :), 1);
periSpikeCt = permute(periSpikeCt, [3 2 1]);
% make spike rates instead of counts?
meanSpikeCt = mean(periSpikeCt, 1);

resSpikeCt = periSpikeCt - meanSpikeCt;


%% Individual neuron slow drift for given behavior
% Which brain area to analyze?
idInd = idM56;
idInd = cell2mat(idAll);

figure(654);
for i = 1:length(idInd)
    clf; hold on
    scatter(1:nBout, resSpikeCt(:,idInd(i)), 'filled')
    plot(1:nBout, movmean(resSpikeCt(:,idInd(i)), 20), 'linewidth', 3)
end
%% PCA projections of slow drift for given behavior/brain area

% Which brain area to analyze?
idInd = idM56;
idInd = cell2mat(idAll);
% Perform PCA
[coeff, score, ~, ~, explained] = pca(resSpikeCt(:,idInd));
figure(62); clf;
for iComp = 1:3
    subplot(3, 1, iComp); hold on;
    scatter(1:size(score, 1), score(:,iComp))
    plot(1:size(score, 1), movmean(score(:,iComp), 20), 'lineWidth', 3)
    ylabel(['Component ', num2str(iComp)])
    xlabel('Bouts')
end
sgtitle(['PCA projections: ', behavior], 'interpreter', 'none')


