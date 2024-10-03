%% Get data from get_standard_data

opts = neuro_behavior_options;
opts.frameSize = .1; % 100 ms framesize for now
opts.collectStart = 0*60*60; % Start collection here
opts.collectFor = 60*60; % Get 45 min


getDataType = 'all';
get_standard_data

colors = colors_for_behaviors(codes);
%%
[dataBhv, bhvIDMat] = curate_behavior_labels(dataBhv, opts);
%% Truncate the ends of the behaviors since we want a window that reaches backward and forward in time
dataBhvTrunc = dataBhv(3:end-2, :);
validBhvTrunc = validBhv(3:end-2,:);


%% What peri-onset window do you want to use?
periTime = -.25 : opts.frameSize : .25; % seconds around onset
periWindow = periTime(1:end-1) / opts.frameSize; % frames around onset w.r.t. zWindow (remove last frame)


%% If you want to subsample bouts (match the number of bouts for each behavior...
nBout = zeros(length(analyzeCodes), 1);
for i = 1 : length(analyzeCodes)
    nBout(i) = sum(dataBhvTrunc.ID == analyzeCodes(i) & validBhvTrunc(:, codes == analyzeCodes(i)));
end
nSample = min(nBout);

% %% Do you want to subsample to match the number of bouts?
% matchBouts = 0;
%
%
% %  Get relevant data
% periEventTime = -.1 : opts.frameSize : .1; % seconds around onset
% dataWindow = periEventTime(1:end-1) / opts.frameSize; % frames around onset (remove last frame)
%
%
% meanSpikes = zeros(length(analyzeBhv), size(dataMat, 2));
% meanSpikesZ = meanSpikes;
% spikesPerTrial = cell(length(analyzeBhv), 1);
% spikesPerTrialZ = cell(length(analyzeBhv), 1);
% for iBhv = 1 : length(analyzeBhv)
%     bhvCode = analyzeCodes(strcmp(analyzeBhv, analyzeBhv{iBhv}));
%
%     iStartFrames = 1 + floor(dataBhvTrunc.StartTime(dataBhvTrunc.ID == bhvCode) ./ opts.frameSize);
%     % bhvStartFrames(bhvStartFrames < 10) = [];
%     % bhvStartFrames(bhvStartFrames > size(dataMat, 1) - 10) = [];
%
%     if matchBouts
%         iRand = randperm(length(iStartFrames));
%         iStartFrames = iStartFrames(iRand(1:nSample));
%     end
%     nTrial = length(iStartFrames);
%
%     iEventMat = zeros(nTrial, size(dataMat, 2)); % nTrial X nNeurons
%     iMeanMatZ = zeros(nTrial, size(dataMat, 2)); % nTrial X nNeurons
%     for j = 1 : nTrial
%         iEventMat(j,:) = sum(dataMat(iStartFrames(j) + dataWindow ,:), 1);
%         iMeanMatZ(j,:) = mean(dataMatZ(iStartFrames(j) + dataWindow ,:), 1);
%         % dataMatZ(iStartFrames(j) + dataWindow ,:)
%     end
%
%     meanSpikes(iBhv, :) = mean(iEventMat, 1);
%     meanSpikesZ(iBhv, :) = mean(iMeanMatZ, 1);
%     spikesPerTrial{iBhv} = iEventMat;
%     spikesPerTrialZ{iBhv} = iMeanMatZ;
%
% end



%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %             Which neurons are positively and negatively (significantly) modulated for each behavior?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Create a 3-D psth data matrix of stacked peri-event start time windows (time X neuron X trial)
%
% fullTime = -1 : opts.frameSize : 1; % seconds around onset
% fullWindow = round(fullTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)
% periTime = -.1 : opts.frameSize : .1; % seconds around onset
% periWindowInd = periTime(1:end-1) / opts.frameSize; % frames around onset (remove last frame)
%
% % eventMat = cell(length(analyzeCodes), 1);
% eventMatZ = cell(length(analyzeCodes), 1);
% periMatZ = cell(length(analyzeCodes), 1);
% for iBhv = 1 : length(analyzeCodes)
%
%     iValidBhv = opts.validBhv(:, opts.bhvCodes == analyzeCodes(iBhv));
%     bhvStartFrames = 1 + floor(dataBhv.StartTime(dataBhv.ID == analyzeCodes(iBhv) & iValidBhv) ./ opts.frameSize);
%     bhvStartFrames(bhvStartFrames < fullWindow(end) + 1) = [];
%     bhvStartFrames(bhvStartFrames > size(dataMatZ, 1) - fullWindow(end)) = [];
%
%     nTrial = length(bhvStartFrames);
%
%     % iEventMat = zeros(length(fullWindow), size(dataMatZ, 2), nTrial); % peri-event time X neurons X nTrial
%     iEventMatZ = zeros(length(fullWindow), size(dataMatZ, 2), nTrial); % peri-event time X neurons X nTrial
%     iPeriMatZ = zeros(nTrial, size(dataMatZ, 2)); % peri-event time X neurons X nTrial
%     for j = 1 : nTrial
%         % iEventMat(:,:,j) = dataMat(bhvStartFrames(j) + fullWindow ,:);
%         iEventMatZ(:,:,j) = dataMatZ(bhvStartFrames(j) + fullWindow ,:);
%         iPeriMatZ(j,:) = mean(dataMatZ(bhvStartFrames(j) + periWindowInd ,:), 1);
%     end
%     % eventMat{iBhv} = iEventMat;
%     eventMatZ{iBhv} = iEventMatZ;
%     periMatZ{iBhv} = iPeriMatZ;
% end
%





%% Which neurons are and aren't tuned? Test for 2 std above/below for positive/negative tuning

nStd = 2;
posMod = zeros(length(analyzeCodes), size(dataMatZ, 2));
negMod = posMod;
notMod = posMod;

for iBhv = 1 : length(analyzeBhv)
    % mean of spikes in each trial (in pre-window and peri-window)
    periMean = mean(eventMatZ{iBhv}(fullStartInd + periWindow, :, :), 3);

    % Keep track of which neurons are positively (posMod) and negatively
    % (negMod) for each behavior)
    posMod(iBhv, :) = max(periMean) >= nStd; % standard deviations
    negMod(iBhv, :) = min(periMean) <= -nStd & ~posMod(iBhv,:); % standard deviations
    notMod(iBhv, :) = ~negMod(iBhv,:) & ~posMod(iBhv,:); % standard deviations

end



%% Which neurons are and aren't tuned? Test 1: Ttest between mean firing rates baseline vs. per-onset
%
% dataTimes = fullWindow .* opts.frameSize; % this is relative to the eventMat matrix
% preWindowInd = dataTimes >= -1 & dataTimes < -.7; % baseline window
% periWindowInd = dataTimes >= -.1 & dataTimes < .1; % peri-onset window
%
% posMod = zeros(length(analyzeCodes), size(dataMatZ, 2));
% negMod = posMod;
% notMod = posMod;
%
% for iBhv = 1 : length(analyzeBhv)
%     % mean of spikes in each trial (in pre-window and peri-window
%     preMean = mean(eventMatZ{iBhv}(preWindowInd, :, :), 1);
%     periMean = mean(eventMatZ{iBhv}(periWindowInd, :, :), 1);
%
%     preMean = permute(preMean, [3 2 1]);
%     periMean = permute(periMean, [3 2 1]);
%
%     % Are they modulated significantly?
%     [h,p,~,~] = ttest(preMean, periMean);
%     h(isnan(h)) = 0;
%     % h = reshape(h, size(h, 3), 1);
%
%     % Which one is larger (positive or negative modulation?
%     periMinusPre = mean(periMean, 1) - mean(preMean, 1);
%
%     % Keep track of which neurons are positively (posMod) and negatively
%     % (negMod) for each behavior)
%     posMod(iBhv, :) = periMinusPre > 0 & h;
%     negMod(iBhv, :) = periMinusPre < 0 & h;
%     noMod(iBhv, :) = ~posMod(iBhv, :) & ~negMod(iBhv,:);
%
% end

%% Recreate Hsu Fig 2b: What proportion of neurons are positively modulated over how many behaviors?
figure(923);
for i = 1:4
    idInd = idAll{i};
    sumPos = sum(posMod(:,idInd));

    [uniqueInts, ~, idx] = unique(sumPos);
    counts = accumarray(idx, 1);
    countsNorm = counts / sum(counts);
    % Plot the bar graph
    subplot(4,1,i)
    bar(uniqueInts, countsNorm);
    xlim([-.5 14.5])
    title(areaAll{i})
    ylabel('Positively tuned')
    if i == 4
        xlabel("Number of behaviors")
    end
end

%% How many neurons are tuned in each behavior?
figure(924);
idInd = idM56;
sumPos = sum(posMod(:,idInd), 2);
sumNeg = sum(negMod(:,idInd), 2);
sumNot = sum(notMod(:,idInd), 2);
ymax = 1 + max([sumPos; sumNeg; sumNot]);
% Plot the bar graph
subplot(1,3,1)
bar(analyzeCodes, sumNeg);
ylim([0 ymax])
title('Neg modulation')
ylabel('Number of neurons')
xticks(analyzeCodes)
xlabel('behavior')
xticklabels(analyzeBhv)
subplot(1,3,2)
bar(analyzeCodes, sumNot);
ylim([0 ymax])
xticks(analyzeCodes)
xticklabels(analyzeBhv)
title('No modulation')
subplot(1,3,3)
bar(analyzeCodes, sumPos);
xticks(analyzeCodes)
xticklabels(analyzeBhv)
ylim([0 ymax])
title('Pos modulation')

% Run if you want to see the numbers
for iBhv = 1 : length(analyzeBhv)
    fprintf('Behavior: %s\n', analyzeBhv{iBhv})
    fprintf('Positive:\n')
    fprintf('M23:\t%d\tM56\t%d\tDS:\t%d\tVS:\t%d\n', sum(posMod(iBhv, strcmp(areaLabels, 'M23'))), sum(posMod(iBhv, strcmp(areaLabels, 'M56'))), sum(posMod(iBhv, strcmp(areaLabels, 'DS'))), sum(posMod(iBhv, strcmp(areaLabels, 'VS'))));
    fprintf('Negative:\n')
    fprintf('M23:\t%d\tM56\t%d\tDS:\t%d\tVS:\t%d\n', sum(negMod(iBhv, strcmp(areaLabels, 'M23'))), sum(negMod(iBhv, strcmp(areaLabels, 'M56'))), sum(negMod(iBhv, strcmp(areaLabels, 'DS'))), sum(negMod(iBhv, strcmp(areaLabels, 'VS'))));
end








%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   Tuning across behaviors and trials
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For each behavior:
%   - Tuned population: mean > X std
%       - what proportion of the significantly tuned neurons are tuned in each/across trials?
%           - in trial with low proportions, do non- or negatively-tuned
%           neurons compensate? (i.e. are there more non-tuned neurons that
%           step up?)
%       - what proportion of each tuned neuron is tuned across trials?
%       -
%   - Non-tuned population: -X std < mean < X std
%   - Negatively tuned population: mean < X std
nStd = 2;
idInd = idM56;

%%  What proportion of positively tuned and untuned neurons are positively modulated across proportions of trials?

% Get monitor positions and size
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(230);
clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4), [.04 .02], .1);
colors = colors_for_behaviors(analyzeCodes);
edges = -.05 : .1 : 1.05;
binCenters = (edges(1:end-1) + edges(2:end)) / 2;

for iBhv = 1 : length(analyzeBhv)
    iPosTuned = logical(posMod(iBhv, idInd));
    iNegTuned = logical(negMod(iBhv, idInd));
    iNotTuned = logical(notMod(iBhv, idInd));

    % what proportion of tuned/not tuned population is tuned in each/across trials?
    iPosTunedPos = max(eventMatZ{iBhv}(fullStartInd + periWindow, idInd(iPosTuned), :), [], 1) > nStd;
    iPosTunedPos = permute(iPosTunedPos, [3 2 1]);
    iNegTunedPos = max(eventMatZ{iBhv}(fullStartInd + periWindow, idInd(iNegTuned), :), [], 1) > nStd;
    iNegTunedPos = permute(iNegTunedPos, [3 2 1]);
    iPosNotTunedPos = max(eventMatZ{iBhv}(fullStartInd + periWindow, idInd(iNotTuned), :), [], 1) > nStd;
    iPosNotTunedPos = permute(iPosNotTunedPos, [3 2 1]);

    iPropPosTuned = sum(iPosTunedPos, 2) ./ sum(iPosTuned); % Proportion of tuned neurons tuned in each trial
    iPropNegTuned = sum(iNegTunedPos, 2) ./ sum(iNegTuned); % Proportion of tuned neurons tuned in each trial
    iPropNotTuned = sum(iPosNotTunedPos, 2) ./ sum(iNotTuned); % Proportion of not tuned neurons tuned in each trial

    iPropTunedTrials = sum(iPosTunedPos, 1) ./ size(periMatZ{iBhv}, 1);

    % Create histogram with bars centered on unique values
    axes(ax(iBhv))
    hold on
    histogram(iPropPosTuned, edges, 'FaceColor', colors(iBhv,:), 'Normalization', 'probability')
    % histogram(iPropNegTuned, edges, 'FaceColor', [1 .7 .7], 'FaceAlpha', .3)
    histogram(iPropNotTuned, edges, 'FaceColor', [.7 1 .7], 'FaceAlpha', .3, 'Normalization', 'probability')
    % % Set x-axis tick marks and labels
    title([analyzeBhv{iBhv}, '  nPosTune: ', num2str(sum(iPosTuned)), '  nNotTune: ', num2str(sum(iNotTuned)), ' nTrial: ', num2str(length(iPropPosTuned))], 'interpreter', 'none')
    xticks(binCenters)
    xticklabels(binCenters)
    ca = gca;
    ca.YTickLabel = ca.YTick;
    if iBhv == 13
        % Labels for axes and title
        xlabel('Proportion of pop. tuned across trials');
        ylabel('nTrial w/ Prop Pos Modulated');
    end
end
sgtitle('Proportions of postively modulated neurons across trials')
saveas(gcf, fullfile(paths.figurePath, 'Proportions of postively modulated neurons across trials'), 'png')



%% Across trials, how many pos/not/neg/total tuned neurons are tuned for a give trial

% Get monitor positions and size
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(231);
clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4), [.04 .02], .1);

for iBhv = 1 : length(analyzeBhv)
    iPosTuned = logical(posMod(iBhv, idInd));
    iNegTuned = logical(negMod(iBhv, idInd));
    iNotTuned = logical(notMod(iBhv, idInd));

    % what proportion of tuned/not tuned population is tuned in each/across trials?
    iPosTunedPos = max(eventMatZ{iBhv}(fullStartInd + periWindow, idInd(iPosTuned), :), [], 1) > nStd;
    iPosTunedPos = permute(iPosTunedPos, [3 2 1]);
    iNegTunedPos = max(eventMatZ{iBhv}(fullStartInd + periWindow, idInd(iNegTuned), :), [], 1) > nStd;
    iNegTunedPos = permute(iNegTunedPos, [3 2 1]);
    iPosNotTunedPos = max(eventMatZ{iBhv}(fullStartInd + periWindow, idInd(iNotTuned), :), [], 1) > nStd;
    iPosNotTunedPos = permute(iPosNotTunedPos, [3 2 1]);

    % Number of pos, not, neg neurons tuned in each trial
    iNPosTuned = sum(iPosTunedPos, 2);
    iNNegTuned = sum(iNegTunedPos, 2);
    iNNotTuned = sum(iPosNotTunedPos, 2);
    iNTotTuned = sum([iNPosTuned, iNNegTuned, iNNotTuned], 2);

    % Plot lines of numbers of neurons in each category
    axes(ax(iBhv))
    hold on
    plot(1:length(iNPosTuned), iNTotTuned, 'k', 'linewidth', 2)
    plot(1:length(iNPosTuned), iNNegTuned, 'r', 'linewidth', 2)
    plot(1:length(iNPosTuned), iNNotTuned, 'color', [.5 .5 .5], 'linewidth', 2)
    plot(1:length(iNPosTuned), iNPosTuned, 'b', 'linewidth', 2)

    yline(sum(iPosTuned), 'b', 'linewidth', 2)
    yline(sum(iNegTuned), 'r', 'linewidth', 2)
    yline(sum(iNotTuned), 'color', [.5 .5 .5], 'linewidth', 2)
    xlim([0 length(iNPosTuned)])
    title([analyzeBhv{iBhv}, ' nTrial: ', num2str(length(iNPosTuned))], 'interpreter', 'none')
    ca = gca;
    ca.YTickLabel = ca.YTick;
    if iBhv == 13
        % Labels for axes and title
        xlabel('Number of pos. tuned across trials');
        ylabel('Number Postively Modulated');
        legend({'Total', 'Negative Tuned', 'Not Tuned', 'Positive Tuned'})
    end
end
sgtitle('Number of postively tuned neurons across trials')
saveas(gcf, fullfile(paths.figurePath, 'Number of postively modulated neurons across trials'), 'png')


%% Across trials, how modulated are postively tuned and untuned neurons (ones that ?

% Get monitor positions and size
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(231);
clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4), [.04 .02], .1);

for iBhv = 1 : length(analyzeBhv)
    iPosTuned = logical(posMod(iBhv, idInd));
    iNegTuned = logical(negMod(iBhv, idInd));
    iNotTuned = logical(notMod(iBhv, idInd));

    % what proportion of tuned/not tuned population is tuned in each/across trials?
    iPosTunedPos = mean(eventMatZ{iBhv}(fullStartInd + periWindow, idInd(iPosTuned), :), 1);
    iPosTunedPos = permute(iPosTunedPos, [2 3 1]);
    iNegTunedPos = mean(eventMatZ{iBhv}(fullStartInd + periWindow, idInd(iNegTuned), :), 1);
    iNegTunedPos = permute(iNegTunedPos, [2 3 1]);
    iPosNotTunedPos = mean(eventMatZ{iBhv}(fullStartInd + periWindow, idInd(iNotTuned), :), 1);
    iPosNotTunedPos = permute(iPosNotTunedPos, [2 3 1]);


    % Plot lines of numbers of neurons in each category
    axes(ax(iBhv))
    hold on
    % imagesc([iPosTunedPos; iPosNotTunedPos])
    imagesc([iPosNotTunedPos; iPosTunedPos])
    yline(size(iPosNotTunedPos, 1), 'k', 'linewidth', 4)
    xlim([0 size(iPosTunedPos,2)])
    ylim([0 size(iPosTunedPos,1)+size(iPosNotTunedPos,1)])
    colormap(bluewhitered_custom([-8 8]))
    ca = gca;
    ca.YTickLabel = ca.YTick;
    % plot(1:length(iNPosTuned), iNTotTuned, 'k', 'linewidth', 2)
    % plot(1:length(iNPosTuned), iNNegTuned, 'r', 'linewidth', 2)
    % plot(1:length(iNPosTuned), iNNotTuned, 'color', [.5 .5 .5], 'linewidth', 2)
    % plot(1:length(iNPosTuned), iNPosTuned, 'b', 'linewidth', 2)
    %
    % yline(sum(iPosTuned), 'b', 'linewidth', 2)
    % yline(sum(iNegTuned), 'r', 'linewidth', 2)
    % yline(sum(iNotTuned), 'color', [.5 .5 .5], 'linewidth', 2)
    % xlim([0 length(iNPosTuned)])
    % title([analyzeBhv{iBhv}, ' nTrial: ', num2str(length(iNPosTuned))], 'interpreter', 'none')
    % if iBhv == 13
    %     % Labels for axes and title
    %     xlabel('Number of pos. tuned across trials');
    %     ylabel('Number Postively Modulated');
    %     legend({'Total', 'Negative Tuned', 'Not Tuned', 'Positive Tuned'})
    % end
end
sgtitle('Mean peri-onset spiking M56')
% saveas(gcf, fullfile(paths.figurePath, 'Number of postively modulated neurons across trials'), 'png')




%% Across a given brain area, for positively tuned neurons, on what proportion of trials is each one positively modulated?

% Get monitor positions and size
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(237);
clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4), [.04 .02], .1);

for iBhv = 1 : length(analyzeBhv)
    iPosTuned = logical(posMod(iBhv, idInd));
    iNegTuned = logical(negMod(iBhv, idInd));
    iNotTuned = logical(notMod(iBhv, idInd));

    % what proportion of tuned/not tuned population is tuned in each/across trials?
    iPosTunedPos = max(eventMatZ{iBhv}(fullStartInd + periWindow, idInd(iPosTuned), :), [], 1) > nStd;
    iPosTunedPos = permute(iPosTunedPos, [3 2 1]);
    iNegTunedPos = max(eventMatZ{iBhv}(fullStartInd + periWindow, idInd(iNegTuned), :), [], 1) > nStd;
    iNegTunedPos = permute(iNegTunedPos, [3 2 1]);
    iPosNotTunedPos = max(eventMatZ{iBhv}(fullStartInd + periWindow, idInd(iNotTuned), :), [], 1) > nStd;
    iPosNotTunedPos = permute(iPosNotTunedPos, [3 2 1]);

    % Number of pos, not, neg neurons tuned in each trial
    iNPosTuned = sum(iPosTunedPos, 1) ./ size(iPosTunedPos, 1);
    iNNegTuned = sum(iNegTunedPos, 1) ./ size(iPosTunedPos, 1);
    iNNotTuned = sum(iPosNotTunedPos, 1) ./ size(iPosTunedPos, 1);

    % Plot lines of numbers of neurons in each category
    axes(ax(iBhv))
    hold on
    bar(1:length(iNPosTuned), iNPosTuned)
    % plot(1:length(iNPosTuned), iNNegTuned, 'r', 'linewidth', 2)
    % plot(1:length(iNPosTuned), iNNotTuned, 'color', [.5 .5 .5], 'linewidth', 2)
    % plot(1:length(iNPosTuned), iNPosTuned, 'b', 'linewidth', 2)
    %
    % yline(sum(iPosTuned), 'b', 'linewidth', 2)
    % yline(sum(iNegTuned), 'r', 'linewidth', 2)
    % yline(sum(iNotTuned), 'color', [.5 .5 .5], 'linewidth', 2)
    % xlim([0 length(iNPosTuned)])
    % title([analyzeBhv{iBhv}, ' nTrial: ', num2str(length(iNPosTuned))], 'interpreter', 'none')
    ca = gca;
    ca.YTickLabel = ca.YTick;
    % if iBhv == 13
    %     % Labels for axes and title
    xlabel('Individual pos. tuned neurons');
    ylabel('Prop. Trials Postively Modulated');
    %     legend({'Total', 'Negative Tuned', 'Not Tuned', 'Positive Tuned'})
    % end
end
sgtitle('Number of postively tuned neurons across trials')
saveas(gcf, fullfile(paths.figurePath, 'Number of postively modulated neurons across trials'), 'png')










%% Do tunings change over time (drift, jump around, etc)? Sliding window
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get data from get_standard_data

opts = neuro_behavior_options;
opts.collectStart = 0; % seconds
opts.collectFor = 60*60*4; % seconds

get_standard_data

%%



















%% Are there different modes (firing rate patterns across a given brain area) for each behavior?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% t-SNE for each behavior across all trials

% Get monitor positions and size
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(232); clf
set(fig, 'Position', secondMonitorPosition);

nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4), [.04 .02]);

for iBhv = 1 : length(analyzeBhv)

    % iSpikes = sum(eventMatZ{iBhv}(fullStartInd + periWindow, idInd, :), 1);
    iSpikes = sum(eventMat{iBhv}(fullStartInd + periWindow, idDS, :), 1);
    iSpikes = permute(iSpikes, [3 2 1]);
    [Y] = tsne(iSpikes,'Algorithm','exact');

    % Plot 2-D trials in tsne space
    axes(ax(iBhv))
    hold on
    scatter(Y(:,1), Y(:,2), 'MarkerEdgeColor', colors(iBhv,:), 'LineWidth', 2)

    title([analyzeBhv{iBhv}, ' nTrial: ', num2str(size(iSpikes, 1))], 'interpreter', 'none')
end
sgtitle('DS')
% saveas(gcf, fullfile(paths.figurePath, 'Number of postively modulated neurons across trials'), 'png')

%%
% PCA for each behavior across all trials

% Get monitor positions and size
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(232); clf
set(fig, 'Position', secondMonitorPosition);

nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4), [.04 .02]);

for iBhv = 1 : length(analyzeBhv)

    % iSpikes = sum(eventMatZ{iBhv}(fullStartInd + periWindow, idInd, :), 1);
    iSpikes = sum(eventMat{iBhv}(fullStartInd + periWindow, idM56, :), 1);
    iSpikes = permute(iSpikes, [3 2 1]);
    [coeff, score, ~, ~, explained] = pca(iSpikes);

    % Plot 2-D trials in tsne space
    axes(ax(iBhv))
    hold on
    scatter(score(:,1), score(:,2), 'MarkerEdgeColor', colors(iBhv,:), 'LineWidth', 2)

    title([analyzeBhv{iBhv}, ' nTrial: ', num2str(size(iSpikes, 1))], 'interpreter', 'none')
end
sgtitle('PCA: M56')
% saveas(gcf, fullfile(paths.figurePath, 'Number of postively modulated neurons across trials'), 'png')















%% some figure pretty things
fname = 'myfigure';

picturewidth = 20; % set this parameter and keep it forever
hw_ratio = 0.65; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',17) % adjust fontsize to your document

set(findall(hfig,'-property','Box'),'Box','off') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex')
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
%print(hfig,fname,'-dpdf','-vector','-fillpage')
print(hfig,fname,'-dpng','-vector')


















%% Mean firing rate of tuned and not-tuned neurons across trials
% Get monitor positions and size
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(231); clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4), [.04 .02]);
colors = colors_for_behaviors(analyzeCodes);

maxTrial = 50;
for iBhv = 1 : length(analyzeBhv)
    iTuned = logical(posMod(iBhv, idInd));
    meanRateTuned = mean(periMatZ{iBhv}(:, idInd(iTuned)), 2);
    meanRateNotTuned = mean(periMatZ{iBhv}(:, idInd(~iTuned)), 2);

    iTrial = min(maxTrial, length(meanRateTuned));
    iRandTrial = randperm(iTrial);
    axes(ax(iBhv))
    hold on
    plot(meanRateNotTuned(iRandTrial), 'Color', [0 .5 0], 'linewidth', 2);
    plot(meanRateTuned(iRandTrial), 'Color', colors(iBhv,:), 'linewidth', 2);
end
sgtitle('Mean pop firing rate across trials')

%%
% Create a maximized figure on the second monitor
fig = figure(232); clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(analyzeCodes);
[ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4), [.04 .02]);
colors = colors_for_behaviors(analyzeCodes);
for iBhv = 1 : length(analyzeBhv)
    iTuned = logical(posMod(iBhv, idInd));
    meanRateTuned = mean(periMatZ{iBhv}(:, idInd(iTuned)), 2);
    meanRateNotTuned = mean(periMatZ{iBhv}(:, idInd(~iTuned)), 2);

    % Calculate the correlation coefficient
    R = corrcoef(meanRateTuned, meanRateNotTuned);

    % Display the correlation coefficient
    disp(['Correlation coefficient: ', num2str(R(1,2))]);

    % Plot the data points
    axes(ax(iBhv))
    scatter(meanRateTuned, meanRateNotTuned, 'markerEdgeColor', colors(iBhv,:));
    hold on; % Hold on to the current plot

    % Fit a linear regression line
    coefficients = polyfit(meanRateTuned, meanRateNotTuned, 1);

    % Generate x values for the regression line
    xLine = linspace(min(meanRateTuned), max(meanRateTuned), 100);

    % Calculate y values for the regression line
    yLine = polyval(coefficients, xLine);

    % Plot the regression line
    plot(xLine, yLine, 'k', 'LineWidth', 2);

    % Display the correlation coefficient on the plot
    textLocation = [min(meanRateTuned), max(meanRateNotTuned)]; % Set location for the text
    text(textLocation(1), textLocation(2), ['R = ' num2str(R(1,2))], 'FontSize', 12, 'Color', 'blue');

    % Add labels and title
    xlabel('meanRateTuned');
    ylabel('meanRateNotTuned');
end
sgtitle('Tuned vs. not tuned firing rates per trial');








