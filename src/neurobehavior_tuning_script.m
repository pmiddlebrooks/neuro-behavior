%% Run this, then go to spiking_script and get the behavior and neural data matrix
opts.collectFor = 60*45; % Get an hour of data


%% Truncate the ends of the behaviors since we want a window that reaches backward and forward in time
dataBhvTrunc = dataBhv(3:end-2, :);
validBhvTrunc = validBhv(3:end-2,:);

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
% Create a 3-D psth data matrix of stacked peri-event start time windows (time X neuron X trial)

fullTime = -1 : opts.frameSize : 1; % seconds around onset
fullWindow = round(fullTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)
periTime = -.1 : opts.frameSize : .1; % seconds around onset
periWindow = periTime(1:end-1) / opts.frameSize; % frames around onset (remove last frame)

% eventMat = cell(length(analyzeCodes), 1);
eventMatZ = cell(length(analyzeCodes), 1);
periMatZ = cell(length(analyzeCodes), 1);
for iBhv = 1 : length(analyzeCodes)

    iValidBhv = opts.validBhv(:, opts.bhvCodes == analyzeCodes(iBhv));
    bhvStartFrames = 1 + floor(dataBhv.StartTime(dataBhv.ID == analyzeCodes(iBhv) & iValidBhv) ./ opts.frameSize);
    bhvStartFrames(bhvStartFrames < fullWindow(end) + 1) = [];
    bhvStartFrames(bhvStartFrames > size(dataMatZ, 1) - fullWindow(end)) = [];

    nTrial = length(bhvStartFrames);

    % iEventMat = zeros(length(dataWindow), size(dataMatZ, 2), nTrial); % peri-event time X neurons X nTrial
    iEventMatZ = zeros(length(fullWindow), size(dataMatZ, 2), nTrial); % peri-event time X neurons X nTrial
    iPeriMatZ = zeros(nTrial, size(dataMatZ, 2)); % peri-event time X neurons X nTrial
    for j = 1 : nTrial
        % iEventMat(:,:,j) = dataMat(bhvStartFrames(j) + dataWindow ,:);
        iEventMatZ(:,:,j) = dataMatZ(bhvStartFrames(j) + fullWindow ,:);
        iPeriMatZ(j,:) = mean(dataMatZ(bhvStartFrames(j) + periWindow ,:), 1);
    end
    % eventMat{iBhv} = iEventMat;
    eventMatZ{iBhv} = iEventMatZ;
    periMatZ{iBhv} = iPeriMatZ;
end

%% Which neurons are and aren't tuned?

dataTimes = dataWindow .* opts.frameSize; % this is relative to the eventMat matrix
preWindowInd = dataTimes >= -1 & dataTimes < -.7; % baseline window
periWindow = dataTimes >= -.1 & dataTimes < .1; % peri-onset window

posMod = zeros(length(analyzeCodes), size(dataMatZ, 2));
negMod = posMod;
noMod = posMod;

for iBhv = 1 : length(analyzeBhv)
    % mean of spikes in each trial (in pre-window and peri-window
    preRate = mean(eventMatZ{iBhv}(preWindowInd, :, :), 1);
    periRate = mean(eventMatZ{iBhv}(periWindow, :, :), 1);

    preRate = permute(preRate, [3 2 1]);
    periRate = permute(periRate, [3 2 1]);

    % Are they modulated significantly?
    [h,p,~,~] = ttest(preRate, periRate);
    h(isnan(h)) = 0;
    % h = reshape(h, size(h, 3), 1);

    % Which one is larger (positive or negative modulation?
    periMinusPre = mean(periRate, 1) - mean(preRate, 1);

    % Keep track of which neurons are positively (posMod) and negatively
    % (negMod) for each behavior)
    posMod(iBhv, :) = periMinusPre > 0 & h;
    negMod(iBhv, :) = periMinusPre < 0 & h;
    noMod(iBhv, :) = ~posMod(iBhv, :) & ~negMod(iBhv,:);

end

%% Run if you want to see the numbers
for iBhv = 1 : length(analyzeBhv)
    fprintf('Behavior: %s\n', analyzeBhv{iBhv})
    fprintf('Positive:\n')
    fprintf('M23:\t%d\tM56\t%d\tDS:\t%d\tVS:\t%d\n', sum(posMod(iBhv, strcmp(areaLabels, 'M23'))), sum(posMod(iBhv, strcmp(areaLabels, 'M56'))), sum(posMod(iBhv, strcmp(areaLabels, 'DS'))), sum(posMod(iBhv, strcmp(areaLabels, 'VS'))));
    fprintf('Negative:\n')
    fprintf('M23:\t%d\tM56\t%d\tDS:\t%d\tVS:\t%d\n', sum(negMod(iBhv, strcmp(areaLabels, 'M23'))), sum(negMod(iBhv, strcmp(areaLabels, 'M56'))), sum(negMod(iBhv, strcmp(areaLabels, 'DS'))), sum(negMod(iBhv, strcmp(areaLabels, 'VS'))));
end



%%
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
nStd = 1;

idInd = idM56;

    % Get monitor positions and size
    monitorPositions = get(0, 'MonitorPositions');
    if size(monitorPositions, 1) < 2
        error('Second monitor not detected');
    end
    secondMonitorPosition = monitorPositions(2, :);
    % Create a maximized figure on the second monitor
    fig = figure(230);
    clf
    set(fig, 'Position', secondMonitorPosition);
    nPlot = length(analyzeCodes);
    [ax, pos] = tight_subplot(ceil(nPlot/4), ceil(nPlot/4));
    colors = colors_for_behaviors(analyzeCodes);

for iBhv = 1 : length(analyzeBhv)
    iTuned = logical(posMod(iBhv, idInd));

    % periMatZ{iBhv}(:, idInd(iTuned))
    % what proportion of tuned population is tuned in each/across trials?
    iPosTune = periMatZ{iBhv}(:, idInd(iTuned)) > 1;
    iPropTuned = sum(iPosTune, 2) ./ sum(iTuned);

    values = unique(iPropTuned);
    % Create histogram
    % [counts, uniqueValues] = histcounts(iPropTuned, unique(values));
    [counts, uniqueValues] = histcounts(iPropTuned, length(values));

% Create histogram with bars centered on unique values
axes(ax(iBhv))
bar(values, counts, 'BarWidth', 1);

% Set x-axis tick marks and labels
xticks(values);
xticklabels(arrayfun(@num2str, values, 'UniformOutput', false));
title([analyzeBhv{iBhv}, '  nTuned: ', num2str(sum(iTuned))], 'interpreter', 'none')

if iBhv == 13
% Labels for axes and title
xlabel('Proportions');
ylabel('Count');
end
    sgtitle('Proportions of postively tuned neurons across trials')
end








%% Plot some psths
figure(88)
idInd = idM56;
bhvName = 'contra_orient';
% bhvName = 'face_groom_1';
% bhvName = 'locomotion';
bhv = analyzeCodes(strcmp(analyzeBhv, bhvName));
psthMean = mean(eventMatZ{bhv}(:, idInd, :), 3)';

sortWindow = (dataWindow * opts.frameSize) >= -.2 & (dataWindow * opts.frameSize) < .2;
    unrankedPreBhv = mean(psthMean(:, sortWindow), 2);
    [~, sortOrder] = sort(unrankedPreBhv, 'descend');
    rankedPsth = psthMean(sortOrder,:);

imagesc(rankedPsth);
xticks(1:length(dataWindow))
xticklabels(dataWindow)
xline(10.5, 'linewidth', 2)
colormap(bluewhitered_custom), colorbar
title(bhvName, 'Interpreter', 'none')






