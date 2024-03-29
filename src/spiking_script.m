%% Make any options changes you want here

opts = neuro_behavior_options;
get_standard_data

%% Plot neural session data
imagesc(dataMatZ')
hold on;
line([0, size(dataMat, 1)], [idM23(end)+.5, idM23(end)+.5], 'Color', 'r');
line([0, size(dataMat, 1)], [idM56(end)+.5, idM56(end)+.5], 'Color', 'r');
line([0, size(dataMat, 1)], [idDS(end)+.5, idDS(end)+.5], 'Color', 'r');

%% Save it if you want

saveDataPath = strcat(paths.saveDataPath, animal,'/', sessionNrn, '/');
if ~exist(saveDataPath, 'dir')
    mkdir(saveDataPath)
end
saveFileName = ['neural_matrix ', 'frame_size_' num2str(opts.frameSize), [' start_', num2str(opts.collectStart), ' for_', num2str(opts.collectFor), '.mat']];
save(fullfile(saveDataPath,saveFileName), 'dataMat', 'idLabels', 'areaLabels', 'removedNeurons')

%% Or load an existing dataMat
load(fullfile(saveDataPath,saveFileName), 'dataMat', 'idLabels', 'areaLabels', 'removedNeurons')




















%%
brainArea = 'DS';
% get start times of all valid instances of this behavior
bhvCode = analyzeCodes(strcmp(analyzeBhv, 'investigate_1'));

startTimes = dataBhv.StartTime(dataBhv.ID == bhvCode);
% Use all valid behavior startTimes
startTimes(end-3:end) = [];
startTimes(1:3) = [];
startFrames = 1 + floor(startTimes ./ opts.frameSize);


dataTime = -1 : opts.frameSize : 1;
dataWindow = dataTime(1:end-1) / opts.frameSize;
alignedMat = zeros(length(dataWindow), size(dataMat, 2), length(startFrames));
for iBout = 1 : length(startTimes)
    iEpoch = startFrames(iBout) + dataWindow;
    alignedMat(:, :, iBout) = zscore(dataMat(iEpoch, :));

end
meanSpikes = mean(alignedMat, 3);

%% Sort the neurons from highest to lowest activity
sortWindow = length(dataWindow) / 2 - 2 :  length(dataWindow) / 2;
meanSpikesArea = cell(4, 1);
areas = {'M23' 'M56' 'DS' 'VS'};

for iArea = 1 : 4

    meanToSort = mean(meanSpikes(sortWindow, strcmp(areaLabels, areas(iArea))), 1);
    [~, sortInd] = sort(meanToSort, 'ascend');
    iMeanSpikes = meanSpikes(:, strcmp(areaLabels, areas(iArea)));
    meanSpikesArea{iArea} = iMeanSpikes(:, sortInd);
    % = sortMatrixByIndices(meanSpikes(strcmp(areaLabels, 'M56'),:), sortWindow(1), sortWindow(2));
end
%%
figure(10); clf; hold on;

imagesc(meanSpikesArea{2}')
colormap(bluewhitered_custom([-1 1]))
colorbar
% plot(meanSdf(:,  strcmp(areaLabels, 'M23')), 'r');
% plot(meanSpikes(:,  strcmp(areaLabels, 'M56')), 'm');
% plot(sortedMatrix(:,  strcmp(areaLabels, 'DS')), 'b');
% plot(meanSdf(:,  strcmp(areaLabels, 'VS')), 'c');




%%  Neural activity (z-scored for each neuron) for each behavior
warning('You are using mean and std within each behavioral time window to z-score - what might be better?')
spikeCounts = cell(length(bhvView), size(dataMat, 2)); % collects the spike counts surrounding each behavior start
spikeZ = cell(length(bhvView), size(dataMat, 2)); % calculates z-score of the spike counts

for iBhv = 1 : length(bhvView)
    iReg = strcmp(['0 ',bhvView{iBhv}], regressorLabels); % find regressor at start of behavior
    iStartFrames = find(bhvDesign(:,iReg)); % every frame where the behavior starts

    for jNeur = 1 : size(dataMat, 2)
        for k = 1 : length(iStartFrames)
            iRange = iStartFrames(k) - opts.mPreTime/opts.frameSize : iStartFrames(k) + opts.mPostTime/opts.frameSize;
            if iStartFrames(k) - opts.mPreTime/opts.frameSize > 0 && iStartFrames(k) + opts.mPostTime/opts.frameSize < size(dataMat, 1)
                spikeCounts{iBhv, jNeur} = [spikeCounts{iBhv, jNeur}; dataMat(iRange, jNeur)'];
                if sum(dataMat(iRange, jNeur))
                    spikeZ{iBhv, jNeur} = [spikeZ{iBhv, jNeur}; (dataMat(iRange, jNeur)' - mean(dataMat(iRange, jNeur))') / std(dataMat(iRange, jNeur))];
                else
                    spikeZ{iBhv, jNeur} = [spikeZ{iBhv, jNeur}; zeros(1,length(iRange))];
                end
            end
        end
    end
end


































%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               Mahalanobis and Euclidian distances
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

brainArea = 'M56';
dataMatMahala = zscore(dataMat(:, strcmp(areaLabels, brainArea)));
% dataMatMahala = dataMat(:, strcmp(areaLabels, brainArea));

beforeAfterFrames = 3;

dataBhvTruncate = dataBhv(3:end-3, :); % Truncate a few behaviors so we can look back and ahead in time a bit
onsetFrames = 1 + floor(dataBhvTruncate.StartTime ./ opts.frameSize);


mdist = cell(length(analyzeCodes), 1);
mdistBefore = cell(length(analyzeCodes), 1);
mdistAfter = cell(length(analyzeCodes), 1);
for iBhv = 1 : length(analyzeCodes)
    iOnsetFrames = onsetFrames(dataBhvTruncate.ID == analyzeCodes(i));

    mdist{i} = mahal(dataMatMahala(iOnsetFrames, :), dataMatMahala(iOnsetFrames, :));
    mdistBefore{i} = mahal(dataMatMahala(iOnsetFrames, :), dataMatMahala(iOnsetFrames - beforeAfterFrames, :));
    mdistAfter{i} = mahal(dataMatMahala(iOnsetFrames, :), dataMatMahala(iOnsetFrames + beforeAfterFrames, :));
end

%% Get neural data
opts.frameSize = .1;
opts.shiftAlignFactor = -.5;
opts.minFiringRate = .5;
tic
[dataMat, idLabels, areaLabels, removedNeurons] = neural_matrix(data, opts); % Change rrm_neural_matrix
toc

idM23 = find(strcmp(areaLabels, 'M23'));
idM56 = find(strcmp(areaLabels, 'M56'));
idDS = find(strcmp(areaLabels, 'DS'));
idVS = find(strcmp(areaLabels, 'VS'));

fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS))

%% Run mahalanobis distances
brainAreas = {'M23' 'M56' 'DS' 'VS'};
beforeAfterFrames = -5:5;
dataMatMahala = zscore(dataMat);
dataBhvTruncate = dataBhv(3:end-3, :); % Truncate a few behaviors so we can look back and ahead in time a bit

minTrial = sum(strcmp(areaLabels, 'DS')) * 1.3;

% fig = figure('Units', 'inches', 'Position', [0, 0, 16, 4], 'PaperOrientation', 'landscape');
fig = figure(87);
clf
set(fig, 'PaperOrientation', 'portrait','Position', [50, 50, 1000, 800])
figurePath = ['E:/Projects/neuro-behavior/docs/',animal,'/',sessionSave, '/figures/', ['start ' num2str(opts.collectStart), ' for ', num2str(opts.collectFor)]];



mdist = zeros(length(analyzeCodes), length(brainAreas), length(beforeAfterFrames)); % behaviors X brainAreas X timeFrames
% loop through behaviors to get start times of each behavior
for iBhv = 1 : length(analyzeCodes)
    jBhvInd = dataBhvTruncate.ID == analyzeCodes(iBhv);
    jOnsetFrames = 1 + floor(dataBhvTruncate.StartTime(jBhvInd) ./ opts.frameSize);


    if length(jOnsetFrames) > minTrial
        % loop through brain areas to collect mahalanobis distances in each
        % area peri-behavior start times
        for jArea = 1 : length(brainAreas)
            jNrns = strcmp(areaLabels, brainAreas{jArea});
            jMahal = dataMatMahala(jOnsetFrames, jNrns);
            jKeepNrns = std(jMahal) > 0;
            jMahal = jMahal(:, jKeepNrns);

            % brainAreas{jArea}
            % [length(jOnsetFrames) sum(jKeepNrns)]
            for kTime = 1 : length(beforeAfterFrames)
                kMahal = dataMatMahala(jOnsetFrames + beforeAfterFrames(kTime), jNrns);
                kMahal = kMahal(:, jKeepNrns);
                mdist(iBhv, jArea, kTime) = mean(mahal(kMahal,  jMahal));

                % if isnan(mdist(iBhv, jArea, kTime))
                %     [std(kMahal); std(jMahal)]
                %     cov(jMahal)
                %     disp('huh')
                % end

            end
            % reshape(mdist(iBhv, jArea, :), [], 1)

            % % plot(reshape(mdist(iBhv, jArea, :), [], 1))
            % plot(zscore(reshape(mdist(iBhv, jArea, :), [], 1)))
            % title([analyzeBhv{iBhv} ' ' brainAreas{jArea}], 'interpreter', 'none')
            % xticklabels(beforeAfterFrames)
        end
        clf
        hold on;
        plot(zscore(reshape(mdist(iBhv, 1, :), [], 1)), 'r', 'linewidth', 2)
        plot(zscore(reshape(mdist(iBhv, 2, :), [], 1)), 'm', 'linewidth', 2)
        plot(zscore(reshape(mdist(iBhv, 3, :), [], 1)), 'b', 'linewidth', 2)
        plot(zscore(reshape(mdist(iBhv, 4, :), [], 1)), 'c', 'linewidth', 2)
        title([analyzeBhv{iBhv}], 'interpreter', 'none')
        xticklabels(beforeAfterFrames .* 1000 .* opts.frameSize)
        xline(ceil(length(beforeAfterFrames)/2))
        legend(brainAreas)
        ylabel('Mahalanobis diszance (Z-score)')
        xlabel('Time from onset (ms)')
        saveas(fig,fullfile(figurePath, ['mahalanobis ', analyzeBhv{iBhv}]), 'pdf')
        % pause(3)
    end
end









%% =================================================================
%                   Run euclidian distances
%  =================================================================

% dataMatEuc = zscore(dataMat);
dataMatEuc = dataMat;
dataBhvTruncate = dataBhv(3:end-3, :); % Truncate a few behaviors so we can look back and ahead in time a bit

dataBhv.prevBhvID = [nan; dataBhv.ID(1:end-1)];
dataBhv.prevDur = [nan; dataBhv.Dur(1:end-1)];


%%
clf
fig = figure(2);
figureSize = [-1000, 0, 800, 1000]; % [left, bottom, width, height]
set(gcf, 'Position', figureSize);
% Adjust spacing between plots
spacing = -0.05; % You can adjust this value as needed
subplotSpacing = 0.01;


minBeforeDur = .2; % previous behavior must last within a range (sec)
maxBeforeDur = 1;
minNumBoutsPrev = 20;

% brainAreas = {'M23' 'M56' 'DS' 'VS'};
brainAreas = {'M56' 'DS'};
% brainAreas = {'M56'};
beforeAfterFrames = -1 / opts.frameSize : 1 / opts.frameSize;

movMeanFrames = 60;
alphaFaceValue = .3;
oneBackLabels = [];
for iCurr = 1 : length(analyzeCodes)
    for jPrev = 1 : length(analyzeCodes)
        if iCurr ~= jPrev
            % Get euclidians coming into currBhv from all other behaviors
            currIdx = dataBhvTruncate.ID == analyzeCodes(iCurr);
            currBhvOnsetAll = 1 + round(dataBhvTruncate.StartTime(currIdx) ./ opts.frameSize);

            % Get euclidians coming into currBhv from specific prevBhv
            prevIdx = currIdx & ...
                dataBhvTruncate.prevBhvID == analyzeCodes(jPrev) & ...
                dataBhvTruncate.prevDur >= minBeforeDur & ...
                dataBhvTruncate.prevDur <= maxBeforeDur;

            currBhvOnsetSub = 1 + round(dataBhvTruncate.StartTime(prevIdx) ./ opts.frameSize);


            iLabel = [analyzeBhv{jPrev}, ' then ', analyzeBhv{iCurr}];
            fprintf('%d %s\n', sum(prevIdx), iLabel)
            %
            if sum(prevIdx) >= minNumBoutsPrev
                % label it as <current behavior> after <previous behavior>
                oneBackLabels = [oneBackLabels, iLabel];






                % initialize arrays
                currDistAll = zeros(length(brainAreas), length(beforeAfterFrames));
                currDistSub = zeros(length(brainAreas), length(beforeAfterFrames));
                currDistAllSem = zeros(length(brainAreas), length(beforeAfterFrames));
                currDistSubSem = zeros(length(brainAreas), length(beforeAfterFrames));
                meanPsthAll = zeros(length(brainAreas), length(beforeAfterFrames));
                meanPsthSub = zeros(length(brainAreas), length(beforeAfterFrames));
                meanPsthAllSem = zeros(length(brainAreas), length(beforeAfterFrames));
                meanPsthSubSem = zeros(length(brainAreas), length(beforeAfterFrames));


                % Collect euclidian distances in each area peri-behavior start times
                for jArea = 1 : length(brainAreas)
                    jNrns = strcmp(areaLabels, brainAreas{jArea});
                    jNrnIdx = find(jNrns);

                    for kTime = 1 : length(beforeAfterFrames)
                        % mean psths across neurons at each time point
                        % meanPsthAll(jArea, kTime) = mean(mean(dataMatEuc(currBhvOnsetAll + beforeAfterFrames(kTime), jNrnIdx)) ./ frameSize);
                        % meanPsthSub(jArea, kTime) = mean(mean(dataMatEuc(currBhvOnsetSub + beforeAfterFrames(kTime), jNrnIdx)) ./ frameSize);
                        % meanPsthAllSem(jArea, kTime) = std(mean(dataMatEuc(currBhvOnsetAll + beforeAfterFrames(kTime), jNrnIdx)) ./ frameSize) ./ sqrt(length(currBhvOnsetAll));
                        % meanPsthSubSem(jArea, kTime) = std(mean(dataMatEuc(currBhvOnsetSub + beforeAfterFrames(kTime), jNrnIdx)) ./ frameSize) ./ sqrt(length(currBhvOnsetAll)) ;
                        meanPsthAll(jArea, kTime) = mean(mean(dataMatEuc(currBhvOnsetAll + beforeAfterFrames(kTime), jNrnIdx)));
                        meanPsthSub(jArea, kTime) = mean(mean(dataMatEuc(currBhvOnsetSub + beforeAfterFrames(kTime), jNrnIdx)));
                        meanPsthAllSem(jArea, kTime) = std(mean(dataMatEuc(currBhvOnsetAll + beforeAfterFrames(kTime), jNrnIdx))) ./ sqrt(length(currBhvOnsetAll));
                        meanPsthSubSem(jArea, kTime) = std(mean(dataMatEuc(currBhvOnsetSub + beforeAfterFrames(kTime), jNrnIdx))) ./ sqrt(length(currBhvOnsetAll)) ;

                        kMatAll = dataMatEuc(currBhvOnsetAll + beforeAfterFrames(kTime), jNrnIdx);
                        kMatSub = dataMatEuc(currBhvOnsetSub + beforeAfterFrames(kTime), jNrnIdx);

                        % Centroid at this time point
                        kCenterAll = mean(kMatAll, 1);
                        kCenterSub = mean(kMatSub, 1);

                        % Distances of each time point to the centroid
                        % distances = sqrt((vector - mean(vector)).^2);
                        kDistAll = sqrt(sum((kMatAll - kCenterAll).^2, 2));
                        kDistSub = sqrt(sum((kMatSub - kCenterSub).^2, 2));

                        % mean distances across trials
                        currDistAll(jArea, kTime) = mean(kDistAll);
                        currDistSub(jArea, kTime) = mean(kDistSub);
                        currDistAllSem(jArea, kTime) = std(kDistAll) / sqrt(length(kDistAll));
                        currDistSubSem(jArea, kTime) = std(kDistSub) / sqrt(length(kDistSub));

                    end
                end
                clf
                sgtitle([iLabel, '  n=', num2str(sum(prevIdx)), ' trials'], 'interpreter', 'none')

                fillX = [beforeAfterFrames fliplr(beforeAfterFrames)];

                % Plot Euc Distances
                subplot(3, 1, 1);
                hold on
                title('Euclidean Distance', 'interpreter', 'none')
                xline(0, '--k')
                ylabel('Euclidean distance')
                set(gca, 'Position', get(gca, 'Position') + [0, spacing, 0, -spacing - subplotSpacing]);

                plot(beforeAfterFrames, movmean(currDistAll(1,:), movMeanFrames), 'b', 'linewidth', 2)
                fillY = movmean([currDistAll(1,:) - currDistAllSem(1,:) fliplr(currDistAll(1,:) + currDistAllSem(1,:))], movMeanFrames);
                fill(fillX, fillY, 'b', 'facealpha', alphaFaceValue, 'linestyle', 'none')

                plot(beforeAfterFrames, movmean(currDistSub(1,:), movMeanFrames), 'r', 'linewidth', 2)
                fillY = movmean([currDistSub(1,:) - currDistSubSem(1,:) fliplr(currDistSub(1,:) + currDistSubSem(1,:))], movMeanFrames);
                fill(fillX, fillY, 'r', 'facealpha', alphaFaceValue, 'linestyle', 'none')

                plot(beforeAfterFrames, movmean(currDistAll(2,:), movMeanFrames), '--b', 'linewidth', 2)
                fillY = movmean([currDistAll(2,:) - currDistAllSem(2,:) fliplr(currDistAll(2,:) + currDistAllSem(2,:))], movMeanFrames);
                fill(fillX, fillY, 'b', 'facealpha', alphaFaceValue, 'linestyle', 'none')

                plot(beforeAfterFrames, movmean(currDistSub(2,:), movMeanFrames), '--r', 'linewidth', 2)
                fillY = movmean([currDistSub(2,:) - currDistSubSem(2,:) fliplr(currDistSub(2,:) + currDistSubSem(2,:))], movMeanFrames);
                fill(fillX, fillY, 'r', 'facealpha', alphaFaceValue, 'linestyle', 'none')


                % Plot PSTHs
                subplot(3, 1, 2);
                hold on
                title('PSTH', 'interpreter', 'none')
                set(gca, 'Position', get(gca, 'Position') + [0, spacing, 0, -spacing - subplotSpacing]);
                ylabel('Spikes per bin')
                xline(0, '--k')

                plot(beforeAfterFrames, movmean(meanPsthAll(1, :), movMeanFrames), 'b', 'lineWidth', 2)
                fillY = movmean([meanPsthAll(1,:) - meanPsthAllSem(1,:) fliplr(meanPsthAll(1,:) + meanPsthAllSem(1,:))], movMeanFrames);
                fill(fillX, fillY, 'b', 'facealpha', alphaFaceValue, 'linestyle', 'none')

                plot(beforeAfterFrames, movmean(meanPsthSub(1, :), movMeanFrames), 'r', 'lineWidth', 2)
                fillY = movmean([meanPsthSub(1,:) - meanPsthSubSem(1,:) fliplr(meanPsthSub(1,:) + meanPsthSubSem(1,:))], movMeanFrames);
                fill(fillX, fillY, 'r', 'facealpha', alphaFaceValue, 'linestyle', 'none')

                plot(beforeAfterFrames, movmean(meanPsthAll(2, :), movMeanFrames), '--b', 'lineWidth', 2)
                fillY = movmean([meanPsthAll(2,:) - meanPsthAllSem(2,:) fliplr(meanPsthAll(2,:) + meanPsthAllSem(2,:))], movMeanFrames);
                fill(fillX, fillY, 'b', 'facealpha', alphaFaceValue, 'linestyle', 'none')

                plot(beforeAfterFrames, movmean(meanPsthSub(2, :), movMeanFrames), '--r', 'lineWidth', 2)
                fillY = movmean([meanPsthSub(2,:) - meanPsthSubSem(2,:) fliplr(meanPsthSub(2,:) + meanPsthSubSem(2,:))], movMeanFrames);
                fill(fillX, fillY, 'r', 'facealpha', alphaFaceValue, 'linestyle', 'none')


                subplot(3, 1, 3);
                hold on
                title('Distance / PSTH', 'interpreter', 'none')
                set(gca, 'Position', get(gca, 'Position') + [0, spacing, 0, -spacing - subplotSpacing]);
                ylabel('Euclidean distance / Spikes per bin')
                xline(0, '--k')

                plot(beforeAfterFrames, movmean(currDistAll(1,:), movMeanFrames) ./ movmean(meanPsthAll(1, :), movMeanFrames), 'b', 'lineWidth', 2)
                % plot(beforeAfterFrames, movmean(meanPsthAll(1, :), movMeanFrames), 'b', 'lineWidth', 2)
                % fillX = [beforeAfterFrames fliplr(beforeAfterFrames)];
                % fillY = movmean([meanPsthAll(1,:) - meanPsthAllSem(1,:) fliplr(meanPsthAll(1,:) + meanPsthAllSem(1,:))], movMeanFrames);
                % fill(fillX, fillY, 'b', 'facealpha', .5, 'linestyle', 'none')

                plot(beforeAfterFrames, movmean(currDistSub(1,:), movMeanFrames) ./ movmean(meanPsthSub(1, :), movMeanFrames), 'r', 'lineWidth', 2)
                % plot(beforeAfterFrames, movmean(meanPsthSub(1, :), movMeanFrames), 'r', 'lineWidth', 2)

                plot(beforeAfterFrames, movmean(currDistAll(2,:), movMeanFrames) ./ movmean(meanPsthAll(2, :), movMeanFrames), '--b', 'lineWidth', 2)
                % plot(beforeAfterFrames, movmean(meanPsthAll(2, :), movMeanFrames), '--b', 'lineWidth', 2)

                plot(beforeAfterFrames, movmean(currDistSub(2,:), movMeanFrames) ./ movmean(meanPsthSub(2, :), movMeanFrames), '--r', 'lineWidth', 2)
                % plot(beforeAfterFrames, movmean(meanPsthSub(2, :), movMeanFrames), '--r', 'lineWidth', 2)
                disp('hisdf')
                % if savePlot
                %     saveas(gcf,fullfile(figurePath, [num2str(opts.frameSize * 1000), ' ms bins  Euclidean Dist and PSTHs ', iLabel]), 'pdf')
                % end
                % pause(2)
            end
        end
    end
end



%% Test euclidian distances on DataHigh example data
load('E:/Projects/toolboxes/DataHigh1.3/data/ex2_rawspiketrains.mat');

%%
frameSize = .005;
dataEx = [];

%
% reach1: use iTrial = 1 : 40
% reach2: use iTrial = 61 : 100
for iTrial = 1:40
    trialDur(iTrial) = size(D(iTrial).data, 2);
end
minTrialDur = min(trialDur)/1000/frameSize;
for iTrial = 1:40
    iData = neural_matrix_ms_to_frames(D(iTrial).data', frameSize);
    dataEx(:, :, iTrial) = iData(1:minTrialDur, :);
end
% dataEx = zscore(dataEx, 0, 1);

centerNrns = mean(dataEx, 3); %take mean (center) of each neuron at each time point across trials.

% Distances from each neuron to centroid
distNrns = sqrt(sum((dataEx - centerNrns).^2, 3));

% Mean distance to centroid per time point across trials
distAll = mean(distNrns, 2);

% mean psth across neurons
meanPsth = mean(mean(dataEx, 3), 2) ./ frameSize;
clf
% plot(currDistAll, 'k', 'linewidth', 2)
% hold on
plot(meanPsth, 'b', 'linewidth', 2)
plot(distAll ./ meanPsth, 'r', 'linewidth', 2)
plot(distAll, 'r', 'linewidth', 2)























%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             peri-onset PSTHs over the course of the 4-hour session
%
opts = neuro_behavior_options;
opts.collectStart = 0;
opts.collectFor = 4*60*60;

%             Go get  behavior data and neural matrix from above, thne run
%             below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%

area = 'M56';
area = 'VS';
bhv = 'investigate_1';
bhv = 'locomotion';
% bhv = 'face_groom_1';
bhv = 'contra_orient';
bhv = 'investigate_2';

nrnInd = strcmp(areaLabels, area);
bhvCode = analyzeCodes(strcmp(analyzeBhv, bhv));

nTrial = 40;

% X min of data every 30 min
bhvBlockWindow = 15 * 60 / opts.frameSize;
dataWindow = -2 / opts.frameSize : 2 / opts.frameSize;

bhvStartFrames = 1 + floor(dataBhv.StartTime(dataBhv.ID == bhvCode) ./ opts.frameSize);

blockFrameStarts = 1 + linspace(0, 210, 8) .* 60 ./ opts.frameSize; % get a 10 min span every 30 min

psths = zeros(sum(nrnInd), length(dataWindow), nTrial);
blockPsth = cell(length(blockFrameStarts), 1);
for iBlock = 1 : length(blockFrameStarts)
    iBlockFrameStart = blockFrameStarts(iBlock);
    iBlockFrameEnd = iBlockFrameStart + bhvBlockWindow -1;
    % iDataMat = dataMat(iFrameStart : iFrameEnd, nrnInd);

    % How many behaviors within that window?
    iBhvStarts = bhvStartFrames(bhvStartFrames >= iBlockFrameStart & bhvStartFrames < iBlockFrameEnd);
    jMat = zeros(length(dataWindow), sum(nrnInd), nTrial); % peri-event time X neurons X nTrial
    for j = 1 : nTrial
        jMat(:,:,j) = zscore(dataMat(iBhvStarts(j) + dataWindow,nrnInd));
        % jMat(:,:,j) = dataMat(iBhvStarts(j) + dataWindow,nrnInd);
        % sum(jMat(:,:,j))
        % psths(j, :) = mean(dataMat(iBhvStarts + dataWindow), nrnInd);
    end
    unrankedPsth = mean(jMat, 3)';
    unrankedPreBhv = mean(unrankedPsth(:, -dataWindow(1) : -dataWindow(1)+1), 2);
    [~, sortOrder] = sort(unrankedPreBhv, 'descend');
    rankedPsth = unrankedPsth(sortOrder,:);
    blockPsth{iBlock} = rankedPsth;

end
%%

fig = figure(3);
figureSize = [-1600, 20, 1400, 800]; % [left, bottom, width, height]
set(gcf, 'Position', figureSize);
% Adjust spacing between plots
spacing = -0.00; % You can adjust this value as needed
subplotSpacing = 0.00;

hold on
for i = 1 : length(blockPsth)
    subplot(1, length(blockPsth), i)
    imagesc(blockPsth{i})
end























%%
function sortedMatrix = sortMatrixByIndices(matrix, startIndex, endIndex)
% Check if the indices are valid
if startIndex < 1 || endIndex > size(matrix, 2) || startIndex > endIndex
    error('Invalid range of element indices.');
end

% Sort the matrix based on the specified range of column indices
[~, indexOrder] = sortrows(matrix(:, startIndex:endIndex), -1);

% Rearrange the matrix based on the sorted indices
sortedMatrix = matrix(indexOrder, :);
end

