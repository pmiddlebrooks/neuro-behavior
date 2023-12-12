%%%%%%%%%%%%%%%%%%%%
%   TESTING TO REPLICATE PSTHS
%%%%%%%%%%%%%%%%%%%%


%%
%% get desired file paths
computerDriveName = 'ROSETTA'; % 'Z' or 'home'
paths = rrm_get_paths(computerDriveName);


opts = rrm_options;
animal = 'ag25290';
sessionBhv = '112321_1';
sessionNrn = '112321';

if strcmp(sessionBhv, '112321_1')
    sessionSave = '112321';
end


%%
figurePath = strcat(paths.figurePath, animal, '/', sessionSave, '/figures/', ['start ' num2str(opts.collectStart), ' for ', num2str(opts.collectFor)]);
if ~exist(figurePath, 'dir')
    mkdir(figurePath);
end
%%
bhvDataPath = strcat(paths.bhvDataPath, 'animal_',animal,'/');
bhvFileName = ['behavior_labels_', animal, '_', sessionBhv, '.csv'];


opts.dataPath = bhvDataPath;
opts.fileName = bhvFileName;

dataBhv = load_data(opts, 'behavior');



codes = unique(dataBhv.bhvID);
% codes(codes == -1) = []; % Get rid of the nest/irrelevant behaviors
behaviors = {};
for iBhv = 1 : length(codes)
    firstIdx = find(dataBhv.bhvID == codes(iBhv), 1);
    behaviors = [behaviors, dataBhv.bhvName{firstIdx}];
    % fprintf('behavior %d:\t code:%d\t name: %s\n', i, codes(i), dataBhvAlex.Behavior{firstIdx})
end

opts.behaviors = behaviors;
opts.bhvCodes = codes;
opts.validCodes = codes(codes ~= -1);


%% Select valid behaviors
validBhv = behavior_selection(dataBhv, opts);
opts.validBhv = validBhv;
allValid = logical(sum(cell2mat(validBhv),2)); % A list of all the valid behvaior indices


%% Select which behaviors we want to view/analyze
rmvBhv = zeros(1, length(validBhv));
for i = 1 : length(validBhv)
    if sum(validBhv{i}) < 20
    rmvBhv(i) = 1;
    end
end
analyzeBhv = behaviors(~rmvBhv);
analyzeCodes = codes(~rmvBhv);









%% Get Neural matrix
nrnDataPath = strcat(paths.nrnDataPath, 'animal_',animal,'/', sessionNrn, '/');
nrnDataPath = [nrnDataPath, 'recording1/'];
opts.dataPath = nrnDataPath;

data = load_data(opts, 'neuron');
data.bhvDur = dataBhv.bhvDur;
clusterInfo = data.ci;
spikeTimes = data.spikeTimes;
spikeClusters = data.spikeClusters;


%% Find the neuron clusters (ids) in each brain region


allGood = strcmp(data.ci.group, 'good') & strcmp(data.ci.KSLabel, 'good');

goodM23 = allGood & strcmp(data.ci.area, 'M23');
goodM56= allGood & strcmp(data.ci.area, 'M56');
goodDS = allGood & strcmp(data.ci.area, 'DS');
goodVS = allGood & strcmp(data.ci.area, 'VS');




%% Make or load neural matrix

% which neurons to use in the neural matrix
opts.useNeurons = find(goodM23 | goodM56 | goodDS | goodVS);

tic
[dataMat, idLabels, areaLabels, removedNeurons] = neural_matrix(data, opts); % Change rrm_neural_matrix
toc

idVS = find(strcmp(areaLabels, 'VS'));
idDS = find(strcmp(areaLabels, 'DS'));
idM56 = find(strcmp(areaLabels, 'M56'));
idM23 = find(strcmp(areaLabels, 'M23'));

fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS))

%%

saveDataPath = strcat(paths.saveDataPath, animal,'/', sessionNrn, '/');
saveFileName = ['neural_matrix ', 'frame_size_' num2str(opts.frameSize), [' start_', num2str(opts.collectStart), ' for_', num2str(opts.collectFor), '.mat']];
save(fullfile(saveDataPath,saveFileName), 'dataMat', 'idLabels', 'areaLabels', 'removedNeurons')
%%
load(fullfile(saveDataPath,saveFileName), 'dataMat', 'idLabels', 'areaLabels', 'removedNeurons')



%% Normalize and zero-center the neural data matrix 

% Normalize (z-score)
dataMatZ = zscore(dataMat, 0, 1);

% imagesc(dataMat')
imagesc(dataMatZ')
hold on;
line([0, size(dataMat, 1)], [find(strcmp(areaLabels, 'VS'), 1, 'last'),find(strcmp(areaLabels, 'VS'), 1, 'last')], 'Color', 'r');
line([0, size(dataMat, 1)], [find(strcmp(areaLabels, 'DS'), 1, 'last'),find(strcmp(areaLabels, 'DS'), 1, 'last')], 'Color', 'r');
line([0, size(dataMat, 1)], [find(strcmp(areaLabels, 'M56'), 1, 'last'),find(strcmp(areaLabels, 'M56'), 1, 'last')], 'Color', 'r');










%%
brainArea = 'DS';
% get start times of all valid instances of this behavior
bhvCode = analyzeCodes(strcmp(analyzeBhv, 'investigate_1'));

startTimes = dataBhv.bhvStartTime(dataBhv.bhvID == bhvCode);
% Use all valid behavior startTimes
% startTimes = dataBhv.bhvStartTime(allValid);
startTimes(end-3:end) = [];
startTimes(1:3) = [];
startFrames = floor(startTimes ./ opts.frameSize);

alignedMat = cell(length(startFrames), 1);
 
preStartFrames = 1 / opts.frameSize;
postStartFrames = 1 / opts.frameSize;
for iBout = 1 : length(startTimes)
    iEpoch = startFrames(iBout) + 1 - preStartFrames: startFrames(iBout) + postStartFrames;
    % iEpoc = startFrames(iTrial) + 1 - opts.mPreTime / opts.frameSize : startFrames(iTrial);
    % iEpoch = startFrames(iTrial) : startFrames(iTrial) + opts.mPostTime / opts.frameSize;
    % alignedMat{iBout} = dataMatZ(iEpoch, :);
    alignedMat{iBout} = zscore(dataMat(iEpoch, :));

end
meanSpikes = mean(cat(4, alignedMat{:}), 4)';

%% Sort the neurons from highest to lowest activity
sortWindow = length(iEpoch) / 2 - 2 :  length(iEpoch) / 2 + 2;
meanSpikesArea = cell(4, 1);
areas = {'M23' 'M56' 'DS' 'VS'};

for iArea = 1 : 4

meanToSort = mean(meanSpikes(strcmp(areaLabels, areas(iArea)), sortWindow), 2);
[~, sortInd] = sort(meanToSort, 'ascend');
iMeanSpikes = meanSpikes(strcmp(areaLabels, areas(iArea)), :);
meanSpikesArea{iArea} = iMeanSpikes(sortInd, :);
% = sortMatrixByIndices(meanSpikes(strcmp(areaLabels, 'M56'),:), sortWindow(1), sortWindow(2));
end
%%
figure(10); clf; hold on;

imagesc(meanSpikesArea{2})
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
            iRange = iStartFrames(k) - opts.mPreTime : iStartFrames(k) + opts.mPostTime;
            if iStartFrames(k) - opts.mPreTime > 0 && iStartFrames(k) + opts.mPostTime < size(dataMat, 1)
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


%% Plot some betas and psths

% set up figure
nRow = 4;
nColumn = 2;
orientation = 'portrait';
figureHandle = 34;
[axisWidth, axisHeight, xAxesPosition, yAxesPosition] = standard_figure(nRow, nColumn, orientation, figureHandle);
for col = 1 : nColumn
    for row = 1 : nRow
        ax(row,col) = axes('units', 'centimeters', 'position', [xAxesPosition(row, col) yAxesPosition(row, col) axisWidth axisHeight]);
        hold on;
    end
end
% colormap(bluewhitered)
colormap(jet)


% Indices of the neurons in each area
neuronsIdx = [goodM23, goodM56, goodDS, goodVS];
plotTitles = {'M23', 'M56', 'DS', 'VS'};


% Loop through all the behaviors you want to see
for iBhv = 1 : length(bhvView)

    % Initialize a logical array to find relevant regressors
    containsReg = false(size(regressorLabels));

    % Loop through the regressors to check for the set corresponding to
    % that behavior
    for rIdx = 1:numel(regressorLabels)
        if ischar(regressorLabels{rIdx}) && contains(regressorLabels{rIdx}, bhvView{iBhv})
            containsReg(rIdx) = true;
        end
    end

    % Specify the custom title position (in normalized figure coordinates)
    titleText = bhvView{iBhv};
    annotation('textbox', [.5, 1, 0, 0], 'String', titleText, 'Interpreter', 'none', 'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    annotation('textbox', [.2, 1, 0, 0], 'String', 'Betas', 'Interpreter', 'none', 'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    annotation('textbox', [.8, 1, 0, 0], 'String', 'Spikes', 'Interpreter', 'none', 'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    nrnIdx = 0;
    for row = 1 : nRow
        % Example plotting in the first subplot (ax1)
        neuronsPlot = intersect(idLabels, data.ci.cluster_id(neuronsIdx(:,row))); % Plot the neurons within this brain area in this row
        nrnIdx = 1 + nrnIdx(end) : nrnIdx(end) + length(neuronsPlot);

        % Beta value plot
        betaPlot = dimBeta(containsReg, nrnIdx);
        axes(ax(row, 1));
        if row < nRow
            set(ax(row, 1), 'xticklabel',{[]})
        end
        imagesc('CData', betaPlot');
        plot(ax(row, 1), [10.5 10.5], [0 length(neuronsPlot)], 'k', 'linewidth', 3)
        title(plotTitles{row});
        xlim([0.5 20.5])
        ylim([0.5 length(neuronsPlot)+.5])


        % PSTH plot
        axes(ax(row, 2))
        if row < nRow
            set(ax(row, 2), 'xticklabel',{[]})
        end
        imagesc(cell2mat(cellfun(@mean, (spikeZ(iBhv,nrnIdx)), 'UniformOutput', false)'))
        plot(ax(row, 2), [10.5 10.5], [0 length(neuronsPlot)], 'k', 'linewidth', 3)
        title(plotTitles{row});
        xlim([0.5 20.5])
        ylim([0.5 length(neuronsPlot)+.5])

    end



    c = colorbar;
    c.Position = [0.93, 0.1, 0.02, 0.5];

    if savePlot
        saveas(figureHandle,fullfile(figurePath, ['betas_and_spikes ', bhvView{iBhv}]), 'pdf')
    end
    pause(10)
    delete(findall(gcf,'type','annotation'))
    % clf
end

close(figureHandle)






%%
iBhv = 14;
% imagesc(cell2mat(cellfun(@sum, (spikeCounts(1,:)), 'UniformOutput', false)'))
imagesc(cell2mat(cellfun(@mean, (spikeZ(iBhv,:)), 'UniformOutput', false)'))
colormap(bluewhitered), colorbar











%% Examine where corpus colosum might be:
opts.removeSome = false;
opts.frameSize = .4; % use a big frame to go quickly
opts.useNeurons = 1 : size(data.ci, 1);

    
% Make or load neural matrix
[dataMat, idLabels, rmvNeurons] = neural_matrix(data, opts);


%%
depths = unique(data.ci.depth);
nNeurons = zeros(length(depths), 1);
for i = 1:length(depths)

    nNeurons(i) = sum(data.ci.depth == depths(i) & strcmp(data.ci.group, 'good') & strcmp(data.ci.group, 'good'));
end

%% get avg firing rate for the window 

meanRates = sum(dataMat, 1) ./ (size(dataMat, 1) * opts.frameSize);

plot(data.ci.depth, meanRates)












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



