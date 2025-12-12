%% go to desired folder
cd([paths.homePath, 'toolboxes/gpfa_v0203/'])
startup
% cd('E:/Projects/toolboxes/DataHigh1.3/')
% cd([paths.homePath, 'toolboxes/DataHigh1.3/'])

% Get data from get_standard_data

opts = neuro_behavior_options;
opts.collectStart = 0; % seconds
opts.collectFor = 30 * 60; % seconds
opts.frameSize = .001;

getDataType = 'spikes';
get_standard_data
[dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);


%% DataHigh: Test Dim Reduction with 8 behaviors
nBhv = 8;
nBout = 40;
idInd = idM56;

periTime = -.2 : opts.frameSize : .2;
periWindow = periTime(1:end-1) / opts.frameSize; % frames around onset w.r.t. zWindow (remove last frame)
colors = colors_for_behaviors(analyzeCodes);

D = struct();
bhvList = {};
bout = 1;
for iBhv = 1 : nBhv
    if size(eventMat{iBhv}, 3) >= nBout
        iBout = randperm(nBout);

        for jBout = 1 : nBout
            iData = eventMat{iBhv}(fullStartInd + periWindow, idInd, jBout)';
            % iData = permute(iData, [2 1 3]);
            % iData = reshape(iData, size(iData, 1), size(iData, 2) * size(iData, 3));
            D(bout).data = iData;
            D(bout).condition = analyzeBhv{iBhv};
            D(bout).epochStarts = 1;
            D(bout).epochColors = colors(iBhv,:);
            bout = bout + 1;
        end
    end
end
%%
DataHigh(D, 'DimReduce');






%% DataHigh: For test/practice, look at a handful of trajectories leading up to a given behavior
clear paulD

idInd = idM56;
nTraj = 50;

% get start times of all valid instances of this behavior
bhvCode = analyzeCodes(strcmp(analyzeBhv, 'locomotion'));

startTimes = dataBhv.StartTime(dataBhv.ID == bhvCode & dataBhv.Valid);
% Use all valid behavior startTimes
startTimes(end-8:end) = [];
startTimes(1:8) = [];
startFrames = 1 + floor(startTimes ./ opts.frameSize);
for iTrial = 1 : length(startTimes)
    iEpoch = startFrames(iTrial) + 1 - .6 / opts.frameSize : startFrames(iTrial);

    paulD(iTrial).condition = brainArea;
    paulD(iTrial).data = dataMat(iEpoch, idInd)';
    paulD(iTrial).epochStarts = 1;
    paulD(iTrial).epochColors = [1 0 0]; %

end

%%
dView = paulD(randperm(length(startFrames), nTraj));
DataHigh(dView,'DimReduce')










%% For test/practice, let's look at trajectories of a particular sequence of behaviors from start of one to start of another

currBhv = 'locomotion';
nextBhv = 'investigate_2';

currCode = analyzeCodes(strcmp(analyzeBhv, currBhv));
nextCode = analyzeCodes(strcmp(analyzeBhv, nextBhv));

% Find all indices of currBhv that are followed by nextBhv
% Use this version to ensure only the next behavior is valid, ignoring whether the current behavior is valid
currIdx = (dataBhv.ID == currCode);%
% Use this version to ensure both behaviors in the sequence are
% valid
% currIdx = (dataBhv.ID == currCode) & dataBhv.Valid;
nextIdx = (dataBhv.ID == nextCode) & dataBhv.Valid;

compareIdx = [nextIdx(2:end); 0];

validCurr = currIdx & compareIdx;
validNext = [false; validCurr(1:end-1)];


startEpochs = dataBhv.StartTime(validCurr);
endEpochs = dataBhv.StartTime(validNext);
startFrames = 1 + floor(startEpochs / opts.frameSize);
endFrames = 1 + floor(endEpochs / opts.frameSize);
%% check data for sanity:
dataBhv(validCurr | validNext, :)
plot(endFrames - startFrames)

%%
clear paulD
brainArea = 'M56';
for iTrial = 1 : length(startEpochs)
    iEpoch = startFrames(iTrial) : endFrames(iTrial);

    paulD(iTrial).condition = brainArea;
    paulD(iTrial).data = dataMat(iEpoch, strcmp(areaLabels, brainArea))';
    paulD(iTrial).epochStarts = 1;
    paulD(iTrial).epochColors = [1 0 0]; %

end

%%

DataHigh(paulD,'DimReduce')





%%     Single long span of behavior/neural using 2 sec consecutive windows
clear dat
brainArea = 'M56';
binWidth = 50;
fullDurFrames = 6 / opts.frameSize;
secDurFrames = 2 / opts.frameSize;
nSections = floor(fullDurFrames / secDurFrames);
startInd =  1 : secDurFrames : fullDurFrames;
for iTrial = 1 : length(startInd)
    iEpoch = startInd(iTrial) : startInd(iTrial) + secDurFrames;% - 1;
    dat(iTrial).spikes = dataMat(iEpoch, strcmp(areaLabels, brainArea))';
    dat(iTrial).trialId = iTrial;
    marks{iTrial} = [1 floor(length(iEpoch) / binWidth)];
    % paulD(iTrial).epochStarts = 1;
    % paulD(iTrial).epochColors = [1 0 0]; % m23 is red
    %
end

%%
tic
method = 'gpfa';
runIdx = 2;
% Select number of latent dimensions
xDim = 8;
kernSD = 30;
% Extract neural trajectories
result = neuralTraj(runIdx, dat, 'method', method, 'xDim', xDim,...
    'kernSDList', kernSD, 'seglength', 50, 'binWidth', binWidth);
% NOTE: This function does most of the heavy lifting.

% Orthonormalize neural trajectories
[estParams, seqTrain] = postprocess(result, 'kernSD', kernSD);
% NOTE: The importance of orthnormalization is described on
%       pp.621-622 of Yu et al., J Neurophysiol, 2009.

% Plot neural trajectories in 3D space
plot3D_marks(seqTrain, 'xorth', 'dimsToPlot', 1:3, 'marks', marks);

% Plot each dimension of neural trajectories versus time
plotEachDimVsTime(seqTrain, 'xorth', result.binWidth);

toc








%%

displayTime = [10 15]; % how long to display (s)
displayFrames = 1 + displayTime(1) / opts.frameSize : displayTime(2) / opts.frameSize; % convert time to frames
m23Data = dataMat(displayFrames, strcmp(areaLabels, 'M23'))';
m56Data = dataMat(displayFrames, strcmp(areaLabels, 'M56'))';
dsData = dataMat(displayFrames, strcmp(areaLabels, 'DS'))';
vsData = dataMat(displayFrames, strcmp(areaLabels, 'VS'))';

%% HighData
paulD(1).data = m56Data;
paulD(1).condition = 'm23';
paulD(1).epochStarts = 1;
paulD(1).epochColors = [1 0 0]; % m23 is red

% paulD(2).data = m56Data;
% paulD(2).condition = 'm56';
% paulD(2).epochStarts = 1;
% paulD(2).epochColors = [1 0 1]; % m23 is magenta
%
% paulD(3).data = dsData;
% paulD(3).condition = 'ds';
% paulD(3).epochStarts = 1;
% paulD(3).epochColors = [0 0 1]; % m23 is red
%
% paulD(4).data = vsData;
% paulD(4).condition = 'vs';
% paulD(4).epochStarts = 1;
% paulD(4).epochColors = [0 1 1]; % m23 is red

DataHigh(paulD,'DimReduce')


















%% Original gpfa code
displayTime = [10 30]; % how long to display (s)
displayFrames = 1 + displayTime(1) / opts.frameSize : displayTime(2) / opts.frameSize; % convert time to frames
m23Data = dataMat(displayFrames, strcmp(areaLabels, 'M23'))';
m56Data = dataMat(displayFrames, strcmp(areaLabels, 'M56'))';
dsData = dataMat(displayFrames, strcmp(areaLabels, 'DS'))';
vsData = dataMat(displayFrames, strcmp(areaLabels, 'VS'))';


tic
method = 'gpfa';
runIdx = 1;
dat(1).trialId = 1;
dat(1).spikes = m56Data;
% Select number of latent dimensions
xDim = 8;
kernSD = 30;
% Extract neural trajectories
result = neuralTraj(runIdx, dat, 'method', method, 'xDim', xDim,...
    'kernSDList', kernSD, 'seglength', 50, 'binWidth', 50);
% NOTE: This function does most of the heavy lifting.

% Orthonormalize neural trajectories
[estParams, seqTrain] = postprocess(result, 'kernSD', kernSD);
% NOTE: The importance of orthnormalization is described on
%       pp.621-622 of Yu et al., J Neurophysiol, 2009.

% Plot neural trajectories in 3D space
plot3D(seqTrain, 'xorth', 'dimsToPlot', 1:3);

toc






%% Plot a section of continuous behavior with markers at each behavior start time
clear dat

displayTime = 60 + [1 45]; % how long to display (s)
displayFrames = 1 + displayTime(1) / opts.frameSize : displayTime(2) / opts.frameSize; % convert time to frames

startInd = dataBhv.StartTime >= displayTime(1) & dataBhv.StartTime < displayTime(end);
startFrames = dataBhv.StartFrame(startInd);
startFrames = startFrames - displayFrames(1);
labels = dataBhv.ID(startInd);

% m23Data = dataMat(displayFrames, strcmp(areaLabels, 'M23'))';
m56Data = dataMat(displayFrames, strcmp(areaLabels, 'M56'))';
dsData = dataMat(displayFrames, strcmp(areaLabels, 'DS'))';
% vsData = dataMat(displayFrames, strcmp(areaLabels, 'VS'))';

%%

method = 'gpfa';
runIdx = 1;
dat(1).trialId = 1;
dat(1).spikes = dsData;
binWidth = 100;
marks = round(startFrames/binWidth);
marks(marks == 0) = [];

% Select number of latent dimensions
xDim = 8;
kernSD = 30;

% Extract neural trajectories
result = neuralTraj(runIdx, dat, 'method', method, 'xDim', xDim,...
    'kernSDList', kernSD, 'seglength', 50, 'binWidth', binWidth);
% NOTE: This function does most of the heavy lifting.

% Orthonormalize neural trajectories
[estParams, seqTrain] = postprocess(result, 'kernSD', kernSD);
% NOTE: The importance of orthnormalization is described on
%       pp.621-622 of Yu et al., J Neurophysiol, 2009.

% Plot neural trajectories in 3D space
plot3D(seqTrain, 'xorth', 'dimsToPlot', 1:3);
grid on;
%
colorsForPlot = arrayfun(@(x) colors(x,:), labels + 2, 'UniformOutput', false);
colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

seq = seqTrain(1).xorth(1:3,:);
for i = 1 : length(marks)
    scatter3(seq(1,marks(i)), seq(2,marks(i)), seq(3,marks(i)), 100, 'filled', 'MarkerFaceColor', colorsForPlot(i,:));
end








%% Plot a (bigger) section of continuous behavior with markers at each behavior start time:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% 1. Break up data into sections, to feed into GPFA as "trials"
% 2. Run GPFA, and concatenated "trials" back into single long trajectory
% 3. Plot trajectories with behaviors color-coded
idInd = idM56;
% idInd = idDS;

%% 1. Break up data into sections, to feed into GPFA as "trials"
binSize = 2; % How many seconds is each "trial"
nPerBin = round(binSize / opts.frameSize); % Number of frames per trial
nBin = floor(size(dataMat, 1) / nPerBin); % How many "trials" are there in the entire dataMat?

totalTime = binSize * nBin;
% If n is not exactly divisible by k, you might want to handle the remainder
% For this example, we'll trim the excess
% dataMatTrimmed = dataMat(1 : nBin * nPerBin, :);

% Reshape dataMat into "trials"
dataMatReshaped = reshape(dataMat, nPerBin, nBin, size(dataMat, 2));
% bhvIDReshaped = reshape(bhvIDMat, nPerBin, nBin);


% %% use a version of dataMat and dataBhvMat shifted 500 ms to see if it overlaps with non-shifted projections
% shift = .5 / opts.frameSize;
% dataMatShift = dataMat(shift+1 : end, :);
% bhvIDMat = bhvIDMat(shift+1:end);
% 
% binSize = 2; % How many seconds is each "trial"
% nPerBin = round(binSize / opts.frameSize); % Number of frames per trial
% nBin = floor(size(dataMatShift, 1) / nPerBin); % How many "trials" are there in the entire dataMat?
% % If n is not exactly divisible by k, you might want to handle the remainder
% % For this example, we'll trim the excess
% dataMatShift = dataMatShift(1 : nBin * nPerBin, :);
% bhvIDMat = bhvIDMat(1 : nBin * nPerBin);
% dataMatReshaped = reshape(dataMatShift, nPerBin, nBin, size(dataMatShift, 2));


%% 2. Run GPFA, and concatenate "trials" back into single long trajectory
clear dat
nTrial = 400;
nTrial = nBin;
for i = 1 : nTrial
    dat(i).trialId = i;
    dat(i).spikes = squeeze(dataMatReshaped(:, i, idInd))';
end

method = 'gpfa';
runIdx = 1;
binWidth = 50;
% marks = round(startFrames/binWidth);

% Select number of latent dimensions
xDim = 8;
kernSD = 30;

% Extract neural trajectories
result = neuralTraj(runIdx, dat, 'method', method, 'xDim', xDim,...
    'kernSDList', kernSD, 'seglength', 50, 'binWidth', binWidth);
% NOTE: This function does most of the heavy lifting.

% Orthonormalize neural trajectories
[estParams, seqTrain] = postprocess(result, 'kernSD', kernSD);
% NOTE: The importance of orthnormalization is described on
%       pp.621-622 of Yu et al., J Neurophysiol, 2009.
%%
projections = [];
for i = 1 :length(seqTrain)
    projections = [projections; seqTrain(i).xorth'];
end
samples = 1 : binWidth : nTrial * nPerBin;
labels = bhvID(samples);
%% Plot the entire stretch of time
d1 = 1;
d2 = 2;
d3 = 3;
figure(32); clf;
% clf;
hold on;
colorsForPlot = arrayfun(@(x) colors(x,:), labels + 2, 'UniformOutput', false);
colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
plot3(projections(:,d1), projections(:,d2), 1:size(projections, 1), 'linewidth', 1, 'color', [.7 .7 .7]);
scatter3(projections(:,d1), projections(:,d2), 1:size(projections, 1), [], colorsForPlot, 'linewidth', 2);
% plot3(projections(:,d1), projections(:,d2), projections(:,d3), 'linewidth', 1, 'color', [.7 .7 .7]);
% scatter3(projections(:,d1), projections(:,d2), projections(:,d3), [], colorsForPlot, 'linewidth', 2);
xlabel('D1'); ylabel('D2'); zlabel('Time')
grid on;
title('GPFA ')
%%
movieName = ['M56 ', num2str(totalTime/60), ' min 2D over time'];
% movieName = ['M56_', num2str(totalTime/60)];
make_movie(movieName, paths)
%% Iteratively plot shorter stretches of time
plotTime = .5 * 60 * 1000; % how long to plot in each plot (in ms)?
plotFrames = plotTime / binWidth; %
nPlot = floor(size(projections, 1) / plotFrames);

figure(33); clf; hold on;
for iPlot = 1 : nPlot
    clf; hold on;
    iStart = 1 + (iPlot - 1) * plotFrames;
    iSpan = iStart : iStart + plotFrames - 1;
    plot3(projections(iSpan, 1), projections(iSpan, 2), iSpan, 'k', 'linewidth', 1);
    scatter3(projections(iSpan, 1), projections(iSpan, 2), iSpan, 50, colorsForPlot(iSpan, :), 'linewidth', 3);
    grid on;
end

%% Histograms of the different behaviors per dimension, in GPFA space

% Create a figure window that fills the screen
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one
fig = figure(234); clf;
set(fig, 'Position', secondMonitorPosition);
sgtitle('GPFA Dimension histograms')
% Loop through behaviors to plot historgram of each
for jBhv = 1 : length(codes)
    for iDim = 1 : 8
        subplot(4,2,iDim); hold on;

        % edges = floor(min(longTraj(iDim,:))) : .5 : ceil(max(longTraj(iDim,:)));
        edges = min(projections(:,iDim)) : .4 : max(projections(:,iDim));
        binCenters = (edges(1:end-1) + edges(2:end)) / 2;

        % xxxx
        N = histcounts(projections(labels == codes(jBhv), iDim), edges, 'Normalization', 'pdf');
        plot(binCenters, N, 'Color', colors(jBhv,:), 'lineWidth', 3)
        title(['Dimension ', num2str(iDim)])
        xline(0)
    end
end











%%                      Compare common behavioral sequences
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nSeq = 2;
requireValid = [0 1 0];
requireValid = [1 0];
[uniqueSequences, sequenceIndices] = find_unique_sequences(dataBhv, nSeq, requireValid);
uniqueSequences(1:40)
cellfun(@length, sequenceIndices(1:40))

startBhvIdx = 1; % Which behavior in the sequence to plot as the start point. (The index in the sequence)
%% Which sequences to plot
seqIdx = [1 2 14 18];
% seqIdx = [1 14 18 24];
% seqIdx = [2 16 17 23];
% seqIdx = 1;
nTrial = min(30, min(cellfun(@length, sequenceIndices(seqIdx))));

%%
idInd = idM56;

clear dat
method = 'gpfa';
runIdx = 1;
binWidth = 50;

% Select number of latent dimensions
xDim = 8;
kernSD = 30;


for seq = 1 : length(seqIdx)
    iSeq = seqIdx(seq);
    startFrame = dataBhv.StartFrame(sequenceIndices{iSeq} + startBhvIdx - 1);
    durFrame = dataBhv.DurFrame(sequenceIndices{iSeq} + startBhvIdx - 1);
    for jTrial = 1 : nTrial
        trialIdx = (seq-1) * nTrial + jTrial;
        dat(trialIdx).trialId = iSeq;
        % dat(trialIdx).spikes = dataMat(startFrame(jTrial) : startFrame(jTrial) + trajDur - 1, idInd)';
        dat(trialIdx).spikes = dataMat(startFrame(jTrial) : startFrame(jTrial) + durFrame(jTrial), idInd)';
    end
end

% Extract neural trajectories
result = neuralTraj(runIdx, dat, 'method', method, 'xDim', xDim,...
    'kernSDList', kernSD, 'seglength', 50, 'binWidth', binWidth);
% NOTE: This function does most of the heavy lifting.

% Orthonormalize neural trajectories
[estParams, seqTrainM56] = postprocess(result, 'kernSD', kernSD);
% NOTE: The importance of orthnormalization is described on
%       pp.621-622 of Yu et al., J Neurophysiol, 2009.

%%
idInd = idDS;

clear dat

for seq = 1 : length(seqIdx)
    iSeq = seqIdx(seq);
    startFrame = dataBhv.StartFrame(sequenceIndices{iSeq} + startBhvIdx - 1);
    durFrame = dataBhv.DurFrame(sequenceIndices{iSeq} + startBhvIdx - 1);
    for jTrial = 1 : nTrial
        trialIdx = (seq-1) * nTrial + jTrial;
        dat(trialIdx).trialId = iSeq;
        % dat(trialIdx).spikes = dataMat(startFrame(jTrial) : startFrame(jTrial) + trajDur - 1, idInd)';
        dat(trialIdx).spikes = dataMat(startFrame(jTrial) : startFrame(jTrial) + durFrame(jTrial), idInd)';
    end
end

% Extract neural trajectories
result = neuralTraj(runIdx, dat, 'method', method, 'xDim', xDim,...
    'kernSDList', kernSD, 'seglength', 50, 'binWidth', binWidth);
% NOTE: This function does most of the heavy lifting.

% Orthonormalize neural trajectories
[estParams, seqTrainDS] = postprocess(result, 'kernSD', kernSD);
% NOTE: The importance of orthnormalization is described on
%       pp.621-622 of Yu et al., J Neurophysiol, 2009.

%%
uniqueSequences(seqIdx)

% colors = [1 0 0; 0 0 1; 0 .7 0; 0 0 .15];
colors = colors_for_behaviors(codes);
figure(430); clf; hold on; title(['GPFA M56', num2str(xDim)]);
figure(431); clf; hold on; title(['GPFA DS', num2str(xDim)]);
for i = 1 : length(seqTrainM56)
    trialID = seqTrainM56(i).trialId;
    colorIdx = find(seqIdx == trialID);
    bhv1 = uniqueSequences{trialID}(1);
    color1 = colors(bhv1+2,:);
    bhv2 = uniqueSequences{trialID}(2);
    color2 = colors(bhv2+2,:);

    figure(430);
    plot3(seqTrainM56(i).xorth(1,:), seqTrainM56(i).xorth(2,:), seqTrainM56(i).xorth(3,:), '.-', 'Color', color1, 'LineWidth', 2, 'MarkerSize', 10')
    scatter3(seqTrainM56(i).xorth(1,1), seqTrainM56(i).xorth(2,1), seqTrainM56(i).xorth(3,1), 100, color1, 'filled')
    scatter3(seqTrainM56(i).xorth(1,end), seqTrainM56(i).xorth(2,end), seqTrainM56(i).xorth(3,end), 100, color2, 'filled')
    grid on;
    xlabel('D1'); ylabel('D2'); zlabel('D3')

    figure(431)
    plot3(seqTrainDS(i).xorth(1,:), seqTrainDS(i).xorth(2,:), seqTrainDS(i).xorth(3,:), '.-', 'Color', color1, 'LineWidth', 2, 'MarkerSize', 10')
    scatter3(seqTrainDS(i).xorth(1,1), seqTrainDS(i).xorth(2,1), seqTrainDS(i).xorth(3,1), 100, color1, 'filled')
    scatter3(seqTrainDS(i).xorth(1,end), seqTrainDS(i).xorth(2,end), seqTrainDS(i).xorth(3,end), 100, color2, 'filled')
    xlabel('D1'); ylabel('D2'); zlabel('D3')
    grid on;
end
% title('All loco/orient/investigate M56')
% title('All loco/orient/investigate DS')
% title('Locos and some grooms M56')
% title('Locos and some grooms DS')







%%
colors = [1 0 0; 0 0 1; 0 .6 0; .2 .2 .2];
figure(432); clf; hold on;
for i = 1 : length(seqTrain)
    plot3(seqTrain(i).xorth(1,:), seqTrain(i).xorth(2,:), seqTrain(i).xorth(3,:), 'Color', colors(i,:), 'LineWidth', 2)
    scatter3(seqTrain(i).xorth(1,1), seqTrain(i).xorth(2,1), seqTrain(i).xorth(3,1), 100, colors(i,:), 'filled')
    grid on;
end








%%    Load whole dataMat at .001 frameSize is too big. 
% 1. Load entire behavior data and find startFrames of sequences you want.
% 2. Loop through sections of loading dataMat and add any relevant
% sequences in GPFA style
getDataType = 'behavior';
opts = neuro_behavior_options;
opts.collectStart = 0; % seconds
opts.collectFor = 4 * 60 * 60; % seconds
opts.frameSize = .001;

get_standard_data
%% 
nSeq = 2;
requireValid = [0 1 0];
requireValid = [0 1];
[uniqueSequences, sequenceIndices] = find_unique_sequences(dataBhv, nSeq, requireValid);

startBhvIdx = 2; % Which behavior in the sequence to plot as the start point.

%%
% Succesively get sections of dataMat and build a cell array of the
% relevant sequence data, to be transported into GPFA format later.
getDataType = 'all';
secDur = 20*60;
nSec = floor(4*60*60 / secDur);
for iSec = 1 : nSec
    opts.collectStart = (iSec-1) * secDur;
    opts.collectFor = secDur;
    get_standard_data
end







function make_movie(movieName, paths)
%%
fig = gcf();
view(20, 30); % Initial view

% Set up video writer
videoFilename = [paths.dropPath, movieName, '.mp4'];
v = VideoWriter(videoFilename, 'MPEG-4');
v.FrameRate = 30; % Frames per second
open(v);

% Rotate and save frames
numFrames = 360; % One full rotation
for frame = 1:numFrames
    az = frame; % Azimuth angle
    view(az, 30);
    drawnow;
    frameImg = getframe(fig);
    writeVideo(v, frameImg);
end

% Close the video file
close(v);
disp(['Video saved as ', videoFilename]);

end