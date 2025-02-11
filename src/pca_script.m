%% Get data from get_standard_data

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 60 * 60; % seconds
opts.frameSize = .1;
% opts.shiftAlignFactor = .05; % I want spike counts xxx ms peri-behavior label

getDataType = 'spikes';
get_standard_data

[dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);

colors = colors_for_behaviors(codes);

%% What to fit
forDim = 3;
iDim = forDim;
idSelect = idDS;
selectFrom = 'DS';
fitType = 'PCA';
%% Which data to model:
preInd = [diff(bhvIDMat) ~= 0; 0]; % 1 frame prior to all behavior transitions

%% High-D neral matrix
modelData = zscore(dataMat);

%%

%% within-bout
modelInd = ~preInd & ~[preInd(2:end); 0] & ~(bhvID == -1);
modelID = bhvID(modelInd);
%% all data
modelInd = 1:length(bhvIDMat);
modelID = bhvID;


%% Perform PCA
[coeff, projSelect, ~, ~, explained] = pca(modelData(modelInd, idSelect));

%% Plot results
if plotFullMap && ~umapTransOnly
                    colorsForPlot = arrayfun(@(x) colors(x,:), modelID + 2, 'UniformOutput', false);
                    colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
                    % colorsForPlot = [.2 .2 .2];
                    figH = figHFull;
                    plotPos = [monitorOne(1), 1, monitorOne(3)/2, monitorOne(4)];
                    % titleM = sprintf('%s %s bin=%.2f min_dist=%.2f spread=%.1f nn=%d', selectFrom, fitType, opts.frameSize, min_dist(x), spread(y), n_neighbors(z));
                    titleM = sprintf('%s %s bin=%.2f', selectFrom, fitType, opts.frameSize);
                    % titleM = [selectFrom, ' ', fitType, ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
                    plotFrames = 1:size(projSelect, 1);
                    plot_3d_scatter
                end














%% Create a neural matrix. Each column is a neuron. Each row are spike counts peri-onset of each behavior.
behaviorID = [];
neuralMatrix = [];
periEventTime = -.2 : opts.frameSize : .2; % seconds around onset
dataWindow = fullStartInd + periEventTime(1:end-1) / opts.frameSize; % frames around onset (remove last frame)

for iBhv = 1 : length(analyzeCodes)
    behaviorID = [behaviorID; analyzeCodes(iBhv) * ones(size(eventMatZ{iBhv}, 3), 1)];
    iMat = sum(eventMatZ{iBhv}(dataWindow, :, :), 1);
    iMat = permute(iMat, [3 2 1]);
            neuralMatrix = [neuralMatrix; iMat];
end








%%  Plot PCA projections over time with behaviors labeled by color
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make a modified dataMat with big (e.g. 400ms) bins
binSize = .4;
nPerBin = round(binSize / opts.frameSize);
nBin = floor(size(dataMat, 1) / nPerBin);

% If n is not exactly divisible by k, you might want to handle the remainder
% For this example, we'll trim the excess
% dataMatTrimmed = dataMat(1 : nBin * nPerBin, :);

% Reshape and sum dataMat
dataMatReshaped = reshape(dataMat, nPerBin, nBin, size(dataMat, 2));
dataMatMod = squeeze(sum(dataMatReshaped, 1));

bhvIDReshaped = reshape(bhvIDMat, nPerBin, nBin);
bhvIDMod = bhvIDReshaped(floor(nPerBin/2) + 1, :)';

%%
idInd = cell2mat(idAll); area = 'All';
idInd = idM56; area = 'M56';
% idInd = idDS; area = 'DS';

% Only use part of the dataMat (if binSize is small)
nFrame = floor(size(dataMatMod, 1) / 1.1);  
frameWindow = 1 : nFrame;
% windowStart = floor(size(dataMatMod, 1) / 1);
% frameWindow = windowStart : windowStart + nFrame - 1;



%% Perform PCA
[coeff, score, ~, ~, explained] = pca(dataMatMod(frameWindow, idInd));

%%
hfig = figure(240);
colorsForPlot = arrayfun(@(x) colors(x,:), bhvIDMod(frameWindow) + 2, 'UniformOutput', false);
colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
scatter3(score(:,1), score(:,2), 1:nFrame, [], colorsForPlot, 'linewidth', 2);

title(['PCA ' area, ' binSize = ', num2str(binSize)])
saveas(gcf, fullfile(paths.figurePath, ['PCA ' area, ' binsize ', num2str(binSize), '.png']), 'png')














%%

[coeff,score,latent,tsquared,explained,mu] = PlotPCAWithBehaviors3D(neuralMatrix(:, idM56), behaviorID);




%% PCA trajectories for each trial (adapted from Matt Smith's PCA scripts)
figure(7);
onsetInd = length(dataWindow)/2 + 1;
for iBhv = 1 : length(analyzeCodes)
    iStartFrames = 1 + floor(dataBhv.StartTime(dataBhv.ID == analyzeCodes(iBhv)) ./ opts.frameSize);
    iStartFrames = iStartFrames(3:end-3);
    for jStart = 1 : length(iStartFrames)
        cla
        hold on
        jProj = dataMatZ(iStartFrames(jStart) + dataWindow, idM56) * coeff;
        plot3(jProj(:,1), jProj(:,2), jProj(:,3))
        scatter3(jProj(onsetInd, 1), jProj(onsetInd, 2), jProj(onsetInd, 3), 100, 'filled')
    end
end


%%
nComponents = 5;
pca_project_and_plot(neuralMatrix(:, idM56), behaviorID, nComponents)
% pca_project_and_plot(neuralMatrix(:, [idM56 idDS]), behaviorID)

%%

function pca_project_and_plot(neuralMatrix, behaviorID, nComponents)
bhvList = unique(behaviorID);
% colors = distinguishable_colors(length(bhvList));
colors = colors_for_behaviors(bhvList);

% Perform PCA
[coeff, score, ~, ~, explained] = pca(neuralMatrix);

% Create 3x3 grid of subplots
% Get monitor positions and size
monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(80);
clf
set(fig, 'Position', secondMonitorPosition);
nPlot = nComponents - 1;
[ha, pos] = tight_subplot(3, 3);

for bhv = 1 : length(bhvList)
    bhvIdx = behaviorID == bhvList(bhv);
    bhvColor = colors(bhv,:);
    for i = 1 : nComponents-1
        axes(ha(i))
        % subplot(3, 3, i);
        hold on

        % Extract the successive pairs of components
        componentX = score(bhvIdx, i);
        componentY = score(bhvIdx, i+1);

        % Create a scatter plot of the projected data for each pair of components
        plotDataOrMean = 0;
        if plotDataOrMean
            scatter(componentX, componentY, 30, bhvColor);
        else
            x = mean(componentX);
            y = mean(componentY);
            % w = std(componentX) / sqrt(length(componentX));
            % h = std(componentY) / sqrt(length(componentY));
            w = std(componentX);
            h = std(componentY);

            s = scatter(x, y, 100, bhvColor, 'filled');

            % Generate points on the ellipse
            theta = linspace(0, 2 * pi, 100); % Generate 100 points
            ellipseX = (w / 2) * cos(theta) + x; % X coordinates
            ellipseY = (h / 2) * sin(theta) + y; % Y coordinates

            % Plot the ellipse
            fill(ellipseX, ellipseY, bhvColor, 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Fill the ellipse with semi-transparency

        end
        title(['PCA Components ', num2str(i), ' and ', num2str(i+1)]);
        % xlabel(['Component ', num2str(i)]);
        % ylabel(['Component ', num2str(i+1)]);
        xlabel([]);
        ylabel([]);
        grid on;
    end
end
end



function [coeff,score,latent,tsquared,explained,mu] = PlotPCAWithBehaviors3D(neuralMatrix, behaviorID)
% PlotPCAWithBehaviors3D: Performs PCA on a neural spiking activity matrix and plots the first three principal components.
% neuralMatrix is a n X p matrix (n time points, p neurons).
% behaviorID is a vector of behavior categories


% Generate an n X 3 matrix with each row representing a unique RGB color value.
% n is the input integer representing the number of colors to generate.

% Preallocate the matrix for efficiency
rgbMatrix = zeros(length(unique(behaviorID)), 3);

% Filling the matrix with random RGB values
for i = 1:size(rgbMatrix, 1)
    % Randomly generating RGB values in the range 0 to 255
    rgbMatrix(i, :) = randi([0, 255], 1, 3);
end
rgbMatrix = rgbMatrix ./ 255;


% Check if the length of behaviorID matches the number of rows in neuralMatrix
if length(behaviorID) ~= size(neuralMatrix, 1)
    error('Length of behaviorID must match the number of rows in neuralMatrix');
end

% Perform PCA on the neuralMatrix.
[coeff, score, latent, tsquared, explained, mu] = pca(neuralMatrix);

coeff10 = coeff(:, 1:10);
% Project the data into PCA space:
projData = neuralMatrix * coeff10;


% Plotting the first three principal components.
figure(23);
clf
hold on;
% Adding labels and title.
xlabel('First Principal Component');
ylabel('Second Principal Component');
zlabel('Third Principal Component');
title('3D PCA of Neural Spiking Activity');

% Looping through each behavior to plot.
nRandTrial = 30;
for i = 1:length(unique(behaviorID))
    behaviorIndices = find(behaviorID == i);
    behaviorIndices = behaviorIndices(randperm(length(behaviorIndices)));
    % behaviorIndices = behaviorIndices(1:min(length(behaviorIndices), nRandTrial));
    scatter3(score(behaviorIndices,1), score(behaviorIndices,2), score(behaviorIndices,3), 80, rgbMatrix(i,:));
    % scatter3(projData(behaviorIndices,1), projData(behaviorIndices,2), projData(behaviorIndices,3), 36, rgbMatrix(i,:), 'filled');
end

% legend('Behavior 1', 'Behavior 2', 'Behavior 3');
hold off;
end

