%% Got to spiking_script and get behavioral and neural data

%%

% Create a neural matrix. Each column is a neuron. Each row are spike
% counts peri-onset of each behavior.
behaviorOnsets = [];
neuralMatrix = [];
periEventTime = -.4 : opts.frameSize : .4; % seconds around onset
dataWindow = round(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)

for iBhv = 1 : length(analyzeCodes)
    iStartFrames = floor(dataBhv.bhvStartTime(dataBhv.bhvID == analyzeCodes(iBhv)) ./ opts.frameSize);
    iStartFrames = iStartFrames(3:end-3);
    behaviorOnsets = [behaviorOnsets; iBhv * ones(length(iStartFrames), 1)];
    for jStart = 1 : length(iStartFrames)
        neuralMatrix = [neuralMatrix; sum(dataMatZ(iStartFrames(jStart) + dataWindow, :))];
    end
end
%%

[coeff,score,latent,tsquared,explained,mu] = PlotPCAWithBehaviors3D(neuralMatrix(:, idM56), behaviorOnsets);


%%
figure(7);
onsetInd = length(dataWindow)/2 + 1;
for iBhv = 1 : length(analyzeCodes)
    iStartFrames = floor(dataBhv.bhvStartTime(dataBhv.bhvID == analyzeCodes(iBhv)) ./ opts.frameSize);
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

function [coeff,score,latent,tsquared,explained,mu] = PlotPCAWithBehaviors3D(neuralMatrix, behaviorOnsets)
% PlotPCAWithBehaviors3D: Performs PCA on a neural spiking activity matrix and plots the first three principal components.
% neuralMatrix is a n X p matrix (n time points, p neurons).
% behaviorOnsets is a vector of onset times for different behaviors.
  

% Generate an n X 3 matrix with each row representing a unique RGB color value.
    % n is the input integer representing the number of colors to generate.

    % Preallocate the matrix for efficiency
    rgbMatrix = zeros(length(unique(behaviorOnsets)), 3);

    % Filling the matrix with random RGB values
    for i = 1:size(rgbMatrix, 1)
        % Randomly generating RGB values in the range 0 to 255
        rgbMatrix(i, :) = randi([0, 255], 1, 3);
    end
rgbMatrix = rgbMatrix ./ 255;


% Check if the length of behaviorOnsets matches the number of rows in neuralMatrix
if length(behaviorOnsets) ~= size(neuralMatrix, 1)
    error('Length of behaviorOnsets must match the number of rows in neuralMatrix');
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

% Looping through each behavior to plot.
for i = 1:length(unique(behaviorOnsets))
    behaviorIndices = find(behaviorOnsets == i);
    scatter3(score(behaviorIndices,1), score(behaviorIndices,2), score(behaviorIndices,3), 36, rgbMatrix(i,:), 'filled');
    % scatter3(projData(behaviorIndices,1), projData(behaviorIndices,2), projData(behaviorIndices,3), 36, rgbMatrix(i,:), 'filled');
end

% Adding labels and title.
xlabel('First Principal Component');
ylabel('Second Principal Component');
zlabel('Third Principal Component');
title('3D PCA of Neural Spiking Activity');
% legend('Behavior 1', 'Behavior 2', 'Behavior 3');
hold off;
end

