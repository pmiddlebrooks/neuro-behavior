function mahalanobisDistances = mahalanobis_distances(dataObs, dataRefSamples, onsetFrames, onsetLabels)
    % Input:
    %   dataObs: Time_points x n matrix
    %   timeIndices: Vector of input time indices
    %   startLabels: Vector of labels corresponding to the input time
    %   indices. Needs to be 1 : number of labels

    % Check if the input sizes are consistent
    if numel(onsetFrames) ~= numel(onsetLabels)
        error('Number of time indices must match the number of labels.');
    end

    % Initialize Mahalanobis distance matrix
    numClasses = max(onsetLabels);
    % mahalanobisDistances = zeros(numClasses);
    mahalanobisDistances = cell(numClasses, 1);

    % % Calculate mean and covariance for each class
    % classMeans = zeros(numClasses, size(dataMat, 2));
    % classCovariances = cell(numClasses, 1);
    % 
    % for classLabel = 1:numClasses
    %     classData = dataMat(startLabels == classLabel, :);
    %     classMeans(classLabel, :) = mean(classData);
    %     classCovariances{classLabel} = cov(classData);
    % end

    % % Calculate Mahalanobis distance for each pair of classes
    % for i = 1:numClasses
    %     for j = i+1:numClasses
    %         % mahalanobisDistances(i, j) = mahal(classMeans(i, :), dataMat(startLabels == j, :), classCovariances{i});
    %         % mahalanobisDistances(j, i) = mahal(classMeans(j, :), dataMat(startLabels == i, :), classCovariances{j});
    %         mahalanobisDistances(i, j) = mahal(classMeans(i, :), dataMat(startLabels == j, :));
    %         mahalanobisDistances(j, i) = mahal(classMeans(j, :), dataMat(startLabels == i, :));
    %     end
    % end

    % zDataMat = 

    % Calculate Mahalanobis distance for each behvaior
    for i = 1:numClasses
        iOnsetFrames = onsetFrames(onsetLabels == i);
            mahalanobisDistances{i} = mahal(dataMat(iOnsetFrames, :), dataMat(iOnsetFrames, :));
size(dataMat(iOnsetFrames, :))
             mahalanobisDistances{i}(1:10)

%              mdist = mahal(dataMat(iOnsetFrames, j), dataMat(iOnsetFrames, j))
%                  % Calculate the mean of the sample data
%     sampleMean = mean(dataMat(iOnsetFrames, :));
% 
%     % Calculate the covariance matrix of the sample data
%     sampleCovariance = cov(dataMat(iOnsetFrames, :));
% dataMat(iOnsetFrames, :)
% sampleMean
% sampleCovariance
%                      mahalanobisCheck{i} = sqrt((dataMat(iOnsetFrames, :) - sampleMean) / sampleCovariance .* (dataMat(iOnsetFrames, :) - sampleMean)');
%  [mahalanobisDistances{i}(1:10), mahalanobisCheck{i}(1:10)]
    end
end
