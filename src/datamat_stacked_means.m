function [stackedActivity, stackedLabels] = datamat_stacked_means(dataMat, bhvID, minFrames)
%%
% Inputs:
% bhvID: Vector of behavior labels (samples x 1)
% dataMat: Neural activity matrix (samples x neurons)

% Output:
% stackedActivity: Vertically stacked averaged neural activity (time bins x neurons)
% stackedLabels: Corresponding behavior labels for each row in stackedActivity

% Find all unique behavior IDs
uniqueBhvIDs = unique(bhvID);

% Preallocate storage
nBehaviors = length(uniqueBhvIDs);
medianTrimmedLengths = NaN(1, nBehaviors); % Median trimmed length per behavior
stackedActivity = []; % To hold vertically stacked averaged activity
stackedLabels = []; % To track behavior labels for stacked rows

% Loop through each behavior
for bhvIdx = 1:nBehaviors
    % Get the current behavior ID
    bhvCheck = uniqueBhvIDs(bhvIdx);
    
    % Find indices of the specified behavior
    isBhvCheck = (bhvID == bhvCheck);
    
    % Identify the start and end indices of each bout
    boutStart = find(diff([0; isBhvCheck]) == 1); % Start when switching to bhvCheck
    boutEnd = find(diff([isBhvCheck; 0]) == -1);  % End when switching away from bhvCheck
    
    % Filter bouts to include only those with length >= minFrames
    validBouts = (boutEnd - boutStart + 1) >= minFrames;
    boutStart = boutStart(validBouts);
    boutEnd = boutEnd(validBouts);
    
    % Recalculate number of valid bouts
    nValidBouts = length(boutStart);
    
    % Skip if no valid bouts
    if nValidBouts == 0
        continue;
    end
    
    % Compute the trimmed bout lengths (2nd to next-to-last bin)
    boutLengths = boutEnd - boutStart + 1; % Length of valid bouts
    trimmedBoutLengths = boutLengths - 0; % Length after trimming
    medianLength = round(median(trimmedBoutLengths(trimmedBoutLengths >= minFrames))); % Median trimmed length
    medianTrimmedLengths(bhvIdx) = medianLength;
    
    % Initialize the 3D matrix with NaNs for trimmed bouts
    neuralDataByBout = NaN(max(trimmedBoutLengths), size(dataMat, 2), nValidBouts);
    
    % Loop through each valid bout and collect the neural data
    for boutIdx = 1:nValidBouts
        % Get the current bout's start and end indices
        boutRange = (boutStart(boutIdx)):(boutEnd(boutIdx) - 1); % 1st to next-to-last bins
        
        % Collect neural data for the current bout
        neuralData = dataMat(boutRange, :);
        
        % Assign the neural data into the 3D matrix
        neuralDataByBout(1:length(boutRange), :, boutIdx) = neuralData;
    end
    
    % Compute the nanmean across bouts
    meanNeuralActivity = nanmean(neuralDataByBout, 3);
    
    % Trim the data to the first `medianLength` time bins
    trimmedActivity = meanNeuralActivity(1:medianLength, :);
    
    % Stack the trimmed activity vertically
    stackedActivity = [stackedActivity; trimmedActivity];
    
    % Track the behavior labels for each row in the stacked activity
    stackedLabels = [stackedLabels; repmat(bhvCheck, medianLength, 1)];
end
