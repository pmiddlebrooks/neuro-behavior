% Input:
% kinData: Matrix of kinematic data (samples x 2*bodyPositions), where columns are
% arranged as [x1, y1, x2, y2, ..., xN, yN] for each body position.

% Output:
% velocities: Matrix of velocities for each body position (samples-1 x bodyPositions)
% angles: Matrix of relative angles between each pair of body positions (samples-1 x numPairs)

% Get number of samples and number of body positions
[numSamples, ~] = size(kinData);
numPositions = length(bodyParts);

% Preallocate velocity matrix
velocities = zeros(numSamples - 1, numPositions);
velNames = {};
% Calculate velocities for each body position
for iPos = 1:numPositions

    xCol = find(contains(kinHeader(3,:), bodyParts(iPos)), 1);
    % Extract x and y coordinates for the current position
    xPos = kinData{:, xCol};
    yPos = kinData{:, xCol+1};
    
    % Compute velocity as the Euclidean distance between successive samples
    deltaX = diff(xPos);
    deltaY = diff(yPos);
    velocities(:, iPos) = sqrt(deltaX.^2 + deltaY.^2);
    velNames{iPos} = bodyParts{iPos};
end

% Preallocate angle matrix (unique pairs of body positions)
numPairs = nchoosek(numPositions, 2);
angles = zeros(numSamples - 1, numPairs);
pairIdx = 1;
angleNames = {};
% Calculate relative angles between each pair of body positions
for i = 1:numPositions - 1
    iXcol = find(contains(kinHeader(3,:), bodyParts(i)), 1);
    iYcol = iXcol + 1;
    for j = i + 1:numPositions
        jXcol = find(contains(kinHeader(3,:), bodyParts(j)), 1);
        jYcol = jXcol+1;

        % Extract x and y coordinates for each position
        x1 = kinData{:, iXcol}; y1 = kinData{:, iYcol};
        x2 = kinData{:, jXcol}; y2 = kinData{:, jYcol};
        
        % Compute differences in coordinates
        deltaX1 = diff(x1); deltaY1 = diff(y1);
        deltaX2 = diff(x2); deltaY2 = diff(y2);
        
        % Calculate the angle between vectors (deltaX1, deltaY1) and (deltaX2, deltaY2)
        dotProduct = (deltaX1 .* deltaX2 + deltaY1 .* deltaY2);
        mag1 = sqrt(deltaX1.^2 + deltaY1.^2);
        mag2 = sqrt(deltaX2.^2 + deltaY2.^2);
        
        % Compute relative angle in radians and store
        angles(:, pairIdx) = acos(dotProduct ./ (mag1 .* mag2));
        pairIdx = pairIdx + 1;

        if sum(isnan(angles(:,pairIdx-1)))
            disp('asdf')
        end
        angleNames = [angleNames, bodyParts(i) + "_" + bodyParts(j)];
    end
end
headers = [velNames, angleNames];
velocitiesAngles = [velocities, angles];

% Convert the matrix to a table and assign column headers
velocitiesAnglesTable = array2table(velocitiesAngles, 'VariableNames', headers);

% Save the table as a CSV file
writetable(velocitiesAnglesTable, [paths.saveDataPath,  'velocitiesAngles.csv']);

% Outputs: velocities and angles matrices containing the velocities for each
% body position and the relative angles between each pair of body positions.
