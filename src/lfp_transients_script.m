

% Script to find transients in power spectra bands and their temporal patterns

% Example input: binned power spectra (time x frequency bands)
binnedPower = randn(1000, 4); % Replace with actual binned power data
bandNames = {'Alpha', 'Beta', 'LowGamma', 'HighGamma'}; % Name of each band
nStd = 2; % Threshold in standard deviations for identifying transients

% Z-score normalize the power bands
zScoredPower = zscore(binnedPower);
[numFrames, numBands] = size(zScoredPower);

% Initialize a struct to store transient times for each band and type
transientTimes = struct();

for b = 1:numBands
    bandName = bandNames{b};
    bandData = zScoredPower(:, b);

    % Find increases (positive transients)
    increaseFrames = find(diff(bandData > nStd) == 1) + 1;

    % Find decreases (negative transients)
    decreaseFrames = find(diff(bandData < -nStd) == 1) + 1;

    % Store results
    transientTimes.(bandName).Increase = increaseFrames;
    transientTimes.(bandName).Decrease = decreaseFrames;
end

% Analyze patterns of co-occurring transients
coOccurrences = struct();

for b1 = 1:numBands
    band1Name = bandNames{b1};
    for b2 = 1:numBands
        if b1 ~= b2
            band2Name = bandNames{b2};

            % Check for co-occurrences (same frames)
            coOccurrences.(band1Name).(band2Name).IncreaseWithIncrease = intersect(...
                transientTimes.(band1Name).Increase, transientTimes.(band2Name).Increase);

            coOccurrences.(band1Name).(band2Name).IncreaseWithDecrease = intersect(...
                transientTimes.(band1Name).Increase, transientTimes.(band2Name).Decrease);

            coOccurrences.(band1Name).(band2Name).DecreaseWithIncrease = intersect(...
                transientTimes.(band1Name).Decrease, transientTimes.(band2Name).Increase);

            coOccurrences.(band1Name).(band2Name).DecreaseWithDecrease = intersect(...
                transientTimes.(band1Name).Decrease, transientTimes.(band2Name).Decrease);
        end
    end
end

% Display results
for b = 1:numBands
    bandName = bandNames{b};
    fprintf('Band: %s\n', bandName);
    fprintf('  Increases: %d events\n', length(transientTimes.(bandName).Increase));
    fprintf('  Decreases: %d events\n', length(transientTimes.(bandName).Decrease));
end

% Example visualization for one band
exampleBand = 'Alpha';
figure;
hold on;
plot(zScoredPower(:, strcmp(bandNames, exampleBand)), 'k');
plot(transientTimes.(exampleBand).Increase, zScoredPower(transientTimes.(exampleBand).Increase, strcmp(bandNames, exampleBand)), 'g^', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
plot(transientTimes.(exampleBand).Decrease, zScoredPower(transientTimes.(exampleBand).Decrease, strcmp(bandNames, exampleBand)), 'rv', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
hold off;
xlabel('Frame');
ylabel('Z-scored Power');
title(sprintf('Transients in %s Band', exampleBand));
legend('Power', 'Increase', 'Decrease');
