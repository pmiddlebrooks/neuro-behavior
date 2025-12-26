function [dataMat] = neural_matrix_ms_to_frames(data, frameSize)
%
% Converts a time (ms) X samples (e.g. neurons) matrix (in 1000 Hz) to a binned spike
% matrix, each bin = frameSize (in sec)

durFrames = floor(size(data, 1) / 1000 / frameSize);
dataIndPerFrame = 1000 * frameSize;

% Preallocate data matrix
dataMat = zeros(durFrames, size(data, 2));

% Loop through each neuron and put spike numbers within each frame
for i = 1 : durFrames

    iStartWin = 1 + (i-1) * dataIndPerFrame;
    iWind = round(iStartWin : iStartWin + dataIndPerFrame - 1);
    dataMat(i,:) = sum(data(iWind, :), 1);

end












