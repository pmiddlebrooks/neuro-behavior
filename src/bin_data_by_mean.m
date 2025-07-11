function binned_data = bin_data_by_mean(data, bin_size_ms, sampling_rate)
%BIN_DATA_BY_MEAN Bin time-series data by averaging frames within each bin
%
%   BINNED_DATA = BIN_DATA_BY_MEAN(DATA, BIN_SIZE_MS, SAMPLING_RATE)
%   takes time-series data and bins it by averaging frames within each
%   specified bin size.
%
%   Inputs:
%       data - Time-series data matrix (time x features) or vector
%       bin_size_ms - Bin size in milliseconds
%       sampling_rate - Sampling rate of the data in Hz
%
%   Outputs:
%       binned_data - Binned data matrix (bins x features) or vector
%
%   Example:
%       % Bin 60Hz kinematic data into 50ms bins
%       binned_kinematics = bin_data_by_mean(kinematic_data, 50, 60);
%
%       % Bin 1000Hz neural data into 10ms bins
%       binned_neural = bin_data_by_mean(neural_data, 10, 1000);
%
%   Notes:
%       - If the data length is not perfectly divisible by the bin size,
%         the last incomplete bin is dropped
%       - The function works with both 1D vectors and 2D matrices
%       - For matrices, each column (feature) is binned independently

    % Calculate number of frames per bin
    frames_per_bin = round(bin_size_ms / 1000 * sampling_rate);
    
    if frames_per_bin < 1
        error('Bin size too small for given sampling rate. Increase bin_size_ms or sampling_rate.');
    end
    
    % Get data dimensions
    [n_frames, n_features] = size(data);
    
    % Calculate number of complete bins
    n_bins = floor(n_frames / frames_per_bin);
    
    if n_bins == 0
        error('Data too short for specified bin size. Need at least %d frames.', frames_per_bin);
    end
    
    % Initialize output
    binned_data = zeros(n_bins, n_features);
    
    % Bin the data by averaging frames within each bin
    for i = 1:n_bins
        start_frame = (i-1) * frames_per_bin + 1;
        end_frame = i * frames_per_bin;
        binned_data(i, :) = mean(data(start_frame:end_frame, :), 1);
    end
    
    % If input was a vector, return as vector
    if n_features == 1
        binned_data = binned_data(:);
    end
end 