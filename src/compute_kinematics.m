function [features, scaled_features] = compute_kinematics(data_path, save_dir, fps)
% COMPUTE_KINEMATICS Extracts kinematic features from DeepLabCut tracking data
%
% This function processes DeepLabCut CSV files to extract kinematic features
% including displacement, distance between points, and angular features.
%
% Inputs:
%   data_path - Path to DeepLabCut CSV file
%   save_dir - Directory to save output features
%   fps - Camera frame rate (default: 60)
%
% Outputs:
%   features - Raw kinematic features
%   scaled_features - Z-scored kinematic features
%
% Example:
%   [features, scaled_features] = compute_kinematics('path/to/dlc_data.csv', 'output_dir', 60);

if nargin < 3
    fps = 60; % Default frame rate
end

% Read and process the data
[currdf, perc_rect] = adp_filt(data_path);

% Extract features
[features, scaled_features] = get_features(currdf, fps);

% Save features if save directory is provided
if nargin >= 2 && ~isempty(save_dir)
    % Create save directory if it doesn't exist
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    
    % Generate filename
    [~, filename, ~] = fileparts(data_path);
    file_name = [filename, '_kinematics'];
    output_path = fullfile(save_dir, file_name);
    
    % Save features as .mat file
    save(output_path, 'features', 'scaled_features', '-v7.3');
    fprintf('Features saved to: %s\n', output_path);
    
    % Save features as CSV file (no headers)
    csv_output_path = [output_path, '.csv'];
    writematrix(features', csv_output_path); % Transpose to get features as columns
    fprintf('Features saved to: %s\n', csv_output_path);
    
    % Save scaled features as CSV file (no headers)
    scaled_csv_output_path = [output_path, '_scaled.csv'];
    writematrix(scaled_features', scaled_csv_output_path); % Transpose to get features as columns
    fprintf('Scaled features saved to: %s\n', scaled_csv_output_path);
end

end

function moving_avg = boxcar_center(a, n)
% BOXCAR_CENTER Applies centered moving average with boxcar window
%
% Inputs:
%   a - Input data vector
%   n - Window size
%
% Outputs:
%   moving_avg - Smoothed data

% Convert to column vector if needed
if isrow(a)
    a = a';
end

% Initialize output
moving_avg = zeros(size(a));

% Handle edge cases
half_window = floor(n/2);

% Apply moving average
for i = 1:length(a)
    start_idx = max(1, i - half_window);
    end_idx = min(length(a), i + half_window);
    moving_avg(i) = mean(a(start_idx:end_idx));
end

end

function [currdf_filt, perc_rect] = adp_filt(data_path)
% ADP_FILT Adaptive filtering of DeepLabCut data based on likelihood
%
% Inputs:
%   data_path - Path to DeepLabCut CSV file
%
% Outputs:
%   currdf_filt - Filtered position data
%   perc_rect - Percentage of frames below likelihood threshold

% Read CSV data
data = readtable(data_path);
currdf = table2array(data);

% Find column indices for likelihood, x, and y
headers = data.Properties.VariableNames;
lIndex = [];
xIndex = [];
yIndex = [];

for i = 1:length(headers)
    if contains(headers{i}, 'likelihood')
        lIndex = [lIndex, i];
    elseif contains(headers{i}, 'x')
        xIndex = [xIndex, i];
    elseif contains(headers{i}, 'y')
        yIndex = [yIndex, i];
    end
end

% Extract data
datax = currdf(:, xIndex);
datay = currdf(:, yIndex);
data_lh = currdf(:, lIndex);

% Initialize output
currdf_filt = zeros(size(datax, 1), length(xIndex) * 2);
perc_rect = zeros(1, length(xIndex));

% Process each tracked point
fprintf('Processing %d tracked points...\n', length(xIndex));
for x = 1:length(xIndex)
    % Calculate likelihood threshold using histogram
    lh_data = data_lh(:, x);
    [counts, edges] = histcounts(lh_data, 50);
    
    % Find first significant rise in histogram
    diff_counts = diff(counts);
    rise_indices = find(diff_counts >= 0);
    
    if ~isempty(rise_indices)
        if rise_indices(1) > 1
            llh = edges(rise_indices(1));
        else
            llh = edges(rise_indices(2));
        end
    else
        llh = median(lh_data); % Fallback threshold
    end
    
    % Calculate percentage below threshold
    perc_rect(x) = sum(lh_data < llh) / length(lh_data);
    
    % Apply adaptive filtering
    currdf_filt(1, (2*x-1):(2*x)) = [datax(1, x), datay(1, x)];
    
    for i = 2:size(data_lh, 1)
        if lh_data(i) < llh
            % Use previous frame's position
            currdf_filt(i, (2*x-1):(2*x)) = currdf_filt(i-1, (2*x-1):(2*x));
        else
            % Use current frame's position
            currdf_filt(i, (2*x-1):(2*x)) = [datax(i, x), datay(i, x)];
        end
    end
end

end

function [features, scaled_features] = get_features(data, fps)
% GET_FEATURES Extracts kinematic features from position data
%
% Inputs:
%   data - Position data matrix (frames x 2*num_points)
%   fps - Camera frame rate
%
% Outputs:
%   features - Raw kinematic features
%   scaled_features - Z-scored features

% Calculate window size for smoothing (50ms window)
window = round(0.05 / (1/fps) * 2 - 1);

data_n_len = size(data, 1);
num_points = size(data, 2) / 2;

% Initialize feature arrays
dxy_list = [];
disp_list = [];

fprintf('Computing displacement and distance features...\n');

% Calculate displacement between consecutive frames
for r = 1:data_n_len-1
    disp = [];
    for c = 1:2:size(data, 2)
        disp = [disp, norm(data(r+1, c:c+1) - data(r, c:c+1))];
    end
    disp_list = [disp_list; disp];
end

% Calculate distances between all point pairs
for r = 1:data_n_len
    dxy = [];
    for i = 1:2:size(data, 2)
        for j = (i+2):2:size(data, 2)
            dxy = [dxy, data(r, i:i+1) - data(r, j:j+1)];
        end
    end
    dxy_list = [dxy_list; dxy];
end

% Interpolate displacement to match frame times
interp_times = 1:data_n_len;
computed_times = 1.5:(data_n_len-0.5);

disp_r = zeros(data_n_len, size(disp_list, 2));
for i = 1:size(disp_list, 2)
    disp_r(:, i) = interp1(computed_times, disp_list(:, i), interp_times, 'linear', 'extrap');
end

fprintf('Displacement shape: %s, Distance shape: %s\n', mat2str(size(disp_r)), mat2str(size(dxy_list)));

% Apply smoothing and compute additional features
disp_boxcar = [];
dxy_eu = zeros(data_n_len, size(dxy_list, 2));
ang = zeros(data_n_len, size(dxy_list, 2));
dxy_boxcar = [];
ang_boxcar = [];

fprintf('Computing smoothed features...\n');

% Smooth displacement features
for l = 1:size(disp_r, 2)
    disp_boxcar = [disp_boxcar, boxcar_center(disp_r(:, l), window)];
end

% Compute Euclidean distances and angles
for k = 1:size(dxy_list, 2)
    % Extract x,y components for this pair
    pair_data = dxy_list(:, (2*k-1):(2*k));
    
    % Compute Euclidean distances
    for kk = 1:data_n_len
        dxy_eu(kk, k) = norm(pair_data(kk, :));
        
        % Compute angles between consecutive frames
        if kk < data_n_len
            v1 = pair_data(kk, :);
            v2 = pair_data(kk+1, :);
            
            % Compute angle using cross product
            if norm(v1) > 0 && norm(v2) > 0
                cos_angle = dot(v1, v2) / (norm(v1) * norm(v2));
                cos_angle = max(-1, min(1, cos_angle)); % Clamp to [-1, 1]
                angle_rad = acos(cos_angle);
                ang(kk, k) = angle_rad * 180 / pi;
            else
                ang(kk, k) = 0;
            end
        end
    end
    
    % Apply smoothing
    dxy_boxcar = [dxy_boxcar, boxcar_center(dxy_eu(:, k), window)];
    ang_boxcar = [ang_boxcar, boxcar_center(ang(:, k), window)];
end

% Organize features
disp_feat = disp_boxcar;
dxy_feat = dxy_boxcar;
ang_feat = ang_boxcar;

fprintf('Feature shapes - Displacement: %s, Distance: %s, Angle: %s\n', ...
    mat2str(size(disp_feat)), mat2str(size(dxy_feat)), mat2str(size(ang_feat)));

% Combine all features
features = [dxy_feat; ang_feat; disp_feat'];

% Z-score normalization
scaled_features = zscore(features, 0, 2); % Normalize along rows

end

function main()
% MAIN Example usage of compute_kinematics function
%
% This function demonstrates how to use compute_kinematics with example paths

% Example paths (modify these for your data)
in_paths = {
    'path/to/your/dlc_data1.csv',
    'path/to/your/dlc_data2.csv'
};

save_dirs = {
    'path/to/output1',
    'path/to/output2'
};

% Process each file
for i = 1:length(in_paths)
    fprintf('Processing file %d of %d: %s\n', i, length(in_paths), in_paths{i});
    
    [features, scaled_features] = compute_kinematics(in_paths{i}, save_dirs{i}, 60);
    
    fprintf('Features shape: %s, Scaled features shape: %s\n', ...
        mat2str(size(features)), mat2str(size(scaled_features)));
end

end
