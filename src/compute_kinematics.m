
function [features, scaled_features] = compute_kinematics(data_path, save_dir, fps)
% COMPUTE_KINEMATICS Extracts kinematic features from DeepLabCut tracking data
%
% This function processes DeepLabCut CSV files to extract kinematic features
% including displacement, distance between points, and angular features.
%
% Inputs:
%   data_path - Path to DeepLabCut CSV file OR data matrix (time x features)
%   save_dir - Directory to save output features
%   fps - Camera frame rate (default: 60)
%
% Outputs:
%   features - Raw kinematic features
%   scaled_features - Z-scored kinematic features
%
% Example:
%   [features, scaled_features] = compute_kinematics('path/to/dlc_data.csv', 'output_dir', 60);
%   [features, scaled_features] = compute_kinematics(data_matrix, '', 60);

if nargin < 3
    fps = 60; % Default frame rate
end


fprintf('Step 1: Running adp_filt...\n');
[currdf, perc_rect] = adp_filt(data_path);
fprintf('adp_filt output shape: %s\n', mat2str(size(currdf)));
fprintf('perc_rect: %s\n', mat2str(perc_rect));

% Extract featuresdata
fprintf('Step 2: Running get_features...\n');
[features, scaled_features] = get_features(currdf, fps);
fprintf('get_features output shapes - features: %s, scaled_features: %s\n', ...
    mat2str(size(features)), mat2str(size(scaled_features)));


return
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

% % Convert to column vector if needed
% if isrow(a)
%     a = a';
% end
% 
% % VECTORIZED VERSION - Much faster than nested loops
% % Use conv for moving average (much faster than loops)
% half_window = floor(n/2);
% if n > 1
%     % Create boxcar kernel
%     kernel = ones(n, 1) / n;
% 
%     % Apply convolution with proper padding
%     padded_a = [a(1)*ones(half_window,1); a; a(end)*ones(half_window,1)];
%     moving_avg = conv(padded_a, kernel, 'valid');
% else
%     moving_avg = a;
% end

% ORIGINAL LOOP-BASED VERSION (commented out for reference)
% % Initialize output
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

% Read CSV data - DeepLabCut format has headers starting at "Coords" row
fprintf('Reading DeepLabCut CSV file: %s\n', data_path);

% Read the raw CSV file to find the header row
fid = fopen(data_path, 'r');
if fid == -1
    error('Could not open file: %s', data_path);
end

% Find the row that starts with "Coords"
header_row = 0;
line_num = 0;
while ~feof(fid)
    line = fgetl(fid);
    line_num = line_num + 1;
    if startsWith(line, 'coords')
        header_row = line_num;
        break;
    end
end
fclose(fid);

if header_row == 0
    error('Could not find "Coords" header row in DeepLabCut CSV file');
end

fprintf('Found header row at line %d\n', header_row);

% Read the CSV file starting from the header row
data = readtable(data_path, 'HeaderLines', header_row - 1);
currdf = table2array(data);

% Find column indices for likelihood, x, and y
headers = data.Properties.VariableNames;
lIndex = [];
xIndex = [];
yIndex = [];

fprintf('Searching for coordinate columns in %d headers...\n', length(headers));

for i = 1:length(headers)
    header_lower = lower(headers{i});
    if contains(header_lower, 'likelihood')
        lIndex = [lIndex, i];
        fprintf('Found likelihood column %d: %s\n', i, headers{i});
    elseif contains(header_lower, 'x') && ~contains(header_lower, 'likelihood')
        xIndex = [xIndex, i];
        fprintf('Found x column %d: %s\n', i, headers{i});
    elseif contains(header_lower, 'y') && ~contains(header_lower, 'likelihood')
        yIndex = [yIndex, i];
        fprintf('Found y column %d: %s\n', i, headers{i});
    end
end

fprintf('Found %d likelihood, %d x, and %d y columns\n', length(lIndex), length(xIndex), length(yIndex));

% Validate that we found the expected columns
if isempty(xIndex) || isempty(yIndex) || isempty(lIndex)
    error('Could not find required coordinate columns (x, y, likelihood) in DeepLabCut CSV file');
end

if length(xIndex) ~= length(yIndex) || length(xIndex) ~= length(lIndex)
    error('Mismatch in number of x (%d), y (%d), and likelihood (%d) columns', length(xIndex), length(yIndex), length(lIndex));
end

% Extract data
datax = currdf(:, xIndex);
datay = currdf(:, yIndex);
data_lh = currdf(:, lIndex);

fprintf('Data matrix size: %s\n', mat2str(size(currdf)));
fprintf('Extracted data sizes - X: %s, Y: %s, Likelihood: %s\n', mat2str(size(datax)), mat2str(size(datay)), mat2str(size(data_lh)));

% Initialize output
currdf_filt = zeros(size(datax, 1), length(xIndex) * 2);
perc_rect = zeros(1, length(xIndex));

% % VECTORIZED VERSION - Much faster processing
% fprintf('Processing %d tracked points (vectorized)...\n', length(xIndex));
% 
% % Pre-allocate arrays for thresholds and percentages
% thresholds = zeros(length(xIndex), 1);
% perc_rect = zeros(length(xIndex), 1);
% 
% % Calculate thresholds for all points at once
% for x = 1:length(xIndex)
%     lh_data = data_lh(:, x);
%     [counts, edges] = histcounts(lh_data, 50);
% 
%     % Find first significant rise in histogram
%     diff_counts = diff(counts);
%     rise_indices = find(diff_counts >= 0);
% 
%     if ~isempty(rise_indices)
%         if rise_indices(1) > 1
%             thresholds(x) = edges(rise_indices(1));
%         else
%             thresholds(x) = edges(rise_indices(2));
%         end
%     else
%         thresholds(x) = median(lh_data); % Fallback threshold
%     end
% 
%     % Calculate percentage below threshold
%     perc_rect(x) = sum(lh_data < thresholds(x)) / length(lh_data);
% end
% 
% % VECTORIZED ADAPTIVE FILTERING - Much faster than nested loops
% % Initialize with first frame data
% currdf_filt(1, :) = [datax(1, :), datay(1, :)];
% 
% % Create logical masks for low likelihood frames
% low_likelihood_mask = data_lh < thresholds';
% 
% % Apply vectorized filtering
% for i = 2:size(data_lh, 1)
%     % Find which points have low likelihood at current frame
%     lowLhPoints = find(low_likelihood_mask(i, :)); % Indices of low likelihood points
% 
%     % For low likelihood points, use previous frame's position
%     currdf_filt(i, :) = currdf_filt(i-1, :);
% 
%     % For high likelihood points, use current frame's position
%     % X coordinates
%     currdf_filt(i, lowLhPoints) = datax(i, lowLhPoints);
%     % Y coordinates (offset by number of points)
%     currdf_filt(i, lowLhPoints + length(xIndex)) = datay(i, lowLhPoints);
% end

% ORIGINAL LOOP-BASED VERSION (commented out for reference)
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

% VECTORIZED VERSION - Much faster computation
% fprintf('Computing displacement and distance features (vectorized)...\n');
% 
% % Pre-allocate arrays for better performance
% disp_list = zeros(data_n_len-1, num_points);
% dxy_list = zeros(data_n_len, num_points * (num_points - 1) / 2 * 2);
% 
% % VECTORIZED DISPLACEMENT CALCULATION
% % Calculate displacement between consecutive frames using vectorized operations
% for c = 1:num_points
%     col_idx = (c-1)*2 + 1;
%     % Calculate displacement for this point across all frames
%     disp_vec = sqrt(sum((data(2:end, col_idx:col_idx+1) - data(1:end-1, col_idx:col_idx+1)).^2, 2));
%     disp_list(:, c) = disp_vec;
% end
% 
% % VECTORIZED DISTANCE CALCULATION
% % Calculate distances between all point pairs using vectorized operations
% pair_idx = 1;
% for i = 1:num_points
%     for j = (i+1):num_points
%         col_i = (i-1)*2 + 1;
%         col_j = (j-1)*2 + 1;
% 
%         % Calculate vector differences for all frames at once
%         diff_vec = data(:, col_i:col_i+1) - data(:, col_j:col_j+1);
%         dxy_list(:, (pair_idx-1)*2+1:pair_idx*2) = diff_vec;
%         pair_idx = pair_idx + 1;
%     end
% end

% ORIGINAL LOOP-BASED VERSION (commented out for reference)
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

% % VECTORIZED INTERPOLATION AND SMOOTHING
% interp_times = 1:data_n_len;
% computed_times = 1.5:(data_n_len-0.5);
% 
% % Vectorized interpolation for all displacement columns at once
% disp_r = interp1(computed_times, disp_list, interp_times, 'linear', 'extrap');
% 
% fprintf('Displacement shape: %s, Distance shape: %s\n', mat2str(size(disp_r)), mat2str(size(dxy_list)));
% 
% % Pre-allocate arrays for better performance
% num_pairs = size(dxy_list, 2) / 2;
% dxy_eu = zeros(data_n_len, num_pairs);
% ang = zeros(data_n_len, num_pairs);
% 
% fprintf('Computing smoothed features (vectorized)...\n');
% 
% % VECTORIZED EUCLIDEAN DISTANCE AND ANGLE CALCULATION
% for k = 1:num_pairs
%     % Extract x,y components for this pair
%     pair_data = dxy_list(:, (2*k-1):(2*k));
% 
%     % Vectorized Euclidean distance calculation
%     dxy_eu(:, k) = sqrt(sum(pair_data.^2, 2));
% 
%     % Vectorized angle calculation between consecutive frames
%     if data_n_len > 1
%         v1 = pair_data(1:end-1, :);
%         v2 = pair_data(2:end, :);
% 
%         % Compute norms for all frames at once
%         norm_v1 = sqrt(sum(v1.^2, 2));
%         norm_v2 = sqrt(sum(v2.^2, 2));
% 
%         % Avoid division by zero
%         valid_frames = (norm_v1 > 0) & (norm_v2 > 0);
% 
%         % Compute cosine of angles for valid frames
%         cos_angles = zeros(size(v1, 1), 1);
%         cos_angles(valid_frames) = sum(v1(valid_frames, :) .* v2(valid_frames, :), 2) ./ ...
%             (norm_v1(valid_frames) .* norm_v2(valid_frames));
% 
%         % Clamp to [-1, 1] and convert to degrees
%         cos_angles = max(-1, min(1, cos_angles));
%         ang(1:end-1, k) = acos(cos_angles) * 180 / pi;
%     end
% end
% 
% % VECTORIZED SMOOTHING - Apply boxcar smoothing to all columns at once
% disp_boxcar = zeros(data_n_len, size(disp_r, 2));
% dxy_boxcar = zeros(data_n_len, num_pairs);
% ang_boxcar = zeros(data_n_len, num_pairs);
% 
% for i = 1:size(disp_r, 2)
%     disp_boxcar(:, i) = boxcar_center(disp_r(:, i), window);
% end
% 
% for i = 1:num_pairs
%     dxy_boxcar(:, i) = boxcar_center(dxy_eu(:, i), window);
%     ang_boxcar(:, i) = boxcar_center(ang(:, i), window);
% end

% ORIGINAL LOOP-BASED VERSION (commented out for reference)
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

% Organize features to match Python ordering
% Python: np.vstack((dxy_feat, ang_feat, disp_feat.T))
% where dxy_feat, ang_feat are (features x time) and disp_feat.T transposes disp_feat

% Assign to match Python variable names for clarity
disp_feat = disp_boxcar;  % (time x features)
dxy_feat = dxy_boxcar;    % (time x features) 
ang_feat = ang_boxcar;    % (time x features)

% Stack features vertically to match Python's np.vstack
features = [dxy_feat; ang_feat; disp_feat]; % Stack as (features x time)

fprintf('Feature shapes - Displacement: %s, Distance: %s, Angle: %s\n', ...
    mat2str(size(disp_feat)), mat2str(size(dxy_feat)), mat2str(size(ang_feat)));
fprintf('Final features shape: %s (should match Python)\n', mat2str(size(features)));

% Z-score normalization
scaled_features = zscore(features, 0, 1); % Normalize across time (rows)

end

% None of the features in @compute_kinematics.m match the output in @ComputeKinematics.py. Analyzed the files to see if you can spot any discrpencies. They were both produces using the same input, so it's something in the code.