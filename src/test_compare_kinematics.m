function test_compare_kinematics(data_path)
% TEST_COMPARE_KINEMATICS Compare MATLAB and Python kinematics on first 20 data rows
%
% Inputs:
%   data_path - Path to DeepLabCut CSV file with multi-row headers
%
% Steps:
% 1) Detect header row (line starting with 'coords')
% 2) Create a temporary CSV with only the first 20 data rows after header
% 3) Run MATLAB compute_kinematics on the sliced CSV
% 4) Run Python ComputeKinematics.py on the same sliced CSV
% 5) Load Python outputs and compare with MATLAB outputs

if nargin < 1 || ~isfile(data_path)
    error('Provide a valid CSV path.');
end

fprintf('=== Kinematics comparison on first 20 data rows ===\n');

% 1) Find header row
fid = fopen(data_path, 'r');
if fid == -1, error('Could not open file: %s', data_path); end
header_row = 0; line_num = 0; first_line = '';
while ~feof(fid)
    line = fgetl(fid); line_num = line_num + 1;
    if ischar(line) && startsWith(lower(strtrim(line)), 'coords')
        header_row = line_num; first_line = line; break;
    end
end
fclose(fid);
if header_row == 0, error('Could not find "coords" header row.'); end
fprintf('Header row at line %d\n', header_row);

% 2) Write sliced CSV: include header lines up to header_row and next 20 data rows
[temp_dir,~,~] = fileparts(tempname);
if ~exist(temp_dir,'dir'), mkdir(temp_dir); end
sliced_csv = fullfile(temp_dir, 'dlc_first20.csv');

inF = fopen(data_path,'r'); outF = fopen(sliced_csv,'w');
% Copy header rows
frewind(inF);
for i = 1:header_row
    l = fgetl(inF);
    fprintf(outF, '%s\n', l);
end
% Copy first 20 data rows
rows_written = 0;
while ~feof(inF) && rows_written < 20
    l = fgetl(inF);
    if ischar(l)
        fprintf(outF, '%s\n', l);
        rows_written = rows_written + 1;
    end
end
fclose(inF); fclose(outF);
fprintf('Wrote sliced CSV to: %s (20 data rows)\n', sliced_csv);

% 3) Run MATLAB pipeline
fprintf('\n[MATLAB] Running compute_kinematics...\n');
[mat_features, mat_scaled] = compute_kinematics(sliced_csv, '', 60);
fprintf('[MATLAB] Features: %s, Scaled: %s\n', mat2str(size(mat_features)), mat2str(size(mat_scaled)));

% 4) Instructions for running Python separately
fprintf('\n[Python] To run Python comparison separately:\n');
fprintf('python "%s" "%s" "%s"\n', ...
    fullfile(fileparts(mfilename('fullpath')), 'test_compare_kinematics.py'), ...
    sliced_csv, temp_dir);
fprintf('\nAfter running the Python script, press any key to continue with comparison...\n');
pause;

% 5) Load Python outputs
py_feat_path = fullfile(temp_dir, 'python_features.npy');
py_scaled_path = fullfile(temp_dir, 'python_scaled_features.npy');
if ~isfile(py_feat_path) || ~isfile(py_scaled_path)
    error('Python outputs not found in %s. Make sure you ran the Python script first.', temp_dir);
end
py_features = readNPY(py_feat_path)';
py_scaled = readNPY(py_scaled_path)';
fprintf('[Python] Features: %s, Scaled: %s\n', mat2str(size(py_features)), mat2str(size(py_scaled)));

% Compare
fprintf('\n=== Comparison ===\n');
compare_arrays('features', mat_features, py_features);
compare_arrays('scaled', mat_scaled, py_scaled);

fprintf('\nDone.\n');
end

function compare_arrays(name, A, B)
fprintf('%s: ', name);
if isequal(size(A), size(B))
    d = abs(A - B);
    fprintf('shape match %s | max=%.6g mean=%.6g\n', mat2str(size(A)), max(d(:)), mean(d(:)));
    % report top-5 diffs
    [~, idx] = maxk(d(:), min(5, numel(d)));
    for k = 1:numel(idx)
        [i,j] = ind2sub(size(A), idx(k));
        fprintf('  [%d,%d] mat=%.6g py=%.6g diff=%.6g\n', i, j, A(i,j), B(i,j), d(i,j));
    end
else
    fprintf('shape mismatch: MATLAB %s vs Python %s\n', mat2str(size(A)), mat2str(size(B)));
end
end
