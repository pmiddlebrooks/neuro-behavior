%% Script to rename existing SVM decoding result files to include kernel type
% This script adds "_polynomial" to existing result files that don't have kernel type in filename

% Get the save path (actual location of files)
savePath = '/Users/paulmiddlebrooks/Library/CloudStorage/Dropbox/Data/decoding';

if ~exist(savePath, 'dir')
    fprintf('Save directory does not exist: %s\n', savePath);
    return;
end

fprintf('Looking for files to rename in: %s\n', savePath);

% Get all files in the directory
files = dir(fullfile(savePath, '*.mat'));
filesRenamed = 0;

for i = 1:length(files)
    oldName = files(i).name;
    
    % Check if filename already contains kernel type
    if contains(oldName, 'polynomial_') || contains(oldName, 'linear_')
        fprintf('Skipping %s (already has kernel type)\n', oldName);
        continue;
    end
    
    % Check if it's an SVM decoding result file
    if contains(oldName, 'svm_decoding_compare') && contains(oldName, '.mat')
        % Create new filename with polynomial_ prefix
        newName = strrep(oldName, 'svm_decoding_compare', 'polynomial_svm_decoding_compare');
        
        oldPath = fullfile(savePath, oldName);
        newPath = fullfile(savePath, newName);
        
        % Rename the file
        try
            movefile(oldPath, newPath);
            fprintf('Renamed: %s -> %s\n', oldName, newName);
            filesRenamed = filesRenamed + 1;
        catch ME
            fprintf('Error renaming %s: %s\n', oldName, ME.message);
        end
    end
end

% Also rename summary files
summaryFiles = dir(fullfile(savePath, '*_summary.txt'));
for i = 1:length(summaryFiles)
    oldName = summaryFiles(i).name;
    
    % Check if filename already contains kernel type
    if contains(oldName, 'polynomial_') || contains(oldName, 'linear_')
        fprintf('Skipping %s (already has kernel type)\n', oldName);
        continue;
    end
    
    % Check if it's an SVM decoding summary file
    if contains(oldName, 'svm_decoding_compare') && contains(oldName, '_summary.txt')
        % Create new filename with polynomial_ prefix
        newName = strrep(oldName, 'svm_decoding_compare', 'polynomial_svm_decoding_compare');
        
        oldPath = fullfile(savePath, oldName);
        newPath = fullfile(savePath, newName);
        
        % Rename the file
        try
            movefile(oldPath, newPath);
            fprintf('Renamed: %s -> %s\n', oldName, newName);
            filesRenamed = filesRenamed + 1;
        catch ME
            fprintf('Error renaming %s: %s\n', oldName, ME.message);
        end
    end
end

fprintf('\nRenaming complete. %d files renamed.\n', filesRenamed);
