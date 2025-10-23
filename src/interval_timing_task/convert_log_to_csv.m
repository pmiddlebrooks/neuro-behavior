% One-time conversion script
% function convert_log_to_csv(logFile, csvFile)
function convert_log_to_csv()

%%
    % File paths
    dataPath = 'E:\Dropbox\Data\interval_timing_task\data\mouse99';
    resultsPath = 'E:\Dropbox\Data\interval_timing_task\results\mouse99';
    logFile = 'log.txt';
    csvFile = 'log.csv';

    % Full file paths
    logFilePath = fullfile(dataPath, logFile);
    csvFilePath = fullfile(dataPath, csvFile);
    
    % Check if log file exists
    if ~exist(logFilePath, 'file')
        error('Log file not found: %s', logFilePath);
    end
    
    fprintf('Converting %s to CSV...\n', logFilePath);
    
    fid = fopen(logFilePath, 'r');
    if fid == -1
        error('Could not open log file: %s', logFilePath);
    end
    
    data = textscan(fid, '%f %s %s', 'Delimiter', ' ');
    fclose(fid);
    
    % Create table and save as CSV
    T = table(data{1}, data{2}, data{3}, ...
             'VariableNames', {'timestamp', 'type', 'value'});
    
    % Create results directory if it doesn't exist
    if ~exist(resultsPath, 'dir')
        mkdir(resultsPath);
    end
    
    writetable(T, csvFilePath);
    fprintf('CSV file saved to: %s\n', csvFilePath);
end