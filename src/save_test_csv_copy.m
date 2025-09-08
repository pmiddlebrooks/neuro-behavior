function outPath = save_test_csv_copy(dataPath, saveDir, n)
% SAVE_TEST_CSV_COPY Write a CSV containing only the first n data rows
%
% outPath = save_test_csv_copy(dataPath, saveDir, n)
% - Detects multi-row DeepLabCut headers (line starting with 'coords')
% - Writes all header lines up to and including the 'coords' line
% - Then writes the first n data rows
% - If no 'coords' header found, writes the first line as header then n rows
% - Appends '_test' to the original filename and saves in saveDir
%
% Inputs:
%   dataPath (char|string): path to original CSV
%   saveDir  (char|string): directory to save the test copy
%   n        (double): number of data rows to keep
%
% Output:
%   outPath (char): path to the written test CSV

if nargin < 3 || isempty(n) || n <= 0
	error('Provide dataPath, saveDir, and a positive n.');
end

if ~isfile(dataPath)
	error('File not found: %s', dataPath);
end

if ~exist(saveDir, 'dir')
	mkdir(saveDir);
end

[~, baseName, ext] = fileparts(dataPath);
if isempty(ext)
	ext = '.csv';
end
outFile = sprintf('%s_test%s', baseName, ext);
outPath = fullfile(saveDir, outFile);

fidIn = fopen(dataPath, 'r');
if fidIn == -1
	error('Could not open input file: %s', dataPath);
end
cleaner = onCleanup(@() fclose(fidIn));

fidOut = fopen(outPath, 'w');
if fidOut == -1
	error('Could not open output file: %s', outPath);
end
cleaner2 = onCleanup(@() fclose(fidOut));

% Detect multi-row header (look for line that starts with 'coords')
headerRow = 0;
lineNum = 0;
fileLines = {};
while true
	line = fgetl(fidIn);
	if ~ischar(line)
		break;
	end
	lineNum = lineNum + 1;
	fileLines{lineNum,1} = line; %#ok<AGROW>
	if headerRow == 0 && startsWith(lower(strtrim(line)), 'coords')
		headerRow = lineNum;
	end
end

% Decide header strategy
if headerRow > 0
	% Write all header lines up to headerRow
	for i = 1:headerRow
		fprintf(fidOut, '%s\n', fileLines{i});
	end
	% Write first n data rows after headerRow
	rowsWritten = 0;
	for i = headerRow+1:min(headerRow+n, numel(fileLines))
		fprintf(fidOut, '%s\n', fileLines{i});
		rowsWritten = rowsWritten + 1;
	end
	fprintf('Wrote %d data rows (after header) to %s\n', rowsWritten, outPath);
else
	% Fallback: assume first line is header, copy it + first n rows
	if ~isempty(fileLines)
		fprintf(fidOut, '%s\n', fileLines{1});
	end
	rowsWritten = 0;
	for i = 2:min(1+n, numel(fileLines))
		fprintf(fidOut, '%s\n', fileLines{i});
		rowsWritten = rowsWritten + 1;
	end
	fprintf('No coords header found. Wrote header + %d data rows to %s\n', rowsWritten, outPath);
end

end
