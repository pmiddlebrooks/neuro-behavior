function tbl = read_delimited_table(filePath, delimiter)
% READ_DELIMITED_TABLE - Load a delimited text file as a table
%
% Variables:
%   filePath  - Full path to the file
%   delimiter - Field delimiter (default tab). Use ',' for CSV.
%
% Goal:
%   Avoid readtable/detectImportOptions, which throw Unrecognized field
%   name "text" on this MATLAB for text/CSV/TSV imports. Parse every data
%   line (textscan with WhiteSpace '' only reads the first row). Infer
%   numeric vs text from all rows so an empty first-row field does not
%   mis-type a later text column.

    if nargin < 1 || isempty(filePath)
        error('read_delimited_table:MissingFile', 'filePath is required.');
    end
    if nargin < 2 || isempty(delimiter)
        delimiter = sprintf('\t');
    end
    if ~isfile(filePath)
        error('read_delimited_table:NotFound', 'File not found: %s', filePath);
    end

    rawText = fileread(filePath);
    if ~isempty(rawText) && double(rawText(1)) == 65279
        rawText = rawText(2:end);
    end
    lines = regexp(rawText, '\r\n|\n|\r', 'split');
    lines = lines(:);
    if ~isempty(lines) && isempty(lines{end})
        lines(end) = [];
    end
    if isempty(lines)
        error('read_delimited_table:EmptyFile', 'No header row in %s', filePath);
    end

    headerLine = lines{1};
    rawNames = strsplit(headerLine, delimiter, 'CollapseDelimiters', false);
    varNames = matlab.lang.makeValidName(rawNames);
    varNames = matlab.lang.makeUniqueStrings(varNames, {}, namelengthmax);
    nCols = numel(varNames);

    dataLines = lines(2:end);
    nRows = numel(dataLines);
    if nRows == 0
        tbl = table('Size', [0, nCols], 'VariableTypes', repmat({'double'}, 1, nCols), ...
            'VariableNames', varNames);
        return
    end

    cells = repmat({''}, nRows, nCols);
    for iRow = 1:nRows
        parts = strsplit(dataLines{iRow}, delimiter, 'CollapseDelimiters', false);
        nTake = min(nCols, numel(parts));
        cells(iRow, 1:nTake) = parts(1:nTake);
    end

    tableCols = cell(1, nCols);
    for iCol = 1:nCols
        colVals = strtrim(cells(:, iCol));
        if column_is_numeric(colVals)
            tableCols{iCol} = str2double(colVals);
        else
            tableCols{iCol} = colVals;
        end
    end
    tbl = table(tableCols{:}, 'VariableNames', varNames);
end

function tf = column_is_numeric(colVals)
% COLUMN_IS_NUMERIC - True if every nonempty token parses as a number

    emptyMask = cellfun(@isempty, colVals);
    nums = str2double(colVals);
    tf = all(emptyMask | ~isnan(nums));
end
