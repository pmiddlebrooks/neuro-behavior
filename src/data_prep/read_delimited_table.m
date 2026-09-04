function tbl = read_delimited_table(filePath, delimiter)
% READ_DELIMITED_TABLE - Load a delimited text file as a table
%
% Variables:
%   filePath  - Full path to the file
%   delimiter - Field delimiter (default tab). Use ',' for CSV.
%
% Goal:
%   Avoid readtable/detectImportOptions, which throw Unrecognized field
%   name "text" on this MATLAB for text/CSV/TSV imports. Infer numeric vs
%   text from the first data row and scan with a mixed format.

    if nargin < 1 || isempty(filePath)
        error('read_delimited_table:MissingFile', 'filePath is required.');
    end
    if nargin < 2 || isempty(delimiter)
        delimiter = sprintf('\t');
    end
    if ~isfile(filePath)
        error('read_delimited_table:NotFound', 'File not found: %s', filePath);
    end

    fid = fopen(filePath, 'rt');
    if fid < 0
        error('read_delimited_table:OpenFailed', 'Could not open %s', filePath);
    end
    closer = onCleanup(@() fclose(fid));

    headerLine = fgetl(fid);
    if ~ischar(headerLine)
        error('read_delimited_table:EmptyFile', 'No header row in %s', filePath);
    end
    if ~isempty(headerLine) && headerLine(1) == char(65279)
        headerLine = headerLine(2:end);
    end

    rawNames = strsplit(headerLine, delimiter, 'CollapseDelimiters', false);
    varNames = matlab.lang.makeValidName(rawNames);
    varNames = matlab.lang.makeUniqueStrings(varNames, {}, namelengthmax);
    nCols = numel(varNames);

    dataStartPos = ftell(fid);
    firstDataLine = fgetl(fid);
    if ~ischar(firstDataLine)
        tbl = table('Size', [0, nCols], 'VariableTypes', repmat({'double'}, 1, nCols), ...
            'VariableNames', varNames);
        return
    end

    isNumericCol = classify_delimited_columns(firstDataLine, delimiter, nCols);
    formatSpec = build_delimited_format(isNumericCol);
    fseek(fid, dataStartPos, 'bof');

    rawCols = textscan(fid, formatSpec, ...
        'Delimiter', delimiter, ...
        'MultipleDelimsAsOne', false, ...
        'WhiteSpace', '', ...
        'EndOfLine', '\n');

    nRows = 0;
    if ~isempty(rawCols)
        nRows = max(cellfun(@numel, rawCols));
    end

    tableCols = cell(1, nCols);
    for iCol = 1:nCols
        colVals = rawCols{iCol};
        if isNumericCol(iCol)
            if numel(colVals) < nRows
                colVals(nRows, 1) = NaN;
            end
            tableCols{iCol} = colVals(:);
        else
            if numel(colVals) < nRows
                colVals(nRows, 1) = {''};
            end
            tableCols{iCol} = colVals(:);
        end
    end
    tbl = table(tableCols{:}, 'VariableNames', varNames);
end

function isNumericCol = classify_delimited_columns(firstDataLine, delimiter, nCols)
% CLASSIFY_DELIMITED_COLUMNS - Numeric columns from the first data row

    parts = strsplit(firstDataLine, delimiter, 'CollapseDelimiters', false);
    isNumericCol = false(1, nCols);
    nCheck = min(nCols, numel(parts));
    for iCol = 1:nCheck
        token = strtrim(parts{iCol});
        isNumericCol(iCol) = isempty(token) || ~isnan(str2double(token));
    end
end

function formatSpec = build_delimited_format(isNumericCol)
% BUILD_DELIMITED_FORMAT - textscan format: %f numeric, %s text

    tokens = repmat({'%s'}, 1, numel(isNumericCol));
    tokens(isNumericCol) = {'%f'};
    formatSpec = [tokens{:}];
end
