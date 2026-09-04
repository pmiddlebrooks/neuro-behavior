function tbl = read_tsv_table(filePath)
% READ_TSV_TABLE - Load a tab-separated file as a table
%
% Variables:
%   filePath - Full path to a .tsv (or other tab-delimited) file
%
% Goal:
%   Wrapper for read_delimited_table with a tab delimiter.

    tbl = read_delimited_table(filePath, sprintf('\t'));
end
