function validate_workspace_vars(requiredVars, dataStruct, varargin)
% VALIDATE_WORKSPACE_VARS Validate required workspace variables or struct fields
%
% Variables:
%   requiredVars - Cell array of variable or field names
%   dataStruct - Structure to check (empty to check workspace)
%   varargin - 'errorMsg', 'source' (loader or script name for error text)
%
% Goal:
%   Ensure required fields exist on dataStruct before running session analyses.

    p = inputParser;
    addParameter(p, 'errorMsg', 'Required variable', @ischar);
    addParameter(p, 'source', '', @ischar);
    parse(p, varargin{:});

    errorMsg = p.Results.errorMsg;
    source = p.Results.source;

    checkWorkspace = (nargin < 2 || isempty(dataStruct));

    missingVars = {};
    for i = 1:length(requiredVars)
        varName = requiredVars{i};

        if checkWorkspace
            if ~exist(varName, 'var')
                missingVars{end + 1} = varName; %#ok<AGROW>
            end
        else
            if ~isfield(dataStruct, varName)
                missingVars{end + 1} = varName; %#ok<AGROW>
            end
        end
    end

    if isempty(missingVars)
        return;
    end

    missingStr = strjoin(missingVars, ', ');
    if checkWorkspace
        if ~isempty(source)
            error('%s(s) %s not found in workspace. Please run %s first.', ...
                errorMsg, missingStr, source);
        else
            error('%s(s) %s not found in workspace.', errorMsg, missingStr);
        end
    else
        if ~isempty(source)
            error('%s(s) %s not found in data structure. Please ensure %s provides these fields.', ...
                errorMsg, missingStr, source);
        else
            error('%s field(s) %s not found in data structure.', errorMsg, missingStr);
        end
    end
end
