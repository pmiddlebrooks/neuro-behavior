function validate_workspace_vars(requiredVars, dataStruct, varargin)
% VALIDATE_WORKSPACE_VARS Validate that required variables exist
%
% Variables:
%   requiredVars - Cell array of variable names to check
%   dataStruct - Structure containing data (if provided, checks structure fields)
%                 If empty or not provided, checks workspace variables
%   varargin - Optional name-value pairs:
%       'errorMsg' - Custom error message prefix (default: 'Required variable')
%       'source' - Source script name for error message (e.g., 'criticality_sliding_data_prep.m')
%
% Goal:
%   Check that all required variables exist either in workspace or as fields
%   in dataStruct. Provides clear error messages indicating which variables
%   are missing and where they should come from.
%
% Returns:
%   None (throws error if validation fails)

    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'errorMsg', 'Required variable', @ischar);
    addParameter(p, 'source', '', @ischar);
    parse(p, varargin{:});
    
    errorMsg = p.Results.errorMsg;
    source = p.Results.source;
    
    % Determine if checking workspace or structure
    checkWorkspace = (nargin < 2 || isempty(dataStruct));
    
    missingVars = {};
    for i = 1:length(requiredVars)
        varName = requiredVars{i};
        
        if checkWorkspace
            % Check workspace
            if ~exist(varName, 'var')
                missingVars{end+1} = varName;
            end
        else
            % Check structure fields
            if ~isfield(dataStruct, varName)
                missingVars{end+1} = varName;
            end
        end
    end
    
    % Report missing variables
    if ~isempty(missingVars)
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
end

