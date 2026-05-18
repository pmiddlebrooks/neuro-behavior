function add_figure_tools_path()
% ADD_FIGURE_TOOLS_PATH - Add external figure_tools (tight_subplot, etc.) to path
%
% Goal:
%   Prefer E:/Projects/figure_tools/tight_subplot.m over any repo-local copy.

paths = get_paths;
if ~isfield(paths, 'figureToolsPath') || isempty(paths.figureToolsPath)
    warning('figureToolsPath is not defined in get_paths.');
    return;
end
if ~exist(paths.figureToolsPath, 'dir')
    warning('figure_tools directory not found: %s', paths.figureToolsPath);
    return;
end
addpath(paths.figureToolsPath);

end
