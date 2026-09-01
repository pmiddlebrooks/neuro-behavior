function plotBase = prepend_manuscript_plot_file_tag(plotBase, plotConfig)
% PREPEND_MANUSCRIPT_PLOT_FILE_TAG Optional stem prefix from plotConfig.fileTag
%
% Variables:
%   plotBase   - Filename stem without extension
%   plotConfig - May contain .fileTag (e.g. subject_task_d2method)
%
% Goal:
%   Keep spontaneous-vs-task figures from overwriting across-task plots.

if nargin < 2 || isempty(plotConfig) || ~isstruct(plotConfig)
  return;
end
if ~isfield(plotConfig, 'fileTag') || isempty(plotConfig.fileTag)
  return;
end
plotBase = [char(plotConfig.fileTag), '_', char(plotBase)];
end
