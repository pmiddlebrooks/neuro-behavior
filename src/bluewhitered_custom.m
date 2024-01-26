function customColormap = bluewhitered_custom(range)
% Sets color map from red (positive) to blue (negative), symmetric around
% zero
%
% Input: 
%   range: [x y], where x is lower bound and y is upper bound of values.
%   Needs to be symmetric around zero to plot zero as white, red as
%   positive, and blue as negative

if nargin == 0
    range = [-1 1];
end
% Number of rows in the colormap
nRows = 256;


bwr = @(n)interp1([1 2 3], [0 0 1; 1 1 1; 1 0  0], linspace(1, 3, nRows), 'linear');
customColormap = bwr(256);
set(gca, 'clim', range)