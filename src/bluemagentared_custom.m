function customColormap = bluemagentared_custom(nColors)
% Sets color map from blue (1) to red (nColors), symmetric around magenta
%
% Input: 
%   nColors: how many steps from blue to red

if nargin == 0
    error('input range is required')
end
if nColors == 1
    customColormap = [0 0 1];
    return
end

% Number of rows in the colormap
nRows = nColors;
% range = [1 nColors];

bwr = @(n)interp1([1 2 3], [0 0 1; 1 0 1; 1 0 0], linspace(1, 3, nRows), 'linear');
customColormap = bwr(nColors);
% set(gca, 'clim', range)