function ha = tight_subplot(nRows, nCols, gaps)
% TIGHT_SUBPLOT Minimal-spacing axes tiling (normalized figure units).
%
% Variables:
%   nRows, nCols - grid size
%   gaps         - [gapV gapH] vertical and horizontal gaps between neighbouring axes;
%                   if scalar, reused for both
%
% Goal:
%   Layout axes with small margins suitable for stacking many panels under a sgtitle.
%   Indices are column-major: ha(1) top of column 1, ha(2) below it, ...
%
% Returns:
%   ha - axes handles (column vector, length nRows * nCols)

    if nargin < 3 || isempty(gaps)
        gapV = 0.04;
        gapH = 0.04;
    elseif isscalar(gaps)
        gapV = double(gaps);
        gapH = gapV;
    else
        gapV = gaps(1);
        gapH = gaps(2);
    end

    marTop = 0.065;
    marBot = 0.10;

    usableH = 1 - marTop - marBot;
    axH = (usableH - gapV * (nRows - 1)) / nRows;
    axW = (1 - gapH * (nCols - 1)) / nCols;

    ha = gobjects(nRows * nCols, 1);
    k = 0;
    for col = 1:nCols
        left = (col - 1) * (axW + gapH);
        for row = 1:nRows
            k = k + 1;
            bottom = marBot + (nRows - row) * (axH + gapV);
            ha(k) = axes('Position', [left, bottom, axW, axH], 'Box', 'off');
        end
    end
end
