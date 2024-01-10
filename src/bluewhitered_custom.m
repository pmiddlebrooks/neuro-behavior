function customColormap = bluewhitered_custom

% Number of rows in the colormap
nRows = 256;

% Preallocate the colormap matrix
customColormap = zeros(nRows, 3);

% Transition from blue to white
for i = 1:(nRows/2)
    customColormap(i, :) = [i-1, i-1, 255] / 255;
end

% Transition from white to red
for i = (nRows/2 + 1):nRows
    customColormap(i, :) = [255, 256-i, 256-i] / 255;
end

% % Apply the colormap to the current figure
% colormap(customColormap);
% 
% % Display the colormap for visualization
% imagesc(reshape(customColormap, [1, nRows, 3]));
axis off;