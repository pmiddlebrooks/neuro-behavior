%%
data = projSelect;

centroid = zeros(length(codes), 1);
std_distance = zeros(length(codes), 1);

for i = 1 : length(codes)

    iIdx = bhvID == codes(i);
% Step 1: Calculate the centroid (mean of x, y, z)
centroid(i) = mean(data(iIdx), 1);  % 1 means along the columns (each dimension)

% Step 2: Calculate the Euclidean distance of each point from the centroid
distances = sqrt(sum((data(iIdx) - centroid(i)).^2, 2));  % Euclidean distance formula

% Step 3: Calculate the standard deviation of the distances
std_distance(i) = std(distances);


% % Display the result
% disp('Standard deviation of distances from the centroid:');
% disp(std_distance(i));

end