

% Assuming dataM56 and dataDS are cell arrays where each cell contains a session dataset
% Each session in dataM56{i} and dataDS{i} should have the same number of samples (rows)

% Initialize correlation coefficients storage
correlationCoefficients = zeros(1, length(dataM56));
% Initialize angle storage for each session
anglesInDegrees = zeros(1, length(dataM56));

% Loop over each session
for i = 1:length(dataM56)
    % Extract the data for the current session
    M56 = dataM56{i};  % Dataset 1
    DS = dataDS{i};    % Dataset 2
    
    % Perform canonical correlation analysis
    [A, B, r, U, V] = canoncorr(M56, DS);
    
    % Project both datasets into the first canonical variate (CV) subspace
    projM56 = U(:, 1);  % First CV for dataM56
    projDS = V(:, 1);   % First CV for dataDS
    
    % Calculate the correlation coefficient between the two projections
    correlationCoefficients(i) = corr(projM56, projDS);

        % Calculate the cosine of the angle between the first CVs
    cosTheta = dot(firstCV_M56, firstCV_DS) / (norm(firstCV_M56) * norm(firstCV_DS));
    
    % Calculate the angle in degrees
    anglesInDegrees(i) = acosd(cosTheta);

end

% Display correlation coefficients for each session
disp('Correlation coefficients between projected datasets in the CV subspace:');
disp(correlationCoefficients);
