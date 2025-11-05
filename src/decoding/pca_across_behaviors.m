%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectEnd = 60 * 60; % seconds
opts.frameSize = .1;

getDataType = 'spikes';
get_standard_data

colors = colors_for_behaviors(codes);

[dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);


% for plotting consistency
%
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one



%
bhvLabels = {'investigate_1', 'investigate_2', 'investigate_3', ...
    'rear', 'dive_scrunch', 'paw_groom', 'face_groom_1', 'face_groom_2', ...
    'head_groom', 'contra_body_groom', 'ipsi_body groom', 'contra_itch', ...
    'ipsi_itch_1', 'contra_orient', 'ipsi_orient', 'locomotion'};


%%
selectFrom = 'M56';
idSelect = idM56;
selectFrom = 'DS';
idSelect = idDS;
% selectFrom = 'VS';
% idSelect = idVS;







%% Explained variance per behavior
    fig = figure(23); clf
    set(fig, 'Position', monitorTwo);
    [ax, pos] = tight_subplot(2, 8, [.08 .02], .1);

    for i = 1 : length(analyzeBhv)
    axes(ax(i)) 

    idx = bhvID == analyzeCodes(i);
    % [coeff, score, ~, ~, explained] = pca(dataMat(idx, idSelect));
    [coeff, score, ~, ~, explained] = pca(zscore(dataMat(idx, idSelect)));

plot(cumsum(explained));
expD = find(cumsum(explained) > 75, 1);
xline(expD)
ylim([0 100]);
title([analyzeBhv{i}, ' ', num2str(expD)], 'interpreter', 'none')
% axc = gca;
% set(gca, 'Color', 'none');
if(i == 1)
    ylabel('Explained Variance')
    xlabel('n Components')
end


    end
    % sgtitle([selectFrom, ' PCA across behaviors'])
    sgtitle([selectFrom, ' PCA across behaviors (Z-scored)'])
    % Set figure background to transparent
% set(fig, 'Color', 'none');
copy_figure_to_clipboard



%%
n_components = 4;
[angleMatrix, overlapMatrix, grassmanMatrix] = analyze_behavior_pca(dataMat(:, idSelect), bhvID, n_components);
 
% Display the confusion matrices
    figure(1);
    imagesc(angleMatrix);
    colormap hot;
    colorbar;
    title([selectFrom, ' Eigenvector Angles (Degrees)']);
    xlabel('Behavior');
    ylabel('Behavior');
    xticks(1:length(analyzeCodes));
    yticks(1:length(analyzeCodes));
    xticklabels(analyzeCodes);
    yticklabels(analyzeCodes);
    axis square;

    figure(2);
    imagesc(overlapMatrix);
    colormap cool;
    colorbar;
    title([selectFrom, ' Subspace Overlap']);
    xlabel('Behavior');
    ylabel('Behavior');
    xticks(1:length(analyzeCodes));
    yticks(1:length(analyzeCodes));
    xticklabels(analyzeCodes);
    yticklabels(analyzeCodes);
    axis square;

    % Display the confusion matrix
    figure(3);
    imagesc(grassmanMatrix);
    colormap(parula);
    colorbar;
    title([selectFrom, ' Grassmann Distances']);
    xlabel('Behavior');
    ylabel('Behavior');
    xticks(1:length(analyzeCodes));
    yticks(1:length(analyzeCodes));
    xticklabels(analyzeCodes);
    yticklabels(analyzeCodes);
    axis square;


    function [angleMatrix, overlapMatrix, grassmanMatrix] = analyze_behavior_pca(dataMat, bhvID, n_components)
    % Performs PCA for each behavior and computes eigenvector angles & subspace overlaps.
    % 
    % INPUTS:
    % - dataMat: Neural data matrix (time bins × neurons)
    % - bhvID: Vector of behavior labels corresponding to each time bin
    % - n_components: Number of top eigenvectors to compare
    %
    % OUTPUT:
    % - results_table: MATLAB table containing behavior pairs, angles, and subspace overlaps

    unique_behaviors = unique(bhvID);
    unique_behaviors(unique_behaviors == -1) = [];
    num_behaviors = length(unique_behaviors);
    pca_dict = struct();

% Perform PCA for each behavior
    for i = 1:num_behaviors
        bhv = unique_behaviors(i);
        bhv_indices = bhvID == bhv;
        bhv_data = dataMat(bhv_indices, :);

        % Compute PCA using singular value decomposition
        [coeff, ~, ~] = pca(bhv_data);
        
        % Store the top principal components
        pca_dict.(sprintf('B%d', bhv)) = coeff(:, 1:n_components);
    end

    % Initialize confusion matrices
    angleMatrix = zeros(num_behaviors, num_behaviors);
    overlapMatrix = zeros(num_behaviors, num_behaviors);
    grassmanMatrix = zeros(num_behaviors, num_behaviors);

    % Compare eigenvector angles and subspace overlaps
    for i = 1:num_behaviors
        for j = 1:num_behaviors
            if i == j
                angleMatrix(i, j) = 0;  % Diagonal = 0 degrees (self-comparison)
                overlapMatrix(i, j) = 1; % Self-overlap should be 1
                grassmanMatrix(i, j) = 0;  % Self-comparison = 0 distance
            else
                bhv1 = unique_behaviors(i);
                bhv2 = unique_behaviors(j);

                V1 = pca_dict.(sprintf('B%d', bhv1));
                V2 = pca_dict.(sprintf('B%d', bhv2));

                % Compute the angle between the first principal components
                angleMatrix(i, j) = compute_angle(V1(:, 1), V2(:, 1));

                % Compute the subspace overlap index
                overlapMatrix(i, j) = subspace_overlap(V1, V2, n_components);

                            % Compute the Grassmann distance
                grassmanMatrix(i, j) = compute_grassmann_distance(V1, V2, n_components);
end
        end
    end
end



function angle = compute_angle(v1, v2)
    % Computes the cosine similarity angle between two vectors
    cos_theta = dot(v1, v2) / (norm(v1) * norm(v2));
    cos_theta = max(min(cos_theta, 1), -1); % Ensure within valid range
    angle = acosd(cos_theta); % Convert to degrees
end

function overlap = subspace_overlap(V1, V2, k)
    % Computes the subspace overlap index between two sets of top-k eigenvectors
    % V1, V2: Matrices where columns are top-k eigenvectors of two behaviors
    % k: Number of eigenvectors to consider

    overlap_matrix = V1(:, 1:k)' * V2(:, 1:k); % Compute pairwise dot products
    overlap = sum(overlap_matrix(:).^2) / k; % Average squared cosine similarity
end

function d_G = compute_grassmann_distance(V1, V2, k)
    % Computes Grassmann distance between two subspaces spanned by top-k eigenvectors
    [~, S, ~] = svd(V1(:, 1:k)' * V2(:, 1:k)); % SVD to compute principal angles
    theta = acos(diag(S)); % Extract principal angles
    d_G = norm(theta, 2); % Compute Grassmann distance using L2 norm (Frobenius norm)
end