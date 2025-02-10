function behavior_durations_boxplot(bhvID)
    % PLOT_BEHAVIOR_DURATIONS Generates a horizontal box plot of behavior durations.
    % Each behavior is stacked vertically and color-coded.
    %
    % Parameters:
    %   - bhvID: Vector of behavior labels corresponding to each time bin.

    unique_behaviors = unique(bhvID);  % Get unique behavior labels
    unique_behaviors(unique_behaviors == -1) = [];
    all_durations = [];  % Store all durations
    all_labels = [];  % Store corresponding behavior labels for durations
    
    % Compute behavior durations
    start_idx = 1;
    for i = 2:length(bhvID)
        if bhvID(i) ~= bhvID(i - 1) % Behavior transition
            bhv = bhvID(i - 1);
            duration = i - start_idx;
            all_durations = [all_durations; duration]; % Store duration
            all_labels = [all_labels; bhv];  % Store behavior label
            start_idx = i; % Update start position
        end
    end
    % Store the last segment
    bhv = bhvID(end);
    duration = length(bhvID) - start_idx + 1;
    all_durations = [all_durations; duration];
    all_labels = [all_labels; bhv];

    % Remove in-nest/sleeping
    idxRmv = all_labels == -1;
    all_labels(idxRmv) = [];
    all_durations(idxRmv) = [];

colors = colors_for_behaviors(unique_behaviors);

    % Convert behavior labels to categorical for grouping
    all_labels = categorical(all_labels, unique_behaviors, string(unique_behaviors));


    figure;
    h = boxplot(all_durations, all_labels, 'Orientation', 'horizontal', 'Whisker', 1.5);
    
   % Apply colors to each box
    hold on;
    h_boxes = findobj(gca, 'Tag', 'Box');
    for i = 1:length(unique_behaviors)
        color = colors(unique_behaviors(i)+1 - min(unique_behaviors), :);
        patch(get(h_boxes(i), 'XData'), get(h_boxes(i), 'YData'), color, ...
              'FaceAlpha', 0.6, 'EdgeColor', 'none');
    end    
    hold off;
    
    % Formatting
    xlabel('Duration (time bins)');
    ylabel('Behavior');
    title('Behavior Duration Box Plot');
    set(gca, 'YDir', 'reverse');  % Keep the order consistent with ethogram
    grid on;
end