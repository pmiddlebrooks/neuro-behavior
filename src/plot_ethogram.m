function plot_ethogram(bhvID, colors, opts)
    % PLOT_ETHOGRAM Plots an ethogram with behaviors on the y-axis and time bins on the x-axis.
    % 
    % Parameters:
    %   - bhvID: Vector of behavior labels corresponding to each time bin.
    %   - colors: Matrix where each row corresponds to an RGB color for a unique behavior.

    plusColor = 2; % Need to add 2 since bhvID starts at -1

    unique_behaviors = unique(bhvID);  % Get unique behavior labels

    % If ordering is enabled, sort behaviors in ascending order
    if nargin > 2 && opts.orderBehaviors
        bhvID = sort(bhvID, 'ascend');
    end


    bhv_to_y = containers.Map(num2cell(unique_behaviors), num2cell(1:length(unique_behaviors)));  % Map behaviors to y positions

    figure(3); clf; hold on;
    
    % Identify continuous stretches of behavior
    start_idx = 1;
    for i = 2:length(bhvID)
        if bhvID(i) ~= bhvID(i - 1) % Behavior transition
            bhv = bhvID(i - 1);
            y_val = bhv_to_y(bhv);
            color = colors(bhv+plusColor, :);

            % Draw filled block
            fill([start_idx, i-1, i-1, start_idx], ...
                 [y_val-0.4, y_val-0.4, y_val+0.4, y_val+0.4], ...
                 color, 'EdgeColor', 'none');

            start_idx = i; % Update start position
        end
    end

    % Plot last segment
    bhv = bhvID(end);
    y_val = bhv_to_y(bhv);
    color = colors(bhv+plusColor, :);
    fill([start_idx, length(bhvID), length(bhvID), start_idx], ...
         [y_val-0.4, y_val-0.4, y_val+0.4, y_val+0.4], ...
         color, 'EdgeColor', 'none');

    % Formatting
    yticks(1:length(unique_behaviors));
    yticklabels(string(unique_behaviors));
    xlabel('Time bins');
    ylabel('Behavior');
    title('Ethogram');
    xlim([1, length(bhvID)]);
    ylim([0.5, length(unique_behaviors) + 0.5]);
    set(gca, 'YDir', 'reverse'); % Ensure correct y-axis order
    box on;

    hold off;
end
