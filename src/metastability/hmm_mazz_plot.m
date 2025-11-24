%% ----------------------------
% PLOTS
%----------------------------

%% USER CHOICE: Specify how to get HMM data
% Set this to 'workspace' to use existing 'res' variable, or 'load' to load a saved model
dataSource = 'load'; % 'workspace' or 'load'

if strcmp(dataSource, 'load')
    fprintf('Loading saved HMM model...\n');
    
    % Parameters for loading saved model - CHANGE THESE AS NEEDED
    natOrReach = 'Nat'; % 'Nat' or 'Reach'
    brainArea = [];     % Optional: 'M23', 'M56', 'DS', 'VS'. If empty, loads all areas
    
    % Optional: specify binSize and minDur to load specific file
    % If not specified, will load most recent file
    binSize = []; % e.g., 0.01
    minDur = [];  % e.g., 0.04
    
    % Load the saved model
    loadArgs = {};
    if ~isempty(brainArea)
        loadArgs = [loadArgs, {'brainArea'}, {brainArea}];
    end
    if ~isempty(binSize)
        loadArgs = [loadArgs, {'binSize'}, {binSize}];
    end
    if ~isempty(minDur)
        loadArgs = [loadArgs, {'minDur'}, {minDur}];
    end
    
    [hmm_res] = hmm_load_saved_model(natOrReach, loadArgs{:});
    
    % Extract binSize and minDur from loaded model for use in pie chart section
    if isempty(brainArea)
        % If all areas loaded, extract from first non-empty area
        if isfield(hmm_res, 'hmm_results')
            for a = 1:length(hmm_res.hmm_results)
                if ~isempty(hmm_res.hmm_results{a}) && isfield(hmm_res.hmm_results{a}, 'HmmParam')
                    if isfield(hmm_res.hmm_results{a}.HmmParam, 'BinSize')
                        binSize = hmm_res.hmm_results{a}.HmmParam.BinSize;
                    end
                    if isfield(hmm_res.hmm_results{a}.HmmParam, 'MinDur')
                        minDur = hmm_res.hmm_results{a}.HmmParam.MinDur;
                    end
                    break;
                end
            end
        end
    else
        % Single area loaded
        if isfield(hmm_res, 'HmmParam')
            if isfield(hmm_res.HmmParam, 'BinSize')
                binSize = hmm_res.HmmParam.BinSize;
            end
            if isfield(hmm_res.HmmParam, 'MinDur')
                minDur = hmm_res.HmmParam.MinDur;
            end
        end
    end
    
    if ~isempty(brainArea)
        fprintf('Loaded model: %s data from %s area\n', natOrReach, brainArea);
    else
        fprintf('Loaded model: %s data from all areas\n', natOrReach);
    end
    if ~isempty(binSize) && ~isempty(minDur)
        fprintf('  binSize: %.3f, minDur: %.3f\n', binSize, minDur);
    end
    
elseif strcmp(dataSource, 'workspace')
    fprintf('Using existing HMM results from workspace.\n');
    
    % Check if required variables exist
    if ~exist('res', 'var') || isempty(res)
        error('No ''res'' variable found in workspace. Set dataSource = ''load'' to load a saved model.');
    end
    
    % Check if we have the trial parameters from hmm_results_save
    if ~exist('trial_params', 'var') && exist('hmm_results_save', 'var')
        trial_params = hmm_results_save.trial_params;
    end
    
else
    error('Invalid dataSource. Must be ''workspace'' or ''load''');
end

% Handle both single area and all areas loaded
if isfield(hmm_res, 'hmm_results') && iscell(hmm_res.hmm_results)
    % All areas loaded - use first area for initial plots, or specify which area
    if isempty(brainArea)
        % Use first available area
        areaIdx = 1;
        while areaIdx <= length(hmm_res.hmm_results) && isempty(hmm_res.hmm_results{areaIdx})
            areaIdx = areaIdx + 1;
        end
        if areaIdx > length(hmm_res.hmm_results)
            error('No HMM results found in loaded data');
        end
        hmm_res_single = hmm_res.hmm_results{areaIdx};
        fprintf('Using area %s for initial plots\n', hmm_res.areas{areaIdx});
    else
        % Extract specific area
        areaMap = containers.Map({'M23', 'M56', 'DS', 'VS'}, {1, 2, 3, 4});
        areaIdx = areaMap(brainArea);
        hmm_res_single = hmm_res.hmm_results{areaIdx};
    end
else
    % Single area loaded
    hmm_res_single = hmm_res;
end

% Extract HMM parameters
HmmParam = hmm_res_single.HmmParam;
if isfield(HmmParam, 'VarStates')
    if isfield(hmm_res_single, 'best_model') && isfield(hmm_res_single.best_model, 'best_state_index')
        bestStateIdx = hmm_res_single.best_model.best_state_index;
        if isscalar(HmmParam.VarStates)
            HmmParam.VarStates = HmmParam.VarStates;
        else
            HmmParam.VarStates = HmmParam.VarStates(bestStateIdx);
        end
    elseif isfield(hmm_res_single, 'best_model') && isfield(hmm_res_single.best_model, 'num_states')
        HmmParam.VarStates = hmm_res_single.best_model.num_states;
    elseif isscalar(HmmParam.VarStates)
        HmmParam.VarStates = HmmParam.VarStates;
    else
        HmmParam.VarStates = HmmParam.VarStates(1); % Use first if no best index
    end
elseif isfield(hmm_res_single, 'best_model') && isfield(hmm_res_single.best_model, 'num_states')
    HmmParam.VarStates = hmm_res_single.best_model.num_states;
end

% Extract HMM results
hmm_results = hmm_res_single.hmm_results;
hmm_postfit = hmm_res_single.hmm_postfit;

% Create distinguishable colors for states
numStates = HmmParam.VarStates;
colors = distinguishable_colors(max(numStates, 4));

% Plot transition probability matrix and emission probability matrix
figure;
[ha, ~] = tight_subplot(1, 2, 0.05, [0.08 0.06], [0.06 0.03]);
axes(ha(1));
imagesc(hmm_res_single.best_model.transition_matrix);
colorbar;
title('Transition Probability Matrix');
xlabel('State'); ylabel('State');

axes(ha(2));
imagesc(hmm_res_single.best_model.emission_matrix(:, 1:end-1)); % Exclude silence column
colorbar;
title('Emission Probability Matrix');
xlabel('Neuron'); ylabel('State');


%% Proportion of data in various states
% This section can be expanded later for additional analyses

%% Plot full sequence
figure(8); clf;
hold on;
sequence = hmm_res_single.continuous_results.sequence;
probabilities = hmm_res_single.continuous_results.pStates;

x = (1:length(sequence)) * HmmParam.BinSize;
numStates = size(probabilities, 2); % Number of states from probability matrix

% Get colors for states
colors = distinguishable_colors(numStates);

% Plot rectangles for each state
for state = 1:numStates
    stateIdx = sequence == state;
    if any(stateIdx)
        % Find contiguous segments
        d = diff([0; stateIdx; 0]);
        startIdx = find(d == 1);
        endIdx = find(d == -1) - 1;
        
        % Plot each segment as a rectangle
        for i = 1:length(startIdx)
            xStart = x(startIdx(i));
            xEnd = x(endIdx(i));
            patch([xStart xEnd xEnd xStart], [0 0 1 1], colors(state,:), ...
                'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
    end
end

% Plot probability traces
for state = 1:numStates
    plot(x, probabilities(:,state), 'Color', colors(state,:), 'LineWidth', 2);
end

xlabel('Time (s)');
ylabel('State Probability');
ylim([0 1]);
hold off;

%% Additional plots can be added here
fprintf('Basic HMM plots completed!\n');
fprintf('You can add more analysis and plotting code below.\n');


%% ============================
% PIE CHARTS: State proportions per brain area (including Unassigned)
%=============================
% Uses saved models per area. natOrReach is reused from above.
    natOrReach = 'Nat'; % 'Nat' or 'Reach'

try
    areas = {'M23','M56','DS','VS'};        % Areas to include
    areasToPlot = 1:numel(areas);            % Indices to include (edit if needed)

    pieData = {};
    pieColors = {};
    pieAreaNames = {};

    % Load all areas at once if available from main section, otherwise load them
    if exist('hmm_res', 'var') && isfield(hmm_res, 'hmm_results') && iscell(hmm_res.hmm_results)
        % Use already loaded all-areas data
        fprintf('Using already loaded all-areas data for pie charts\n');
        all_areas_loaded = true;
    else
        % Load all areas now
        loadArgs = {};
        if exist('binSize', 'var') && ~isempty(binSize)
            loadArgs = [loadArgs, {'binSize'}, {binSize}];
        end
        if exist('minDur', 'var') && ~isempty(minDur)
            loadArgs = [loadArgs, {'minDur'}, {minDur}];
        end
        hmm_res = hmm_load_saved_model(natOrReach, loadArgs{:});
        all_areas_loaded = true;
    end
    
    for ai = areasToPlot
        thisArea = areas{ai};
        
        % Extract area from loaded data
        if all_areas_loaded
            areaMap = containers.Map({'M23', 'M56', 'DS', 'VS'}, {1, 2, 3, 4});
            areaIdx = areaMap(thisArea);
            hmm_res_area = hmm_res.hmm_results{areaIdx};
        end
        
        if isempty(hmm_res_area) || ~isfield(hmm_res_area, 'continuous_results')
            fprintf('Warning: No continuous results for area %s, skipping.\n', thisArea);
            continue
        end

        seq = hmm_res_area.continuous_results.sequence; % 0 = unassigned
        if isempty(seq)
            fprintf('Warning: Empty sequence for area %s, skipping.\n', thisArea);
            continue
        end

        % Determine number of states
        numStatesLocal = 0;
        if isfield(hmm_res_area, 'best_model') && isfield(hmm_res_area.best_model, 'num_states')
            numStatesLocal = hmm_res_area.best_model.num_states;
        elseif isfield(hmm_res_area, 'HmmParam') && isfield(hmm_res_area.HmmParam, 'VarStates')
            if isfield(hmm_res_area, 'best_model') && isfield(hmm_res_area.best_model, 'best_state_index')
                bestStateIdx = hmm_res_area.best_model.best_state_index;
                if isscalar(hmm_res_area.HmmParam.VarStates)
                    numStatesLocal = hmm_res_area.HmmParam.VarStates;
                else
                    numStatesLocal = hmm_res_area.HmmParam.VarStates(bestStateIdx);
                end
            elseif isscalar(hmm_res_area.HmmParam.VarStates)
                numStatesLocal = hmm_res_area.HmmParam.VarStates;
            else
                numStatesLocal = hmm_res_area.HmmParam.VarStates(1);
            end
        else
            assignedStates = seq(seq > 0);
            if ~isempty(assignedStates)
                numStatesLocal = max(assignedStates);
            end
        end

        % Count bins per state and unassigned
        unassignedCount = sum(seq == 0);
        stateCounts = zeros(1, max(1, numStatesLocal));
        for s = 1:numel(stateCounts)
            stateCounts(s) = sum(seq == s);
        end
        counts = [stateCounts, unassignedCount];
        if sum(counts) == 0
            fprintf('Warning: Zero counts for area %s, skipping.\n', thisArea);
            continue
        end

        pieData{end+1} = counts / sum(counts);
        pieAreaNames{end+1} = thisArea;

        % Colors: per-state plus gray for unassigned
        % baseCols = maxdistcolor(max(1, numStatesLocal));
  func = @sRGB_to_OKLab;
  cOpts.exc = [0,0,0;1,1,1];
  baseCols = maxdistcolor(max(1, numStatesLocal),func, cOpts);
        pieColors{end+1} = [baseCols; 0.6 0.6 0.6];
    end

    if ~isempty(pieData)
        figure(99); clf;
        nAreas = numel(pieData);
        nCols = min(4, nAreas);
        nRows = ceil(nAreas / nCols);
        [haPie, ~] = tight_subplot(nRows, nCols, 0.02, [0.06 0.1], [0.05 0.03]);
        for a = 1:nAreas
            axes(haPie(a));
            ph = pie(pieData{a});
            % Apply colors to patches (pie returns patches and text handles)
            patches = findobj(ph, 'Type', 'Patch');
            cols = pieColors{a};
            for k = 1:min(numel(patches), size(cols,1))
                set(patches(end-k+1), 'FaceColor', cols(k,:)); % reverse order
            end
            title(sprintf('%s: State proportions', pieAreaNames{a}), 'Interpreter', 'none');
            axis equal off

            % Legend labels
            numStatesLocal = size(cols,1) - 1;
            labels = arrayfun(@(s) sprintf('State %d', s), 1:numStatesLocal, 'UniformOutput', false);
            labels{end+1} = 'Unassigned';
            legend(labels, 'Location', 'southoutside', 'Box', 'off');
            sgtitle(sprintf('HMM State Proportions_%s', natOrReach), 'interpreter', 'none')
        end

        % Save pie chart figure to the same folder used in hmm_mazz.m
        paths = get_paths;
        hmmdir = fullfile(paths.dropPath, 'metastability');
        if ~exist(hmmdir, 'dir')
            mkdir(hmmdir);
        end
        piePng = fullfile(hmmdir, sprintf('HMM_state_proportions_%s.png', natOrReach));
        pieFig = fullfile(hmmdir, sprintf('HMM_state_proportions_%s.fig', natOrReach));
        try
            exportgraphics(gcf, piePng, 'Resolution', 200);
        catch
            saveas(gcf, piePng);
        end
        saveas(gcf, pieFig);
        fprintf('Saved pie chart figure to: %s\n', piePng);
    else
        fprintf('No pie chart data generated. Check that saved models exist for requested areas.\n');
    end
catch ME
    fprintf('Pie chart generation failed: %s\n', ME.message);
end