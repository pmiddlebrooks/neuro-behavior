% This script uses a q learning model to fit simulated data
% Emulates ymaze_qmodel_script.m but simplified for simulated data

%% Run constants.m to get codes, event markers, subject info
addpath(fileparts(matlab.desktop.editor.getActiveFilename));
constants
paths = get_paths;

% Column indices in sessInfo matrix
subIdx = 1;
alphaRIdx = 2;
alphaUIdx = 3;
betaIdx = 4;
biasIdx = 5;
stickyIdx = 6;
wslsIdx = 7;

%% Run Q model for simulated data

% === PARAMETER CONFIGURATION ===
% Set which parameters to include in the model (1 = include, 0 = exclude)
% Make parameter configuration global for fit_Qvalue function
global paramConfig activeParams

paramConfig.alphaR = 1;    % Learning rate for rewarded trials
paramConfig.alphaU = 1;    % Learning rate for unrewarded trials  
paramConfig.beta = 1;      % Inverse temperature (choice sensitivity)
paramConfig.bias = 1;      % Choice bias
paramConfig.sticky = 1;    % Stickiness (perseveration)
paramConfig.wsls = 1;      % Win-stay, lose-shift

% Create parameter info structure
paramNames = fieldnames(paramConfig);
activeParams = paramNames([paramConfig.alphaR, paramConfig.alphaU, paramConfig.beta, paramConfig.bias, paramConfig.sticky, paramConfig.wsls] == 1);
nActiveParams = length(activeParams);

fprintf('Active parameters: %s\n', strjoin(activeParams, ', '));

% === INITIAL PARAMETER GUESSES AND BOUNDS ===
% Default values for each parameter
defaultValues = struct();
defaultValues.alphaR = 0.2;
defaultValues.alphaU = 0.2;
defaultValues.beta = 2;
defaultValues.bias = 0;
defaultValues.sticky = 0;
defaultValues.wsls = 0;

% Bounds for each parameter
lowerBounds = struct();
lowerBounds.alphaR = 0;
lowerBounds.alphaU = 0;
lowerBounds.beta = 0;
lowerBounds.bias = -10;
lowerBounds.sticky = -10;
lowerBounds.wsls = -10;

upperBounds = struct();
upperBounds.alphaR = 1;
upperBounds.alphaU = 1;
upperBounds.beta = 10;
upperBounds.bias = 10;
upperBounds.sticky = 10;
upperBounds.wsls = 10;

% Build parameter vectors for active parameters only
initParams = [];
lb = [];
ub = [];

for i = 1:length(activeParams)
    param = activeParams{i};
    initParams = [initParams, defaultValues.(param)];
    lb = [lb, lowerBounds.(param)];
    ub = [ub, upperBounds.(param)];
end

nStarts = 50;

% === OPTIMIZATION OPTIONS ===
optsMod = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');

% Constants, initialize variables
plotFlag = 1;  % Enable plotting for simulated data
plotDiagnostics = 0;
nTrialsRemove = 7;  % Number of trials to remove from the beginning of each block

%% Load simulated data
% Assuming simulated data is stored in a .mat file with structure:
sd = load([paths.dropPath, '/ymaze/forPaul']);
% load([paths.dropPath, '/ymaze/forPaul'])
% simData.choices - cell array of choice vectors for each subject
% simData.rewards - cell array of reward vectors for each subject
% simData.trueParams - true parameters used to generate data (optional)

% Get subject fields from sd struct
subjectFields = fieldnames(sd);
nSubjects = length(subjectFields);

%% Initialize storage arrays
% Initialize parameter storage arrays
paramStorage = struct();
for i = 1:length(activeParams)
    param = activeParams{i};
    paramStorage.(param) = cell(nSubjects, 1);
end

% Initialize mean parameter arrays
paramMeans = struct();
for i = 1:length(activeParams)
    param = activeParams{i};
    paramMeans.(param) = nan(nSubjects, 1);
end

sessInfo = [];

%% Fit Q-learning model to each subject's data
for sub = 1:nSubjects
    fprintf('Fitting model for Subject %d\n', sub);
    subj = subjectFields{sub};
    % Get choices and rewards for this subject
    data = getfield(sd, subj);
    choices = data(1,:)' + 1; % First row contains choices
    rewards = data(2,:)'; % Second row contains rewards
    
    % === OBJECTIVE FUNCTION ===
    negLLfun = @(params) -sum(fit_Qvalue(choices, rewards, params));  % minimize -log likelihood

    % === RUN fmincon with multiple starts ===
    bestNegLL = Inf;
    bestParams = [];

    for i = 1:nStarts
        % Generate random initial parameters for active parameters only
        initParams = [];
        for j = 1:length(activeParams)
            param = activeParams{j};
            switch param
                case {'alphaR', 'alphaU'}
                    initParams = [initParams, rand];
                case 'beta'
                    initParams = [initParams, rand*5];
                case {'bias', 'sticky', 'wsls'}
                    initParams = [initParams, randn];
            end
        end
        
        try
            [params, negLL] = fmincon(@(p) -sum(fit_Qvalue(choices, rewards, p)), ...
                initParams, [], [], [], [], lb, ub, [], optsMod);

            if negLL < bestNegLL
                bestNegLL = negLL;
                bestParams = params;
            end
        catch ME
            fprintf('  Warning: Optimization failed for start %d: %s\n', i, ME.message);
        end
    end
    
    if isempty(bestParams)
        fprintf('  Error: All optimization attempts failed for Subject %d\n', sub);
        continue;
    end

    % === STORE RESULTS ===
    paramIdx = 1;
    for i = 1:length(activeParams)
        param = activeParams{i};
        if paramConfig.(param) == 1
            paramStorage.(param){sub} = bestParams(paramIdx);
            paramIdx = paramIdx + 1;
        else
            paramStorage.(param){sub} = nan;
        end
    end
    
    % Get parameter values for sessInfo
    paramValues = [];
    for i = 1:length(activeParams)
        param = activeParams{i};
        if paramConfig.(param) == 1
            paramValues = [paramValues, paramStorage.(param){sub}];
        else
            paramValues = [paramValues, nan];
        end
    end
    
    % Store in session info matrix
    sessInfo = [sessInfo; [sub, paramValues]];

    % Calculate means for active parameters
    for i = 1:length(activeParams)
        param = activeParams{i};
        if paramConfig.(param) == 1
            paramMeans.(param)(sub) = paramStorage.(param){sub};
        else
            paramMeans.(param)(sub) = nan;
        end
    end

    fprintf('  Fit results:\n');
    for i = 1:length(activeParams)
        param = activeParams{i};
        fprintf('    %s = %.3f\n', param, paramStorage.(param){sub});
    end
    fprintf('    Log-likelihood = %.3f\n', -bestNegLL);

    % Visualize Q-values and choice probabilities if plotting is enabled
    if plotDiagnostics
        plot_Qmodel_diagnostics(choices, rewards, bestParams);
        tText = sprintf('Q-Learning Model - Simulated Subject %d', sub);
        sgtitle(tText);
        
        % Save plot
        print('-dpng', fullfile(paths.dropPath, 'ymaze/', sprintf('simulated_subject_%d', sub)));
    end
end

%% Compare fitted parameters with true parameters (if available)
if isfield(simData, 'trueParams')
    fprintf('\n=== PARAMETER RECOVERY ANALYSIS ===\n');
    
    trueParams = simData.trueParams;
    paramNames = activeParams;  % Use active parameters
    
    for p = 1:length(paramNames)
        param = paramNames{p};
        if isfield(trueParams, param) && paramConfig.(param) == 1
            fittedVals = [];
            trueVals = [];
            
            for sub = 1:nSubjects
                if ~isnan(paramStorage.(param){sub})  % Check if fitting was successful
                    fittedVals = [fittedVals; paramStorage.(param){sub}];
                    trueVals = [trueVals; trueParams.(param)(sub)];
                end
            end
            
            if ~isempty(fittedVals)
                % Calculate correlation
                corrVal = corr(fittedVals, trueVals);
                fprintf('%s: r = %.3f\n', param, corrVal);
                
                % Plot recovery
                figure(100 + p);
                scatter(trueVals, fittedVals, 50, 'filled');
                hold on;
                plot([min(trueVals), max(trueVals)], [min(trueVals), max(trueVals)], 'r--');
                xlabel('True Parameter');
                ylabel('Fitted Parameter');
                title(sprintf('%s Parameter Recovery (r = %.3f)', param, corrVal));
                grid on;
                
                % Save recovery plot
                print('-dpng', fullfile(paths.dropPath, 'ymaze/', sprintf('recovery_%s', param)));
            end
        end
    end
end

%% Create summary plots
if plotFlag
    % Plot all fitted parameters across subjects
    figure(200);
    paramNames = activeParams;  % Use active parameters
    
    nParams = length(paramNames);
    ha = tight_subplot(2, ceil(nParams/2), [0.07 0.05], [0.05 0.1]);
    
    for p = 1:nParams
        param = paramNames{p};
        if paramConfig.(param) == 1
            axes(ha(p));
            
            % Collect parameter values
            paramVals = [];
            for sub = 1:nSubjects
                if ~isnan(paramStorage.(param){sub})  % Check if fitting was successful
                    paramVals = [paramVals; paramStorage.(param){sub}];
                end
            end
            
            if ~isempty(paramVals)
                histogram(paramVals, 10);
                title(sprintf('%s Distribution', param));
                xlabel('Parameter Value');
                ylabel('Count');
            end
        end
    end
    
    sgtitle('Fitted Parameter Distributions Across Subjects');
    print('-dpng', fullfile(paths.dropPath, 'ymaze/', 'parameter_distributions'));
end

%% Calculate and plot parameter summaries across subjects
% Plotting constants
xVal = 1:length(activeParams);  % Dynamic based on number of active parameters

% Collect all parameter values for plotting
allParamVals = zeros(nSubjects, length(activeParams));
paramNames = activeParams;

for sub = 1:nSubjects
    for p = 1:length(activeParams)
        param = activeParams{p};
        if paramConfig.(param) == 1
            allParamVals(sub, p) = paramStorage.(param){sub};
        else
            allParamVals(sub, p) = nan;
        end
    end
end

% Remove rows with NaN values (failed fits)
validRows = ~any(isnan(allParamVals), 2);
allParamVals = allParamVals(validRows, :);
nValidSubjects = sum(validRows);

if nValidSubjects > 0
    % Create single figure with scatter plot
    figure('Position', [-1600, 166, 1000, 600]);
    
    % Add jitter to x-positions for better visibility
    xJitter = (rand(nValidSubjects, length(activeParams)) - 0.5) * 0.2;
    
    % Plot individual subject data points with different colors
    legendHandles = [];
    legendLabels = {};
    
    % Get valid subject field names (those that had successful fits)
    validSubjectFields = subjectFields(validRows);
    colors = lines(nValidSubjects); % Use MATLAB's lines colormap
    
    for s = 1:nValidSubjects
        subjectData = allParamVals(s, :);
        h = scatter(xVal + xJitter(s, :), subjectData, 100, 'o', 'MarkerFaceColor', colors(s, :), 'MarkerEdgeColor', 'none');
        hold on;
        
        % Store handle and label for legend using actual subject name
        legendHandles = [legendHandles; h];
        legendLabels{s} = validSubjectFields{s};
    end
    
    % Formatting
    yline(0, '--k');
    xlim([0.5, length(activeParams)+0.5]);
    ylim([-1, 10]);
    xticks(xVal);
    xticklabels(paramNames);
    ylabel('Parameter Value');
    xlabel('Parameters');
    title(sprintf('Simulated Data: Parameter Values Across Subjects (n=%d)', nValidSubjects));
    set(gca, 'YTickLabelMode', 'auto', 'FontSize', 14);
    
    % Add legend
    legend(legendHandles, legendLabels, 'Location', 'best', 'FontSize', 12);
    
    % Save the figure
    print('-dpng', fullfile(paths.dropPath, 'ymaze/', 'simulated_parameter_summary'));
end

%% Save results
save(fullfile(paths.dropPath, 'ymaze/', 'simulated_model_results.mat'), ...
    'paramStorage', 'paramMeans', 'sessInfo', 'simData', 'paramConfig', 'activeParams');

fprintf('\nResults saved to: %s\n', fullfile(paths.dropPath, 'ymaze/', 'simulated_model_results.mat'));

%% Helper function to create example simulated data
function simData = create_example_simulated_data()
    % Create example simulated data for testing
    nSubjects = 10;
    nTrials = 200;
    
    simData.choices = cell(nSubjects, 1);
    simData.rewards = cell(nSubjects, 1);
    simData.trueParams = struct();
    
    % True parameters for each subject
    simData.trueParams.alphaR = 0.1 + 0.2 * rand(nSubjects, 1);
    simData.trueParams.alphaU = 0.1 + 0.2 * rand(nSubjects, 1);
    simData.trueParams.beta = 1 + 3 * rand(nSubjects, 1);
    simData.trueParams.bias = -0.5 + rand(nSubjects, 1);
    simData.trueParams.sticky = -0.5 + rand(nSubjects, 1);
    simData.trueParams.wsls = -0.5 + rand(nSubjects, 1);
    
    for sub = 1:nSubjects
        % Generate choices and rewards using the true parameters
        [choices, rewards] = generate_simulated_data(nTrials, simData.trueParams.alphaR(sub), ...
            simData.trueParams.alphaU(sub), simData.trueParams.beta(sub), ...
            simData.trueParams.bias(sub), simData.trueParams.sticky(sub), ...
            simData.trueParams.wsls(sub));
        
        simData.choices{sub} = choices;
        simData.rewards{sub} = rewards;
    end
    
    fprintf('Created example simulated data for %d subjects with %d trials each\n', nSubjects, nTrials);
end

function [choices, rewards] = generate_simulated_data(nTrials, alphaR, alphaU, beta, bias, sticky, wsls)
    % Generate simulated choices and rewards using Q-learning model
    
    choices = zeros(nTrials, 1);
    rewards = zeros(nTrials, 1);
    
    % Initialize Q-values
    Q = [0.5, 0.5];  % Initial Q-values for right and left
    prev_choice = 0;
    
    for t = 1:nTrials
        % Stickiness term
        stick_term = 0;
        if prev_choice == 1
            stick_term = +1;
        elseif prev_choice == 2
            stick_term = -1;
        end
        
        % WSLS term
        wsls_term = 0;
        if t > 1
            prev_reward = rewards(t-1);
            if (prev_reward == 1 && choices(t) == prev_choice) || (prev_reward == 0 && choices(t) ~= prev_choice)
                wsls_term = +1;
            else
                wsls_term = -1;
            end
        end
        
        % Choice probability
        p_right = 1 / (1 + exp(-beta * (Q(1) - Q(2)) + bias + sticky * stick_term + wsls * wsls_term));
        
        % Make choice
        if rand < p_right
            choices(t) = 1;  % Right
        else
            choices(t) = 2;  % Left
        end
        
        % Generate reward (simple 70% correct reward structure)
        if choices(t) == 1  % Right choice
            if rand < 0.7
                rewards(t) = 1;
            else
                rewards(t) = 0;
            end
        else  % Left choice
            if rand < 0.3
                rewards(t) = 1;
            else
                rewards(t) = 0;
            end
        end
        
        % Update Q-values
        alpha = alphaR * (rewards(t) == 1) + alphaU * (rewards(t) == 0);
        Q(choices(t)) = Q(choices(t)) + alpha * (rewards(t) - Q(choices(t)));
        
        % Update previous choice
        prev_choice = choices(t);
    end
end 






function [LL_total, Q_total, p_choice] = fit_Qvalue(choices, rewards, params)
% Q-learning with softmax, bias, optional stickiness, and optional WSLS term
% Inputs:
% - choices: vector (1 = right, 2 = left)
% - rewards: vector (0 or 1)
% - params: parameter vector with active parameters only
% - paramConfig: structure defining which parameters are active (passed via global)
%
% Outputs:
% - LL_total: log-likelihood per trial
% - Q_total: Q-values per trial (nTrials+1 x 2)
% - p_choice: model probability of choosing right per trial

% Get parameter configuration from global workspace
global paramConfig activeParams

% === UNPACK PARAMS ===
paramIdx = 1;
alphaR = 0.2;  % Default values
alphaU = 0.2;
beta = 2;
bias = 0;
stickiness = 0;
wsls = 0;

% Assign parameters based on active configuration
for i = 1:length(activeParams)
    param = activeParams{i};
    if paramConfig.(param) == 1
        switch param
            case 'alphaR'
                alphaR = params(paramIdx);
            case 'alphaU'
                alphaU = params(paramIdx);
            case 'beta'
                beta = params(paramIdx);
            case 'bias'
                bias = params(paramIdx);
            case 'sticky'
                stickiness = params(paramIdx);
            case 'wsls'
                wsls = params(paramIdx);
        end
        paramIdx = paramIdx + 1;
    end
end

nTrials = length(choices);
Q_total = zeros(nTrials + 1, 2);  % Q(:,1)=right, Q(:,2)=left
LL_total = zeros(nTrials, 1);     % log-likelihood per trial
p_choice = zeros(nTrials, 1);     % predicted P(choose right)

prev_choice = 0;  % initialize previous choice (0 = no prior)

% === MAIN TRIAL LOOP ===
for t = 1:nTrials
    Q = Q_total(t, :);  % current Q-values
    QR = Q(1); QL = Q(2);

    % Stickiness term: +1 if previous choice = right, -1 if left, 0 if no history
    stick_term = 0;
    if prev_choice == 1
        stick_term = +1;
    elseif prev_choice == 2
        stick_term = -1;
    end

    % WSLS term: +1 if previous choice was rewarded and we're choosing the same, or if previous choice was unrewarded and we're choosing different
    wsls_term = 0;
    if t > 1  % Need at least one previous trial
        prev_reward = rewards(t-1);
        if (prev_reward == 1 && choices(t) == prev_choice) || (prev_reward == 0 && choices(t) ~= prev_choice)
            wsls_term = +1;
        else
            wsls_term = -1;
        end
    end

    % Compute softmax probability of choosing right
    p_right = 1 / (1 + exp(-beta * (QR - QL) + bias + stickiness * stick_term + wsls * wsls_term));
    p_choice(t) = p_right;

    % Log-likelihood for actual choice
    if choices(t) == 1
        LL_total(t) = log(p_right + eps);
    elseif choices(t) == 2
        LL_total(t) = log(1 - p_right + eps);
    else
        error('Invalid choice at trial %d. Must be 1 (right) or 2 (left).', t);
    end

    % Update Q-values
    chosen = choices(t);
    reward = rewards(t);
    alpha = alphaR * (reward == 1) + alphaU * (reward == 0);
    Q(chosen) = Q(chosen) + alpha * (reward - Q(chosen));
    Q_total(t + 1, :) = Q;

    % Update previous choice
    prev_choice = chosen;
end
end




function plot_Qmodel_diagnostics(choices, rewards, params)
% Q-learning diagnostic visualizer using tight_subplot (3x2 grid)
% Inputs:
% - choices: vector (1 = right, 2 = left)
% - rewards: vector (0 or 1)
% - params: parameter vector with active parameters only
% - paramConfig: structure defining which parameters are active (passed via global)

% Get parameter configuration from global workspace
global paramConfig activeParams

% Unpack parameters
paramIdx = 1;
alphaR = 0.2;  % Default values
alphaU = 0.2;
beta = 2;
bias = 0;
stickiness = 0;
wsls = 0;

% Assign parameters based on active configuration
for i = 1:length(activeParams)
    param = activeParams{i};
    if paramConfig.(param) == 1
        switch param
            case 'alphaR'
                alphaR = params(paramIdx);
            case 'alphaU'
                alphaU = params(paramIdx);
            case 'beta'
                beta = params(paramIdx);
            case 'bias'
                bias = params(paramIdx);
            case 'sticky'
                stickiness = params(paramIdx);
            case 'wsls'
                wsls = params(paramIdx);
        end
        paramIdx = paramIdx + 1;
    end
end

nTrials = length(choices);

Q = zeros(2,1);
Q_total = zeros(nTrials+1, 2);
p_right = zeros(nTrials, 1);

% Run model with all terms
prev_choice = 0;
for t = 1:nTrials
    QR = Q(1); QL = Q(2);
    
    % Stickiness term
    stick_term = 0;
    if prev_choice == 1
        stick_term = +1;
    elseif prev_choice == 2
        stick_term = -1;
    end
    
    % WSLS term
    wsls_term = 0;
    if t > 1
        prev_reward = rewards(t-1);
        if (prev_reward == 1 && choices(t) == prev_choice) || (prev_reward == 0 && choices(t) ~= prev_choice)
            wsls_term = +1;
        else
            wsls_term = -1;
        end
    end
    
    p = 1 / (1 + exp(-beta * (QR - QL) + bias + stickiness * stick_term + wsls * wsls_term));
    p_right(t) = p;

    chosen = choices(t);
    reward = rewards(t);
    alpha = alphaR * (reward == 1) + alphaU * (reward == 0);
    Q(chosen) = Q(chosen) + alpha * (reward - Q(chosen));
    Q_total(t+1, :) = Q;
    
    prev_choice = chosen;
end

% === Compute choice switching after reward/no-reward ===
switchAfterReward = [];
switchAfterNoReward = [];

for t = 2:nTrials
    if choices(t) ~= choices(t-1)
        if rewards(t-1) == 1
            switchAfterReward(end+1) = 1;
        else
            switchAfterNoReward(end+1) = 1;
        end
    else
        if rewards(t-1) == 1
            switchAfterReward(end+1) = 0;
        else
            switchAfterNoReward(end+1) = 0;
        end
    end
end

% Calculate means
meanSwitchR = mean(switchAfterReward);
meanSwitchU = mean(switchAfterNoReward);

% === PLOT ===
figure(88); clf;
set(gcf, 'Position', [-1650, 31, 1668, 960]); % Example position and size
ha = tight_subplot(3, 2, [0.06 0.06], [0.07 0.08], [0.08 0.03]);

% 1. Q-values
axes(ha(1));
plot(Q_total(1:end-1,1), 'r-', 'DisplayName', 'Q-Right'); hold on;
plot(Q_total(1:end-1,2), 'b-', 'DisplayName', 'Q-Left');
ylabel('Q'); title('Q-values'); legend; set(gca, 'XTick', []);

% 2. Choices
axes(ha(2));
plot(choices, 'ko');
ylim([0.5 2.5]); yticks([1 2]); yticklabels({'Right','Left'});
ylabel('Choice'); title('Subject Choices'); set(gca, 'XTick', []);

% 3. Rewards
axes(ha(3));
stem(rewards, 'filled', 'MarkerSize', 3);
ylim([-0.1 1.1]); ylabel('Rwd');
title('Reward Outcomes'); set(gca, 'XTick', []);

% 4. Choice Probabilities
axes(ha(4));
plot(p_right, 'k-', 'LineWidth', 1.2);
ylim([0 1]); ylabel('P(Right)');
yline(0, '--')
title('Model Choice Prob'); set(gca, 'XTick', []);

% 5. Switch rate after reward/no-reward
axes(ha(5));
bar([1 2], [meanSwitchR meanSwitchU], 0.5);
xticks([1 2]); xticklabels({'After Reward', 'After No Reward'});
yline(0, '--')
ylabel('P(Switch)'); ylim([0 1]); title('Choice Switching');

% 6. Fitted Parameters
axes(ha(6));
param_names = {};
param_vals = [];
for i = 1:length(activeParams)
    param = activeParams{i};
    if paramConfig.(param) == 1
        param_names{end+1} = param;
        switch param
            case 'alphaR', param_vals(end+1) = alphaR;
            case 'alphaU', param_vals(end+1) = alphaU;
            case 'beta', param_vals(end+1) = beta;
            case 'bias', param_vals(end+1) = bias;
            case 'sticky', param_vals(end+1) = stickiness;
            case 'wsls', param_vals(end+1) = wsls;
        end
    end
end
scatter(1:length(param_vals), param_vals, 60, 'ko', 'LineWidth', 2);
xlim([0.5, length(param_vals)+0.5]); ylim padded;
xticks(1:length(param_vals)); xticklabels(param_names);
yline(0, '--')
ylabel('Value'); title('Fitted Parameters');

end



function plotInteraction(T, var1, var2, paramName)
% Helper function to plot interaction effects
% T: data table, var1/var2: variable names, paramName: parameter name

% Get unique values for each variable
vals1 = unique(T.(var1));
vals2 = unique(T.(var2));

% Calculate means for each combination
means = zeros(length(vals1), length(vals2));
sems = zeros(length(vals1), length(vals2));

for i = 1:length(vals1)
    for j = 1:length(vals2)
        subset = T(T.(var1) == vals1(i) & T.(var2) == vals2(j), :);
        if ~isempty(subset)
            means(i, j) = mean(subset.Value);
            sems(i, j) = std(subset.Value) / sqrt(length(subset.Value));
        end
    end
end

% Create plot
hold on;
colors = lines(length(vals1));
for i = 1:length(vals1)
    errorbar(vals2, means(i, :), sems(i, :), 'o-', 'Color', colors(i, :), 'LineWidth', 2, 'MarkerSize', 8);
end

xlabel(var2);
ylabel('Mean Value');
title(sprintf('%s: %s × %s Interaction', paramName, var1, var2));
legend(arrayfun(@(x) sprintf('%s = %d', var1, x), vals1, 'UniformOutput', false), 'Location', 'best');
grid on;
end