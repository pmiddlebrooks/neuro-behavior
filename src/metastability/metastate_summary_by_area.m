%% Metastate Summary by Brain Area
% Loops through brain areas for a given natOrReach, loads HMM results saved by
% hmm_mazz.m, computes metastates using detect_metastates_vidaurre, and plots:
%   1) Proportion of time in each metastate (fractional occupancy, includes Unassigned)
%   2) Proportion of within-metastate vs between-metastate transitions per metastate
%
% Requirements:
% - Files saved as: HMM_results_<NatOrReach>_<Area>.mat in paths.dropPath/metastability
% - Function detect_metastates_vidaurre.m available on MATLAB path (this folder)
% - tight_subplot.m available on MATLAB path for plotting
%
% Notes:
% - Uses hmm_res.best_model.transition_matrix (KxK) for metastate detection
% - Uses hmm_res.continuous_results.sequence for time occupancy and transitions
% - Treats 0 in sequence as Unassigned (excluded from metastate mapping)

% ------------ User-configurable section ------------
natOrReach = 'Nat'; % 'Nat' or 'Reach'
areas = {'M23','M56','DS','VS'}; % adjust if needed
areasToPlot = 1:numel(areas);    % subset indices if desired, e.g., [1 3 4]
verbose = false;                 % verbose output from community detection

% ------------ Locate results directory ------------
paths = get_paths;
hmmdir = fullfile(paths.dropPath, 'metastability');
if ~exist(hmmdir,'dir')
    error('HMM results directory not found: %s', hmmdir);
end

% ------------ Pre-allocate outputs ------------
allAreaOccupancy = cell(1, numel(areas));
allAreaTransStats = cell(1, numel(areas));
allAreaMetaLabels = cell(1, numel(areas));

% ------------ Process areas ------------
for ai = areasToPlot
    areaName = areas{ai};
    resFile = fullfile(hmmdir, sprintf('HMM_results_%s_%s.mat', natOrReach, areaName));
    if ~exist(resFile,'file')
        warning('Missing HMM results for %s (%s): %s', natOrReach, areaName, resFile);
        continue
    end

    S = load(resFile, 'hmm_res');
    if ~isfield(S,'hmm_res')
        warning('File does not contain hmm_res: %s', resFile);
        continue
    end
    hmmRes = S.hmm_res;

    if ~isfield(hmmRes,'best_model') || ~isfield(hmmRes.best_model,'transition_matrix')
        warning('Missing transition matrix in %s', resFile);
        continue
    end
    if ~isfield(hmmRes,'continuous_results') || ~isfield(hmmRes.continuous_results,'sequence')
        warning('Missing continuous sequence in %s', resFile);
        continue
    end

    % Transition matrix and metastate detection
    P = hmmRes.best_model.transition_matrix; % K x K
    try
        communities = detect_metastates_vidaurre(P, verbose); % Kx1 labels (1..M)
    catch ME
        warning('Metastate detection failed for %s: %s', areaName, ME.message);
        continue
    end

    % Sequence in continuous time (1..K states, 0 = unassigned)
    seq = hmmRes.continuous_results.sequence(:);
    K = size(P,1);
    if any(seq > K)
        warning('Sequence contains state labels > K in %s; clipping.', areaName);
        seq(seq > K) = 0;
    end

    % ---------- Fractional occupancy ----------
    % Count time bins per metastate (exclude unassigned=0), and include Unassigned as its own category
    assignedIdx = seq > 0;
    assignedStates = seq(assignedIdx);
    Mlabels = unique(communities(:)'); %#ok<NASGU>
    numMeta = max(communities);
    metaCounts = zeros(numMeta,1);
    for s = 1:K
        meta = communities(s);
        metaCounts(meta) = metaCounts(meta) + sum(assignedStates == s);
    end
    unassignedCount = sum(seq == 0);
    totalBins = numel(seq);
    occupancyFrac = [metaCounts; unassignedCount] / max(1, totalBins);

    % ---------- Transition stats (within vs between) ----------
    % Consider only transitions where both time points are assigned and the state changes
    src = seq(1:end-1);
    dst = seq(2:end);
    valid = src > 0 & dst > 0 & dst ~= src;
    src = src(valid);
    dst = dst(valid);

    % Count transitions by metastate of origin
    withinCounts = zeros(numMeta,1);
    betweenCounts = zeros(numMeta,1);
    for i = 1:numel(src)
        msSrc = communities(src(i));
        msDst = communities(dst(i));
        if msSrc == msDst
            withinCounts(msSrc) = withinCounts(msSrc) + 1;
        else
            betweenCounts(msSrc) = betweenCounts(msSrc) + 1;
        end
    end

    % Overall metastate switching: between / (within + between) across all metastates
    totalWithin = sum(withinCounts);
    totalBetween = sum(betweenCounts);
    totalTransitions = totalWithin + totalBetween;

    if totalTransitions > 0
        overallBetweenFrac = totalBetween / totalTransitions;
        overallWithinFrac = totalWithin / totalTransitions;
    else
        overallBetweenFrac = 0;
        overallWithinFrac = 0;
    end

    % Per-metastate stability: within / (within + between) for each metastate
    transTotals = withinCounts + betweenCounts;
    perMetaWithinFrac = withinCounts ./ max(1, transTotals);
    perMetaBetweenFrac = betweenCounts ./ max(1, transTotals);

    % Store results: [overall_between, overall_within, per_meta_within_1, per_meta_within_2, ...]
    allAreaTransStats{ai} = [overallBetweenFrac, overallWithinFrac, perMetaWithinFrac'];
    allAreaOccupancy{ai} = occupancyFrac;             % (numMeta+1)x1 (last is Unassigned)
    allAreaMetaLabels{ai} = communities;              % Kx1
end

% ------------ Plotting ------------
validAreas = areasToPlot(cellfun(@(c) ~isempty(c), allAreaOccupancy(areasToPlot)));
if isempty(validAreas)
    error('No valid areas processed. Check that HMM results exist.');
end

% Determine max number of metastates across areas for consistent y-lims/legends
maxNumMeta = 0;
for ai = validAreas
    occ = allAreaOccupancy{ai};
    maxNumMeta = max(maxNumMeta, numel(occ) - 1);
end

% Colors
cmap = lines(max(3, maxNumMeta+1)); % +1 for Unassigned category

% Figure 1: Fractional occupancy (top row) and transition statistics (bottom row)
figure(501); clf;
[ha, ~] = tight_subplot(2, numel(validAreas), 0.06, [0.08 0.08], [0.06 0.02]);

colIdx = 0;
for ai = validAreas
    colIdx = colIdx + 1;
    areaName = areas{ai};

    % Occupancy subplot
    ax1 = ha(colIdx);
    axes(ax1); %#ok<LAXES>
    occ = allAreaOccupancy{ai};
    numMeta = numel(occ) - 1;
    bar(1:(numMeta+1), occ, 0.8, 'FaceColor', 'flat');
    for ii = 1:(numMeta+1)
        ax1.Children(1).CData(ii,:) = cmap(ii,:);
    end
    set(ax1, 'XTick', 1:(numMeta+1), 'XTickLabel', [arrayfun(@(x) sprintf('M%d', x), 1:numMeta, 'UniformOutput', false), {'Unassigned'}], 'XTickLabelRotation', 45);
    ylim([0 1]);
    ylabel('Frac. occupancy');
    title(sprintf('%s: Occupancy', areaName));

    % Transition subplot: Overall switching + per-metastate stability
    ax2 = ha(colIdx + numel(validAreas));
    axes(ax2); %#ok<LAXES>
    trans = allAreaTransStats{ai}; % [overall_between, overall_within, per_meta_within_1, per_meta_within_2, ...]
    if isempty(trans) || length(trans) < 2
        bar(0,0);
        title(sprintf('%s: Transitions', areaName));
        continue
    end
    
    overallBetween = trans(1);
    overallWithin = trans(2);
    perMetaWithin = trans(3:end); % per-metastate stability
    
    % Create separate bars: Overall switching + per-metastate stability
    overallData = [overallWithin, overallBetween]; % 1x2 [switching, stability] for overall
    perMetaData = [perMetaWithin; 1-perMetaWithin]'; % numMeta x 2 [stability, switching] for each metastate
    
    % Plot: First bar = overall, then one bar per metastate
    barData = [overallData; perMetaData]; % (1+numMeta) x 2
    b = bar(barData, 'stacked');
    b(1).FaceColor = [0.8 0.3 0.3]; % switching (red)
    b(2).FaceColor = [0.2 0.6 0.8]; % stability (blue)
    
    % Set x-axis labels
    xLabels = {'Overall', arrayfun(@(x) sprintf('M%d', x), 1:length(perMetaWithin), 'UniformOutput', false)};
    set(ax2, 'XTick', 1:size(barData,1), 'XTickLabel', xLabels, 'XTickLabelRotation', 45);
    ylim([0 1]);
    ylabel('Frac. transitions');
    title(sprintf('%s: Switching & Stability', areaName));
    if colIdx == numel(validAreas)
        legend({'Switching', 'Stability'}, 'Location', 'eastoutside');
    end
end

sgtitle(sprintf('Metastate occupancy and switching/stability (%s)', natOrReach));

% ------------ Save figure ------------
outName = fullfile(hmmdir, sprintf('Metastate_Summary_%s.png', natOrReach));
saveas(gcf, outName);


