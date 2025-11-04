%%                     Load results for SVM decoding comparisons
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script loads results saved by svm_decoding_compare.m and prepares the
% workspace variables so that svm_decoding_compare_plot.m can plot from them.

% ------------------------------ User options ---------------------------------
paths = get_paths;

% Where results were saved
savePath = fullfile(paths.dropPath, 'decoding');

% Identify which saved results to load
dataType = 'naturalistic';  % 'naturalistic' or 'reach' (used only for colors/labels and filenames)
kernelFunction = 'polynomial';
transOrWithin = 'all';      % 'all' | 'trans' | 'transPost' | 'within'
nDim = 8;                   % latent dimension used when saving
frameSize = .15;            % frame/bin size used when saving
nShuffles = 2;              % number of permutations used when saving

% Plot flags (compatible with svm_decoding_compare_plot.m)
plotFullMap = 0;      % full time plots require bhvID which is not saved; keep 0 by default
plotModelData = 1;
plotComparisons = 1;
savePlotFlag = 1;

% Choose target monitor for plotting
monitorPositions = get(0, 'MonitorPositions');
targetMonitor = monitorPositions(1, :);

% ---------------------------- Load saved results ------------------------------
fileName = sprintf('svm_%s_decoding_compare_multi_area_%s_nDim%d_bin%.2f_nShuffles%d.mat', ...
    kernelFunction, transOrWithin, nDim, frameSize, nShuffles);
fullFilePath = fullfile(savePath, fileName);

if ~exist(fullFilePath, 'file')
    error('Results file not found: %s', fullFilePath);
end

S = load(fullFilePath, 'allResults');
if ~isfield(S, 'allResults')
    error('allResults not found in file: %s', fullFilePath);
end
allResults = S.allResults;

% Use parameters from file if available (authoritative)
if isfield(allResults, 'parameters')
    if isfield(allResults.parameters, 'frameSize'); frameSize = allResults.parameters.frameSize; end
    if isfield(allResults.parameters, 'nShuffles'); nShuffles = allResults.parameters.nShuffles; end
    if isfield(allResults.parameters, 'kernelFunction'); kernelFunction = allResults.parameters.kernelFunction; end
    if isfield(allResults.parameters, 'nDim'); nDim = allResults.parameters.nDim; end
    if isfield(allResults.parameters, 'transOrWithin'); transOrWithin = allResults.parameters.transOrWithin; end
end

% ------------------------ Reconstruct plotting context ------------------------
% Areas and selection
if isfield(allResults, 'areas'); areas = allResults.areas; else; areas = {'M23','M56','DS','VS'}; end
if isfield(allResults, 'areasToTest'); areasToTest = allResults.areasToTest; else; areasToTest = 1:numel(areas); end

% Labels and colors (approximate defaults to match creation-time behavior)
switch lower(dataType)
    case 'reach'
        behaviors = {'pre-reach', 'reach', 'pre-reward', 'reward', 'post-reward', 'intertrial'};
        colorsAdjust = 0;
        % Color set
        try
            func = @sRGB_to_OKLab; cOpts.exc = [0,0,0]; cOpts.Lmax = .8;
            colors = maxdistcolor(numel(behaviors), func, cOpts);
        catch
            colors = lines(numel(behaviors));
        end
        if size(colors,1) >= numel(behaviors)
            colors(end,:) = [.85 .8 .75];
        end
    otherwise
        % Naturalistic defaults
        behaviors = {'investigate_1', 'investigate_2', 'investigate_3', ...
            'rear', 'dive_scrunch', 'paw_groom', 'face_groom_1', 'face_groom_2', ...
            'head_groom', 'contra_body_groom', 'ipsi_body groom', 'contra_itch', ...
            'ipsi_itch_1', 'contra_orient', 'ipsi_orient', 'locomotion'};
        colorsAdjust = 2;  % offset used in author scripts
        % If colors_for_behaviors available, use it; otherwise generate a palette
        try
            % Require variable 'codes' in path; if absent, generate a palette large enough
            maxCode = max(cellfun(@(v) max(v(:)), allResults.svmID(~cellfun(@isempty, allResults.svmID))));
            if isempty(maxCode); maxCode = numel(behaviors); end
            colors = lines(maxCode + colorsAdjust + 2);
        catch
            colors = lines(20);
        end
end

% Provide bhv2ModelColors as a matrix for row-indexing within the plot script
bhv2ModelColors = colors;

% Provide opts struct with frameSize field for plot titles/filenames
opts = struct();
opts.frameSize = frameSize;

% dataSubset/dataSubsetLabel used by the plot script
dataSubset = transOrWithin;
switch lower(transOrWithin)
    case 'trans'
        dataSubsetLabel = 'transitions: Pre';
    case 'transpost'
        dataSubsetLabel = 'transitions: Post';
    case 'within'
        dataSubsetLabel = 'within-behavior';
    otherwise
        dataSubsetLabel = 'all';
end

% ------------------------------- Ready to plot -------------------------------
fprintf('Loaded: %s\n', fullFilePath);
fprintf('Parameters: dataType=%s, kernel=%s, subset=%s, nDim=%d, bin=%.3f, nShuf=%d\n', ...
    dataType, kernelFunction, transOrWithin, nDim, frameSize, nShuffles);

% Call the plotting script (expects variables prepared above)
% svm_decoding_compare_plot


