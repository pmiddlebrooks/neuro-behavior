%%                     Load results for SVM decoding comparisons
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script loads results saved by svm_decoding_compare.m (per-area) or
% svm_decoding_compare_joint_area.m (conjoint), then prepares workspace variables
% for svm_decoding_compare_plot.m.

% ------------------------------ User options ---------------------------------
paths = get_paths;

% Where results were saved
savePath = fullfile(paths.dropPath, 'decoding');

% 'perArea' — svm_decoding_compare.m MAT; 'joint' — svm_decoding_compare_joint_area.m MAT
decodeResultsLayout = 'perArea';  % 'perArea' | 'joint'

% --- perArea filename knobs (ignored when decodeResultsLayout is 'joint')
dataType = 'spontaneous';  % 'spontaneous' or 'reach' (used only for colors/labels and filenames)
kernelFunction = 'polynomial';
transOrWithin = 'all';      % 'all' | 'trans' | 'transPost' | 'within'
nDim = 8;                   % latent dimension used when saving
frameSize = .15;            % frame/bin size used when saving
nShuffles = 2;              % number of permutations used when saving

% --- joint filename knobs — must match the run that produced the file (ignored for perArea)
jointAreasSlug = 'M56_DS';     % underscores between area names as saved
jointEmbedDim = 8;
jointNComponentsPerArea = 4;
jointNumAreas = 2;
jointDataSubset = 'all';     % subset string as in joint script save

% Plot flags (compatible with svm_decoding_compare_plot.m)
plotFullMap = 0;      % full time plots require bhvID which is not saved; keep 0 by default
plotModelData = 1;
plotComparisons = 1;
savePlotFlag = 1;


% ---------------------------- Load saved results ------------------------------
switch decodeResultsLayout
    case 'perArea'
        fileName = sprintf('svm_%s_decoding_compare_multi_area_%s_nDim%d_bin%.2f_nShuffles%d.mat', ...
            kernelFunction, transOrWithin, nDim, frameSize, nShuffles);
    case 'joint'
        fileName = sprintf('svm_%s_joint_area_%s_subset_%s_embed%d_nComp%d_nAreas%d_bin%.2f_nShuffles%d.mat', ...
            kernelFunction, jointAreasSlug, jointDataSubset, jointEmbedDim, jointNComponentsPerArea, ...
            jointNumAreas, frameSize, nShuffles);
    otherwise
        error('decodeResultsLayout must be ''perArea'' or ''joint''.');
end
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

% Prefer on-disk marker for conjoint vs per-area bundles
if isfield(allResults, 'parameters') && isfield(allResults.parameters, 'analysisLayout')
    decodeResultsLayout = allResults.parameters.analysisLayout;
elseif isfield(allResults, 'jointLatents')
    decodeResultsLayout = 'joint';
end

% ------------------------ Reconstruct plotting context ------------------------
% Areas and selection
if isfield(allResults, 'areas'); areas = allResults.areas; else; areas = {'M23','M56','DS','VS'}; end
if strcmp(decodeResultsLayout, 'joint') && isfield(allResults, 'areasToInclude')
    areasToTest = allResults.areasToInclude(:)';
elseif isfield(allResults, 'areasToTest')
    areasToTest = allResults.areasToTest(:)';
else
    areasToTest = 1:numel(areas);
end

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
        % Spontaneous defaults
        behaviors = {'investigate_1', 'investigate_2', 'investigate_3', ...
            'rear', 'dive_scrunch', 'paw_groom', 'face_groom_1', 'face_groom_2', ...
            'head_groom', 'contra_body_groom', 'ipsi_body groom', 'contra_itch', ...
            'ipsi_itch_1', 'contra_orient', 'ipsi_orient', 'locomotion'};
        colorsAdjust = 2;  % offset used in author scripts
        % If colors_for_behaviors available, use it; otherwise generate a palette
        try
            if strcmp(decodeResultsLayout, 'joint') && isnumeric(allResults.svmID)
                maxCode = max(allResults.svmID(:));
            elseif iscell(allResults.svmID)
                nonEmptyCells = ~cellfun(@isempty, allResults.svmID);
                if any(nonEmptyCells)
                    maxCode = max(cellfun(@(v) max(v(:)), allResults.svmID(nonEmptyCells)));
                else
                    maxCode = numel(behaviors);
                end
            else
                maxCode = numel(behaviors);
            end
            if isempty(maxCode); maxCode = numel(behaviors); end
            colors = lines(maxCode + colorsAdjust + 2);
        catch
            colors = lines(20);
        end
end

% Provide bhv2ModelColors as a matrix for row-indexing within the plot script
if strcmp(decodeResultsLayout, 'joint') && isfield(allResults, 'bhv2ModelColors') && ~isempty(allResults.bhv2ModelColors)
    bhv2ModelColors = allResults.bhv2ModelColors;
else
    bhv2ModelColors = colors;
end

jointLabelStr = '';
if strcmp(decodeResultsLayout, 'joint')
    jointLabelStr = strjoin(allResults.areas(areasToTest), '+');
end

% Provide opts struct with frameSize field for plot titles/filenames
opts = struct();
opts.frameSize = frameSize;

if isfield(allResults, 'parameters') && isfield(allResults.parameters, 'dataSubset')
    dataSubset = allResults.parameters.dataSubset;
else
    dataSubset = transOrWithin;
end
switch lower(dataSubset)
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
fprintf('Layout: %s | dataType=%s | kernel=%s | dataSubset=%s | nDim=%d | bin=%.3f | nShuf=%d\n', ...
    decodeResultsLayout, dataType, kernelFunction, dataSubset, nDim, frameSize, nShuffles);

% svm_decoding_compare_plot   % uncomment to plot from loaded MAT


