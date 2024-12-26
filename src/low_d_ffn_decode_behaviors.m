%% Get data
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 60 * 60; % seconds
opts.frameSize = .2;
% opts.shiftAlignFactor = .05; % I want spike counts xxx ms peri-behavior label

getDataType = 'spikes';
get_standard_data

[dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);
%% Which data to model
forDim = 8;
iDim = forDim;
idSelect = [idM23 idDS];
%% Get a low-D representation of dataMat
lowDModel = 'umap';
switch lowDModel
    case 'umap'
        min_dist = .02;
        spread = 1.3;
        % n_neighbors = [6 8 10 12 15];
        n_neighbors = 10;
        % fprintf('\n%s %s min_dist=%.2f spread=%.1f n_n=%d\n\n', selectFrom, fitType, min_dist(x), spread(y), n_neighbors(z));
        umapFrameSize = opts.frameSize;
        rng(1);
        if exist('/Users/paulmiddlebrooks/Projects/', 'dir')
            cd '/Users/paulmiddlebrooks/Projects/toolboxes/umapFileExchange (4.4)/umap/'
        else
            cd 'E:/Projects/toolboxes/umapFileExchange (4.4)/umap/'
        end
        % [modelData, ~, ~, ~] = run_umap(zscore(dataMat(:, idSelect)), 'n_components', iDim, 'randomize', false, 'verbose', 'none', ...
        %     'min_dist', min_dist(x), 'spread', spread(y), 'n_neighbors', n_neighbors(z));
        [modelData, ~, ~, ~] = run_umap(zscore(dataMat(:, idSelect)), 'n_components', iDim, 'randomize', true, 'verbose', 'none', ...
            'min_dist', min_dist, 'spread', spread, 'n_neighbors', n_neighbors);
        pause(4); close
    case 'tsne'
        exaggeration = 90;
        perplexity = 40;
        % fprintf('\n%s %s exagg=%d perplx=%d \n\n', selectFrom, fitType, exaggeration(x), perplexity(y));
        modelData = tsne(zscore(dataMat(:, idSelect)),'Exaggeration', exaggeration, 'Perplexity', perplexity, 'NumDimensions',iDim);
end

%% Which data to model:
preInd = [diff(bhvIDMat) ~= 0; 0]; % 1 frame prior to all behavior transitions

%% High-D neral matrix
modelData = zscore(dataMat(:, idSelect));

%%

%% within-bout
modelInd = ~preInd & ~[preInd(2:end); 0] & ~(bhvID == -1);

%% all data
modelInd = 1:length(bhvID);
modelID = bhvID;


%% Train and test the FFN
input_dim = size(modelData, 2);
output_dim = length(unique(modelID));
hidden_layers = [64 32 16];
activation = 'relu';
epochs = 20;
batch_size = 16;
learning_rate = .001;
max_trials = 30;

[bestModel, bestHistory, hyperparamResults] = build_and_train_ffn(modelData(modelInd,:), bhvID(modelInd),input_dim, output_dim, max_trials, epochs, batch_size);





