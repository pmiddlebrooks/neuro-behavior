%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 60 * 60; % seconds
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
selectFrom = 'VS';
idSelect = idVS;

%%
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


    end
    sgtitle([selectFrom, ' PCA across behaviors'])



