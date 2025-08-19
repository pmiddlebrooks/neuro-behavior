function bhvID = get_reach_bhv_labels(fileName);


%%
paths = get_paths;

traces = load(fullfile(paths.dropPath, 'reach_data/Y4_06-Oct-2023 14_14_53_NIBEH.mat'));
dataR = load(fullfile(paths.dropPath, 'reach_data/Copy_of_Y4_100623_Spiketimes_idchan_BEH.mat'));

        