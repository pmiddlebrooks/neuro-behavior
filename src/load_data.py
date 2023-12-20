import pandas as pd
import numpy as np

def load_data(opts, dataType):
    # Load a specific type of data (input: dataType) for analysis

    if dataType == 'behavior':
        # Behavioral data is stored with an assigned B-SOiD label every frame.
        data_full = pd.read_csv(opts.dataPath + '/' + opts.fileName)

        # Use a time window of recorded data
        # get_window = slice(int(1 + opts.fsBhv * opts.collectStart), int(opts.fsBhv * (opts.collectStart + opts.collectFor)))
        get_window = slice(int(opts.fsBhv * opts.collectStart), int(opts.fsBhv * (opts.collectStart + opts.collectFor)))
        # get_window = int(1 + opts.fsBhv * opts.collectStart), int(opts.fsBhv * (opts.collectStart + opts.collectFor))
        data_window = data_full.iloc[get_window, :]
        data_window.reset_index(drop=True, inplace=True)
        # data_window['Time'] = data_window['Time'] - data_window['Time'].iloc[0]
        # data_window.loc[:, 'Time'] = data_window['Time'] - data_window['Time'].iloc[0]
        # data_window.loc[:,'Time'] = data_window.loc[:, 'Time'] - data_window.iloc[0, 'Time']
        data_window.loc[:,'Time'] = data_window['Time'] - data_window['Time'].iloc[0]
        bhv_id = data_window['Code']

        change_bhv = np.concatenate([[0], np.diff(bhv_id)])
        change_bhv_idx = np.where(change_bhv)[0]

        data = pd.DataFrame()
        data['bhvDur'] = np.concatenate([np.diff(np.insert(data_window['Time'].iloc[change_bhv_idx].values, 0, 0)),
                                         [opts.collectFor - data_window['Time'].iloc[change_bhv_idx[-1]]]])
        # data['bhvID'] = np.concatenate([bhv_id.iloc[0], bhv_id.iloc[change_bhv_idx]])
        data['bhvID'] = np.concatenate([pd.Series(bhv_id.iloc[0]), bhv_id.iloc[change_bhv_idx]])
        data['bhvName'] = np.concatenate([pd.Series(data_window['Behavior'].iloc[0]), data_window['Behavior'].iloc[change_bhv_idx]])
        data['bhvStartTime'] = np.concatenate([pd.Series(0), data_window['Time'].iloc[change_bhv_idx]])
        
    elif dataType == 'neuron':
        file_name = 'cluster_info.tsv'
        # ci = pd.read_csv(opts.dataPath + file_name, delimiter='\t')
        ci = pd.read_csv(opts.dataPath + file_name, delimiter='\t')

        # some of the depth values aren't in order, so re-sort the data by depth
        ci = ci.sort_values('depth')

        # Reassign depth so 0 is most superficial (M23) and 3840 is deepest (VS)
        ci['depth'] = 3840 - ci['depth']

        # Flip the ci table so the "top" is M23 and "bottom" is DS
        ci = ci.iloc[::-1]

        # Brain area regions as a function of depth from the surface
        m23 = [0, 500]
        m56 = [501, 1240]
        cc = [1241, 1540]
        ds = [1541, 2700]
        vs = [2701, 3840]

        # area = np.full(len(ci['depth']), '')
        area = pd.DataFrame({'area': range(len(ci['depth']))})
        # area = [''] * len(ci['depth'])

        area[(ci['depth'] >= m23[0]) & (ci['depth'] <= m23[1])] = 'M23'
        area[(ci['depth'] >= m56[0]) & (ci['depth'] <= m56[1])] = 'M56'
        area[(ci['depth'] >= cc[0]) & (ci['depth'] <= cc[1])] = 'CC'
        area[(ci['depth'] >= ds[0]) & (ci['depth'] <= ds[1])] = 'DS'
        area[(ci['depth'] >= vs[0]) & (ci['depth'] <= vs[1])] = 'VS'

        ci['area'] = area

        spike_times = np.load(opts.dataPath + 'spike_times.npy') / opts.fsSpike
        spike_clusters = np.load(opts.dataPath + 'spike_clusters.npy').reshape(-1,1)

        # Return the requested window of data, formatted so the start time is zero
        data_window = (spike_times >= opts.collectStart) & (spike_times < (opts.collectStart + opts.collectFor))
        spike_times = spike_times[data_window] - opts.collectStart
        spike_clusters = spike_clusters[data_window]

        data = {'ci': ci, 'spikeTimes': spike_times, 'spikeClusters': spike_clusters}

    elif dataType == 'lfp':
        # dataNeuro =
        pass

    return data
