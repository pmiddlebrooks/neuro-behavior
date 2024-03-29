import pandas as pd
import numpy as np

def load_data(opts, dataType):
    # Load a specific type of data (input: dataType) for analysis

    if dataType == 'behavior':
        # Behavioral data is stored with an assigned B-SOiD label every frame.
        data_full = pd.read_csv(opts['dataPath'] + '/' + opts['fileName'])

        # Use a time window of recorded data
        # get_window = slice(int(1 + opts['fsBhv'] * opts['collectStart']), int(opts['fsBhv'] * (opts['collectStart'] + opts['collectFor'])))
        get_window = slice(int(opts['fsBhv'] * opts['collectStart']), int(opts['fsBhv'] * (opts['collectStart'] + opts['collectFor'])))
        # get_window = int(1 + opts['fsBhv'] * opts['collectStart']), int(opts['fsBhv'] * (opts['collectStart'] + opts['collectFor']))
        data_window = data_full.iloc[get_window, :].copy()
        data_window.reset_index(drop=True, inplace=True)
        data_window.loc[:,'Time'] = data_window.loc[:,'Time'] - data_window.loc[0,'Time']
        bhv_id = data_window['Code']

        change_bhv = np.concatenate([[0], np.diff(bhv_id)])
        change_bhv_idx = np.where(change_bhv)[0]

        data = pd.DataFrame()
        data['Dur'] = np.concatenate([np.diff(np.insert(data_window['Time'].iloc[change_bhv_idx].values, 0, 0)),
                                         [opts['collectFor'] - data_window['Time'].iloc[change_bhv_idx[-1]]]])
        # data['bhvID'] = np.concatenate([bhv_id.iloc[0], bhv_id.iloc[change_bhv_idx]])
        data['ID'] = np.concatenate([pd.Series(bhv_id.iloc[0]), bhv_id.iloc[change_bhv_idx]]).astype(int)
        data['Name'] = np.concatenate([pd.Series(data_window['Behavior'].iloc[0]), data_window['Behavior'].iloc[change_bhv_idx]])
        data['StartTime'] = np.concatenate([pd.Series(0), data_window['Time'].iloc[change_bhv_idx]])
        
        data['Valid'] = behavior_selection(data, opts)

    elif dataType == 'neuron':
        file_name = 'cluster_info.tsv'
        # ci = pd.read_csv(opts['dataPath'] + file_name, delimiter='\t')
        ci = pd.read_csv(opts['dataPath'] + file_name, delimiter='\t')

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

        spike_times = np.load(opts['dataPath'] + 'spike_times.npy') / opts['fsSpike']
        spike_clusters = np.load(opts['dataPath'] + 'spike_clusters.npy').reshape(-1,1)

        # Return the requested window of data, formatted so the start time is zero
        data_window = (spike_times >= opts['collectStart']) & (spike_times < (opts['collectStart'] + opts['collectFor']))
        spike_times = spike_times[data_window] - opts['collectStart']
        spike_clusters = spike_clusters[data_window]

        data = {'ci': ci, 'spikeTimes': spike_times, 'spikeClusters': spike_clusters}

    elif dataType == 'lfp':
        # dataNeuro =
        pass

    return data




def behavior_selection(data, opts):
    # behaviors = opts['behaviors']
    # codes = opts['bhvCodes']
    codes = np.unique(data.ID)
    behaviors = []  # behaviors: a Python list containing the behavior names
    for iBhv in range(len(codes)):
        # first_idx = np.where(dataBhv.ID == codes[iBhv])[0][0]
        firstIdx = np.where(data.ID == codes[iBhv])[0][0]
        behaviors.append(data.Name.iloc[firstIdx])


    # validBhv = [None] * len(behaviors)
    validBhv = np.zeros(data.shape[0], dtype=bool)

    for i in range(len(codes)):
        # if codes[i] in opts['validCodes']:
            iAct = codes[i]
            actIdx = data.ID == iAct  # All instances labeled as this behavior
            allPossible = sum(actIdx)

            longEnough = data.Dur >= opts['minActTime']  # Only use if it lasted long enough to count

            actAndLong = actIdx & longEnough
            andLongEnough = sum(actAndLong)  # for printing sanity check report below

            # iPossible is a list of behavior indices for this behavior that is
            # at least long enough
            # Go through possible instances and discard unusable (repeated) ones
            for iPossible in np.where(actAndLong)[0]:
                # Was there the same behavior within the last minNoRepeat sec?
                # endTime = np.concatenate([data.StartTime[1:], [data.StartTime[-1] + data.Dur[-1]]])
                endTime = np.concatenate([data.StartTime.iloc[1:], [data.StartTime.iloc[-1] + data.Dur.iloc[-1]]])
                # possible repeated behaviors are any behaviors that came
                # before this one that were within the no-repeat minimal time
                iPossRepeat = (endTime < data.StartTime[iPossible]) & (endTime >= (data.StartTime[iPossible] - opts['minNoRepeatTime']))

                # If it's within minNoRepeat and any of the behaviors during that time are the same as this one (this behavior is a repeat), get rid of it
                if np.sum(iPossRepeat) and any(data.ID[iPossRepeat] == iAct):
                    actAndLong[iPossible] = False

            andNotRepeated = sum(actAndLong)

            # print(f'Behavior {codes[i]}: {behaviors[i]}')
            # print(f'{allPossible}: allPossible')
            # print(f'{andLongEnough}: andLongEnough')
            # print(f'{andNotRepeated}: andNotRepeated')
            # print(f'Percent valid: {100 * andNotRepeated / allPossible:.1f}\n')
            print(f'{codes[i]}: {behaviors[i]}: Valid: {andNotRepeated} ({100 * andNotRepeated / allPossible:.1f})%\n')

            # if sum(actAndLong) >= opts['minBhvNum']:
            validBhv[actAndLong] = 1
            # else:
            #     validBhv[:,i] = np.full(len(actAndLong), False)
            #     print(f"Not enough {behaviors[i]} bouts to analyze ({sum(actAndLong)} of {opts['minBhvNum']} needed)\n")

        # else:
        #     validBhv[:,i] = np.full(len(data), False)
        #     print(f'{behaviors[i]} code {codes[i]} is not a valid behavior for this analysis\n\n')

    return validBhv
