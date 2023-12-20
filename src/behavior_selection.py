import numpy as np

def behavior_selection(data, opts):
    behaviors = opts.behaviors
    codes = opts.bhvCodes

    # validBhv = [None] * len(behaviors)
    validBhv = np.zeros((data.shape[0], len(behaviors)))

    for i in range(len(codes)):
        if codes[i] in opts.validCodes:
            iAct = codes[i]
            actIdx = data.bhvID == iAct  # All instances labeled as this behavior
            allPossible = sum(actIdx)

            longEnough = data.bhvDur >= opts.minActTime  # Only use if it lasted long enough to count

            actAndLong = actIdx & longEnough
            andLongEnough = sum(actAndLong)  # for printing sanity check report below

            # iPossible is a list of behavior indices for this behavior that is
            # at least long enough
            # Go through possible instances and discard unusable (repeated) ones
            for iPossible in np.where(actAndLong)[0]:
                # Was there the same behavior within the last minNoRepeat sec?
                # endTime = np.concatenate([data.bhvStartTime[1:], [data.bhvStartTime[-1] + data.bhvDur[-1]]])
                endTime = np.concatenate([data['bhvStartTime'].iloc[1:], [data['bhvStartTime'].iloc[-1] + data['bhvDur'].iloc[-1]]])
                # possible repeated behaviors are any behaviors that came
                # before this one that were within the no-repeat minimal time
                iPossRepeat = (endTime < data.bhvStartTime[iPossible]) & (endTime >= (data.bhvStartTime[iPossible] - opts.minNoRepeatTime))

                # If it's within minNoRepeat and any of the behaviors during that time are the same as this one (this behavior is a repeat), get rid of it
                if np.sum(iPossRepeat) and any(data.bhvID[iPossRepeat] == iAct):
                    actAndLong[iPossible] = False

            andNotRepeated = sum(actAndLong)

            print(f'Behavior {codes[i]}: {behaviors[i]}')
            print(f'{allPossible}: allPossible')
            print(f'{andLongEnough}: andLongEnough')
            print(f'{andNotRepeated}: andNotRepeated')
            print(f'Percent valid: {100 * andNotRepeated / allPossible:.1f}\n')

            if sum(actAndLong) >= opts.minBhvNum:
                validBhv[:,i] = actAndLong
            else:
                validBhv[:,i] = np.full(len(actAndLong), False)
                print(f'Not enough {behaviors[i]} bouts to analyze ({sum(actAndLong)} of {opts.minBhvNum} needed)\n')

        else:
            validBhv[:,i] = np.full(len(data), False)
            print(f'{behaviors[i]} code {codes[i]} is not a valid behavior for this analysis\n\n')

    return validBhv
