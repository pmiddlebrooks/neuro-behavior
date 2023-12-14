import numpy as np

def neural_matrix(data, opts):
    durFrames = int(np.floor(np.sum(data.bhvDur) / opts.frameSize))

    if 'id' in data.ci.columns:
        idLabels = data.ci.id[opts.useNeurons]
    else:
        idLabels = data.ci.cluster_id[opts.useNeurons]

    # Preallocate data matrix
    dataMat = np.zeros((durFrames, len(opts.useNeurons)))

    areaLabels = []

    # Loop through each neuron and put spike numbers within each frame
    for i in range(len(idLabels)):
        iSpikeTimes = data.spikeTimes[data.spikeClusters == idLabels[i]]

        # Define the time interval (in seconds)
        interval = opts.frameSize

        # Initialize a vector to store the counts
        iSpikeCount = np.zeros(dataMat.shape[0])

        # Count the number of time stamps (spikes) within each interval
        for j in range(dataMat.shape[0]):
            lowerBound = ((j - 1) * interval) + (interval * opts.shiftAlignFactor)
            upperBound = (j * interval) + (interval * opts.shiftAlignFactor)
            iSpikeCount[j] = np.sum((iSpikeTimes >= lowerBound) & (iSpikeTimes < upperBound))

        dataMat[:, i] = iSpikeCount

        # Keep track of which neurons (columns) are in which brain area
        areaLabels.append(data.ci.area[opts.useNeurons[i]])

    # Remove neurons that are out of min-max firing rates
    rmvNeurons = []

    if opts.removeSome:
        checkTime = 5 * 60
        checkFrames = int(checkTime / opts.frameSize)

        meanStart = np.sum(dataMat[:checkFrames, :], axis=0) / checkTime
        meanEnd = np.sum(dataMat[-checkFrames:, :], axis=0) / checkTime

        keepStart = (meanStart >= opts.minFiringRate) & (meanStart <= opts.maxFiringRate)
        keepEnd = (meanEnd >= opts.minFiringRate) & (meanEnd <= opts.maxFiringRate)

        rmvNeurons = ~(keepStart & keepEnd)
        print(f'\nKeeping {np.sum(~rmvNeurons)} of {len(rmvNeurons)} neurons')

        dataMat = dataMat[:, ~rmvNeurons]
        idLabels = [idLabels[i] for i in range(len(idLabels)) if not rmvNeurons[i]]
        areaLabels = [areaLabels[i] for i in range(len(areaLabels)) if not rmvNeurons[i]]

    return dataMat, idLabels, areaLabels, rmvNeurons
