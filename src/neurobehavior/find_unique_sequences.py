import pandas as pd
import numpy as np

def find_unique_sequences(dataBhv, nSeq, validInd):
    if len(validInd) != nSeq:
        raise ValueError('validInd needs to be a 1XnSeq logical to determine which index/indices to ensure are "valid"')
    
    allSequences = []
    allIndices = []
    
    # Iterate through ID to extract sequences of length nSeq
    for i in range(len(dataBhv) - nSeq + 1):
        currentSeq = dataBhv['ID'][i:i+nSeq].to_list()  # Extract current sequence of length nSeq
        
        # Skip any sequences that have NaNs
        if any(pd.isna(currentSeq)):
            continue
        
        # Only count the sequence if it's relevant behavior(s) is/are valid.
        if any(validInd[j] and not dataBhv['Valid'].iloc[i+j] for j in range(len(validInd))):
            continue
        
        seqStr = ','.join(map(str, currentSeq))  # Convert to string for comparison
        
        # Check if this sequence is already recorded
        try:
            seqIndex = allSequences.index(seqStr)
            allIndices[seqIndex].append(i)  # Existing sequence, append the new starting index
        except ValueError:
            allSequences.append(seqStr)  # New sequence found, add it to the list
            allIndices.append([i])  # Initialize with the current starting index
    
    # Convert sequences back to original format
    uniqueSequences = [list(map(int, seq.split(','))) for seq in allSequences]
    
    # Sort by frequency (descending)
    sorted_indices = sorted(range(len(allIndices)), key=lambda k: len(allIndices[k]), reverse=True)
    uniqueSequences = [uniqueSequences[i] for i in sorted_indices]
    sequenceIndices = [allIndices[i] for i in sorted_indices]
    
    return uniqueSequences, sequenceIndices

# # Example usage
# dataBhv = pd.DataFrame({'ID': [1, 2, 3, 4, 1, 2, 3, 4, 5], 'Valid': [True, False, True, True, True, True, True, False, True]})
# nSeq = 3
# validInd = [True, False, True]

# uniqueSequences, sequenceIndices = find_unique_sequences(dataBhv, nSeq, validInd)
# print("Unique Sequences:", uniqueSequences)
# print("Sequence Indices:", sequenceIndices)
