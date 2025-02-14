function boutBinNumbers = compute_bout_bin_numbers(bhvID)
    % Computes a vector that tracks the bin number of each bout of every behavior.
    % 
    % INPUT:
    % - bhvID: Vector of behavior labels corresponding to each time bin.
    %
    % OUTPUT:
    % - boutBinNumbers: Vector of the same size as bhvID, where each entry 
    %   counts the bin number within each behavior bout.

    % Initialize output vector
    boutBinNumbers = zeros(size(bhvID));

    % Get the total number of time bins
    numBins = length(bhvID);

    % Initialize tracking variables
    currentBehavior = bhvID(1);
    binCount = 1;

    % Iterate through bhvID to track bin numbers
    for i = 1:numBins
        if i == 1
            % First bin starts at 1
            boutBinNumbers(i) = binCount;
        else
            if bhvID(i) == bhvID(i-1)
                % Continue counting within the same bout
                binCount = binCount + 1;
            else
                % New behavior or new bout starts, reset count
                binCount = 1;
                currentBehavior = bhvID(i);
            end
            boutBinNumbers(i) = binCount;
        end
    end
end
