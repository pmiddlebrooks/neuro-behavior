function complexity = limpel_ziv_complexity(sequence, varargin)
% LEMPEL_ZIV_COMPLEXITY Calculate Lempel-Ziv complexity of a sequence
%
% Variables:
%   sequence - Input sequence (can be numeric vector or binary vector)
%   varargin - Optional name-value pairs:
%       'threshold' - Threshold for binarization (default: median)
%       'method' - Binarization method: 'median', 'mean', or 'value' (default: 'median')
%       'normalize' - Normalize by sequence length (default: false)
%
% Goal:
%   Calculate the Lempel-Ziv complexity of a sequence, which measures the
%   number of distinct substrings in the sequence. This is useful for
%   analyzing the complexity/randomness of neural activity patterns in
%   sliding window analyses.
%
% Returns:
%   complexity - Lempel-Ziv complexity value (scalar)
%
% Example usage:
%   % Binary sequence
%   seq = [1 0 1 0 1 1 0 0 1];
%   c = limpel_ziv_complexity(seq);
%
%   % Numeric sequence with median thresholding
%   seq = randn(100, 1);
%   c = limpel_ziv_complexity(seq, 'method', 'median');
%
%   % Numeric sequence with custom threshold
%   c = limpel_ziv_complexity(seq, 'method', 'value', 'threshold', 0.5);
%
% Based on algorithm from:
%   Lempel, A. & Ziv, J. (1976). On the Complexity of Finite Sequences.
%   IEEE Transactions on Information Theory, 22(1), 75-81.

    % Parse input arguments
    p = inputParser;
    addParameter(p, 'threshold', [], @isnumeric);
    addParameter(p, 'method', 'median', @(x) ismember(x, {'median', 'mean', 'value', 'binary'}));
    addParameter(p, 'normalize', false, @islogical);
    parse(p, varargin{:});
    
    threshold = p.Results.threshold;
    method = p.Results.method;
    normalize = p.Results.normalize;
    
    % Convert to row vector if needed
    if iscolumn(sequence)
        sequence = sequence';
    end
    
    % Check if sequence is already binary (handles both numeric and logical)
    isBinary = (islogical(sequence) && all(ismember(sequence, [false, true]))) || ...
               (isnumeric(sequence) && all(ismember(sequence, [0, 1])));
    
    % Convert to binary if needed
    if ~isBinary && ~strcmp(method, 'binary')
        if strcmp(method, 'median')
            if isempty(threshold)
                threshold = median(sequence);
            end
            binarySeq = double(sequence > threshold);
        elseif strcmp(method, 'mean')
            if isempty(threshold)
                threshold = mean(sequence);
            end
            binarySeq = double(sequence > threshold);
        elseif strcmp(method, 'value')
            if isempty(threshold)
                error('Threshold value must be specified when using ''value'' method');
            end
            binarySeq = double(sequence > threshold);
        else
            error('Unknown binarization method');
        end
    elseif isBinary
        binarySeq = double(sequence);
    else
        error('Input must be binary or numeric sequence');
    end
    
    % Ensure binary sequence is 0s and 1s
    binarySeq = double(binarySeq > 0);
    
    % Calculate Lempel-Ziv complexity using the standard algorithm
    complexity = calc_lz_complexity_core(binarySeq);
    
    % Normalize by sequence length if requested
    if normalize
        n = length(binarySeq);
        % Normalize by theoretical maximum (n / log2(n))
        complexity = complexity / (n / log2(n));
    end
end

function c = calc_lz_complexity_core(binarySeq)
% CALC_LZ_COMPLEXITY_CORE Core Lempel-Ziv complexity calculation
%
% Variables:
%   binarySeq - Binary sequence (vector of 0s and 1s)
%
% Goal:
%   Calculate Lempel-Ziv complexity by counting the number of distinct
%   substrings in the sequence using the exhaustive production process.
%   Algorithm based on Lempel & Ziv (1976): scan left to right, find
%   longest substring that appeared before current position. If found,
%   new substring is match + next character; otherwise, new substring
%   is just current character.
%
% Returns:
%   c - Complexity value (number of distinct substrings)

    n = length(binarySeq);
    
    % Handle edge cases
    if n == 0
        c = 0;
        return;
    elseif n == 1
        c = 1;
        return;
    end
    
    % Initialize complexity counter
    c = 1;  % First substring always exists (first character)
    
    % Current position in sequence
    i = 2;  % Start from second character
    
    while i <= n
        % Find the longest substring starting at i that has appeared
        % in positions 1 to i-1
        
        maxLen = 0;  % Length of longest matching substring
        
        if i > 1
            % Search for matching substrings in previous positions
            for startPos = 1:(i-1)
                % Try to match as many characters as possible
                matchLen = 0;
                while (startPos + matchLen < i) && ...
                      (i + matchLen <= n) && ...
                      (binarySeq(startPos + matchLen) == binarySeq(i + matchLen))
                    matchLen = matchLen + 1;
                end
                maxLen = max(maxLen, matchLen);
            end
        end
        
        % If we found a match and can extend it, the new substring is:
        % match + next character. Otherwise, the new substring is just
        % the current character.
        c = c + 1;
        
        if maxLen > 0 && (i + maxLen < n)
            % Found a match and can extend it by one character
            % Move past the matched substring plus the next character
            i = i + maxLen + 1;
        else
            % No match found or at end of sequence
            % Move past just the current character
            i = i + 1;
        end
    end
end

