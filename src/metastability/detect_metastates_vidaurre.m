function communities = detect_metastates_vidaurre(P, verbose)
% DETECT_METASTATES_VIDAURRE
%   Detect metastates from a transition probability matrix using
%   Louvain community detection (Vidaurre et al. 2017 style).
%
% Usage:
%   communities = detect_metastates_vidaurre(P)
%   communities = detect_metastates_vidaurre(P, true)  % verbose printout
%
% Input:
%   P       - NxN transition probability matrix (rows=from, cols=to).
%             Rows should sum to 1 (row-stochastic). N = number of states.
%   verbose - optional boolean (default false). If true prints mapping:
%             "Metastate L: states [i j ...]".
%
% Output:
%   communities - vector of length N. communities(i) = integer label of the
%                 community that state i belongs to.
%
% Example:
%   communities = [1 1 2 2 3]
%   This vector has length 5, so there are 5 states (indices 1..5).
%     - state 1 -> label 1
%     - state 2 -> label 1
%     - state 3 -> label 2
%     - state 4 -> label 2
%     - state 5 -> label 3
%   Interpretation:
%     Metastate 1: states [1 2]
%     Metastate 2: states [3 4]
%     Metastate 3: states [5]
%
% Notes:
%   - Symmetrizes P as in Vidaurre: A = (P + P')/2
%   - Requires either Brain Connectivity Toolbox's community_louvain or
%     GenLouvain's genlouvain on a modularity matrix.
%   - If neither is installed the function errors.

if nargin < 2, verbose = false; end

% validate input
[nRows,nCols] = size(P);
if nRows ~= nCols
    error('Input matrix must be square.');
end
N = nRows;
if any(isnan(P(:))) || any(~isfinite(P(:)))
    error('P contains NaN or Inf values.');
end

% symmetrize and remove self-loops
A = (P + P') / 2;
A(logical(eye(N))) = 0;

if exist('genlouvain','file') == 2
    % construct Newman-Girvan modularity matrix B (undirected)
    k = sum(A,2);
    m = sum(k) / 2;
    if m == 0
        error('Graph has zero total weight.');
    end
    B = A - (k * k') / (2*m);
    % genlouvain returns community labels for modularity matrix B
    communities = genlouvain(sparse(B));
else
    error(['No community detection function found. ', ...
        'Install Brain Connectivity Toolbox (community_louvain) ', ...
        'or GenLouvain (genlouvain) and retry.']);
end

% ensure column vector
communities = communities(:);

% optional verbose mapping: list states per metastate label
if verbose
    labels = unique(communities,'stable');
    for li = 1:numel(labels)
        lab = labels(li);
        idx = find(communities == lab);
        fprintf('Metastate %d: states [%s]\n', lab, strtrim(sprintf('%d ', idx)));
    end
end
end
