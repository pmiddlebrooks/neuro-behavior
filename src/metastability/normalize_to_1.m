function out = normalize_to_1(mat, dim)
% NORMALIZE Normalize matrix along given dimension
if nargin < 2
    dim = 1;
end
sums = sum(mat, dim);
out = bsxfun(@rdivide, mat, sums);
out(isnan(out)) = 1 / size(mat,dim);
end
