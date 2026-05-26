function y = zeta(s, q)
% ZETA - Numeric Riemann / Hurwitz zeta for Clauset plfit (no Symbolic Toolbox)
%
%   zeta(s)     - Riemann zeta, sum k^{-s}, s > 1
%   zeta(s, q)  - Hurwitz zeta, sum (k+q)^{-s}, s > 1, q > 0

if nargin < 2 || isempty(q)
  y = arrayfun(@riemann_zeta_scalar, s);
else
  if ~isscalar(q) || q <= 0
    error('Hurwitz zeta requires scalar q > 0.');
  end
  y = arrayfun(@(si) hurwitz_zeta_scalar(si, q), s);
end
y = reshape(y, size(s));
end

function val = riemann_zeta_scalar(s)
if ~isfinite(s) || s <= 1
  val = nan;
  return;
end
nTerms = min(1e6, max(200, ceil(30 / (s - 1))));
k = (1:nTerms)';
val = sum(k .^ (-s));
end

function val = hurwitz_zeta_scalar(s, q)
if ~isfinite(s) || s <= 1 || q <= 0
  val = nan;
  return;
end
nTerms = min(1e6, max(200, ceil(30 / (s - 1))));
k = (0:nTerms - 1)';
val = sum((k + q) .^ (-s));
end
