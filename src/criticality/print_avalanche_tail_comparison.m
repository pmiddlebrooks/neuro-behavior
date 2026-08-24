function print_avalanche_tail_comparison(comparison, label)
% PRINT_AVALANCHE_TAIL_COMPARISON - Command-window summary of tail-model tests
%
% Variables:
%   comparison - Struct from compare_avalanche_tail_models
%   label      - Prefix such as 'Size' or 'Duration'

if nargin < 2 || isempty(label)
  label = 'Tail';
end
if nargin < 1 || isempty(comparison) || ~isstruct(comparison)
  fprintf('  %s tail models: not tested\n', label);
  return;
end

decision = 'notTested';
if isfield(comparison, 'decision') && ~isempty(comparison.decision)
  decision = comparison.decision;
end
fprintf('  %s tail-model decision: %s', label, decision);

nTail = nan;
decades = nan;
if isfield(comparison, 'nTail')
  nTail = comparison.nTail;
end
if isfield(comparison, 'decades')
  decades = comparison.decades;
end
if isfinite(nTail) || isfinite(decades)
  fprintf(' (nTail=%s, decades=%s)', format_optional_number(nTail, '%.0f'), ...
    format_optional_number(decades, '%.2f'));
end
fprintf('\n');

if isfield(comparison, 'decisionReason') && ~isempty(comparison.decisionReason)
  fprintf('    %s\n', comparison.decisionReason);
end

print_vuong_line('PL vs exponential', get_vuong(comparison, 'vuongVsExp'));
print_vuong_line('PL vs lognormal', get_vuong(comparison, 'vuongVsLognormal'));

if isfield(comparison, 'nestedTruncatedVsPl') && isstruct(comparison.nestedTruncatedVsPl)
  nested = comparison.nestedTruncatedVsPl;
  if isfield(nested, 'pValue') && isfinite(nested.pValue)
    fprintf('    Nested truncated-PL vs PL: LR=%s, p=%s\n', ...
      format_optional_number(nested.lrStat, '%.2f'), ...
      format_optional_number(nested.pValue, '%.3f'));
  end
end

aicPl = get_aic(comparison, 'powerLaw');
aicExp = get_aic(comparison, 'exponential');
aicLn = get_aic(comparison, 'lognormal');
aicCut = get_aic(comparison, 'truncatedPowerLaw');
if any(isfinite([aicPl, aicExp, aicLn, aicCut]))
  fprintf('    AIC  PL=%s  exp=%s  lognormal=%s  truncPL=%s  (lower is better)\n', ...
    format_optional_number(aicPl, '%.1f'), format_optional_number(aicExp, '%.1f'), ...
    format_optional_number(aicLn, '%.1f'), format_optional_number(aicCut, '%.1f'));
end
end

function vuong = get_vuong(comparison, fieldName)
vuong = struct('R', nan, 'pValue', nan);
if isfield(comparison, fieldName) && isstruct(comparison.(fieldName))
  vuong = comparison.(fieldName);
end
end

function aic = get_aic(comparison, fieldName)
aic = nan;
if isfield(comparison, fieldName) && isstruct(comparison.(fieldName)) ...
    && isfield(comparison.(fieldName), 'aic')
  aic = comparison.(fieldName).aic;
end
end

function print_vuong_line(pairName, vuong)
if ~isstruct(vuong) || ~isfield(vuong, 'pValue') || ~isfinite(vuong.pValue)
  return;
end
rVal = nan;
if isfield(vuong, 'R')
  rVal = vuong.R;
end
fprintf('    Vuong %s: R=%s, p=%s', pairName, ...
  format_optional_number(rVal, '%.2f'), format_optional_number(vuong.pValue, '%.3f'));
if isfield(vuong, 'preferPowerLaw') && vuong.preferPowerLaw
  fprintf(' (favor PL)');
elseif isfield(vuong, 'preferAlternative') && vuong.preferAlternative
  fprintf(' (favor alternative)');
else
  fprintf(' (inconclusive)');
end
fprintf('\n');
end

function textOut = format_optional_number(value, fmt)
if ~isfinite(value)
  textOut = 'nan';
else
  textOut = sprintf(fmt, value);
end
end
