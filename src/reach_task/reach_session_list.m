function sessions = reach_session_list()
% REACH_SESSION_LIST - Reach task session names
%
% Returns:
%   sessions - Cell array of session name strings

sessions = {
  'AB2_28-Apr-2023 17_50_02_NeuroBeh'
  'AB2_01-May-2023 15_34_59_NeuroBeh'
  'AB2_30-May-2023 12_49_52_NeuroBeh'
    'AB6_27-Mar-2025 14_04_12_NeuroBeh'
  'AB6_02-Apr-2025 14_18_54_NeuroBeh'
  'AB19_09-Apr-2026 14_28_19_NeuroBeh'
  'AB19_31-Mar-2026 15_46_45_NeuroBeh'
  'AB21_06-Apr-2026 18_07_42_NeuroBeh'
    'Y4_06-Oct-2023 14_14_53_NeuroBeh'
  'Y12_20-Jan-2026 16_16_42_NeuroBeh'
  'Y16_23-Dec-2025 16_07_49_NeuroBeh'
  'Y16_31-Dec-2025 13_50_49_NeuroBeh'
  'Y15_27-Aug-2025 14_02_21_NeuroBeh'
    'Y15_28-Aug-2025 19_47_07_NeuroBeh'
    'Y17_20-Aug-2025 17_34_48_NeuroBeh'
    };

% Excluded (generally too few neurons for reliable population analyses):
%   AB2_28-Apr-2023 17_50_02_NeuroBeh
%   AB2_01-May-2023 15_34_59_NeuroBeh
%   AB2_30-May-2023 12_49_52_NeuroBeh
%   AB6_02-Apr-2025 14_18_54_NeuroBeh
%   Y15_27-Aug-2025 14_02_21_NeuroBeh

end
