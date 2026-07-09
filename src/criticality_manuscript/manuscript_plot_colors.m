function colors = manuscript_plot_colors()
% MANUSCRIPT_PLOT_COLORS - Shared colors for criticality manuscript figures
%
% Goal:
%   Return struct fields for data/shuffled scatters and histograms, plus
%   engagement-class RGB rows (total, engaged, non-engaged).

colors = struct();
colors.data = [0.2, 0.35, 0.75];
colors.shuffled = [0.55, 0.55, 0.55];
colors.refLine = [0.35, 0.35, 0.35];
colors.trendLine = [0, 0, 0];
colors.engagementClasses = [...
  0.45, 0.45, 0.45;
  0.15, 0.45, 0.75;
  0.85, 0.35, 0.15];
end
