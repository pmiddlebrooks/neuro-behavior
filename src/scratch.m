% SaniOG_etal_2021_NatNeuro
% Modeling behaviorally relevant neural dynamics enabled by preferential subspace identification
%%
cd '/Users/paulmiddlebrooks/Projects/toolboxes/PSID/'
init
%%
opts = neuro_behavior_options;
get_standard_data
%%
y = dataMat;
% y = dataMat(:, idDS);
z = bhvIDMat;
%%
nx = 5;
n1 = 7;
i = 7;
idSys = PSID(y, z, nx, n1, i);

%% predict
[zPred, yPred, xPred] = PSIDPredict(idSys, y);


%%
R2 = evalPrediction(z, zPred, 'R2')