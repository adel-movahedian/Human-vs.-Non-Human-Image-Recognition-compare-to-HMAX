%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PART 2: Dimensionality Reduction (PCA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

% 1. Load the dataset
load('hmaxdata.mat', 'XTrain', 'XTest', 'ytrain', 'ytest');
XTrain = XTrain';
XTest = XTest';

% 2. Apply PCA retaining 95% variance
[coeff, scoreTrain, ~, ~, explained] = pca(XTrain);

% Find the number of components needed to retain 95% variance
cumExplained = cumsum(explained);
numComponents = find(cumExplained >= 95, 1);

% Project training data onto principal components
XTrainPCA = scoreTrain(:, 1:numComponents);

% Transform test data using the same PCA projection
scoreTest = (XTest - mean(XTrain,1)) * coeff; 
XTestPCA = scoreTest(:, 1:numComponents);

fprintf('Number of original features: %d\n', size(XTrain,2));
fprintf('Number of PCA components (95%% variance): %d\n', numComponents);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. Retrain SVM and MLP on PCA-transformed data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SVM on PCA data
svmModelPCA = fitcsvm(XTrainPCA, ytrain, 'KernelFunction','linear', ...
    'ClassNames',[ -1, 1 ], ...
    'Standardize',true);

yPredTest_SVMpca = predict(svmModelPCA, XTestPCA);
testAccuracy_SVMpca = sum(yPredTest_SVMpca == ytest) / numel(ytest) * 100;

[~, scoreSVMpca] = predict(svmModelPCA, XTestPCA);
[svmX_PCA, svmY_PCA, ~, svmAUC_PCA] = perfcurve(ytest, scoreSVMpca(:,2), 1);

% MLP on PCA data
% Convert ytrain/ytest for MLP
ytrainMLP = zeros(2, length(ytrain));
for i = 1:length(ytrain)
    if ytrain(i) == 1
        ytrainMLP(:, i) = [1;0];
    else
        ytrainMLP(:, i) = [0;1];
    end
end

numHiddenNeurons = 10;
netPCA = patternnet(numHiddenNeurons);

netPCA.divideParam.trainRatio = 0.8;
netPCA.divideParam.valRatio   = 0.2;
netPCA.divideParam.testRatio  = 0;

[netPCA, ~] = train(netPCA, XTrainPCA', ytrainMLP);

yPredTestMLPpca = netPCA(XTestPCA');
[~, yPredTest_MLPpca_class] = max(yPredTestMLPpca, [], 1);
yPredTest_MLPpca_class(yPredTest_MLPpca_class == 1) = 1;
yPredTest_MLPpca_class(yPredTest_MLPpca_class == 2) = -1;
testAccuracy_MLPpca = sum(yPredTest_MLPpca_class' == ytest) / numel(ytest) * 100;

% For ROC/AUC
mlpScoresPCA = yPredTestMLPpca(1,:)';
[mlpX_PCA, mlpY_PCA, ~, mlpAUC_PCA] = perfcurve(ytest, mlpScoresPCA, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4. Compare Accuracy and AUC before and after PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('--- After PCA ---\n');
fprintf('SVM PCA-based Test Accuracy: %.2f%%, AUC: %.2f\n', testAccuracy_SVMpca, svmAUC_PCA);
fprintf('MLP PCA-based Test Accuracy: %.2f%%, AUC: %.2f\n', testAccuracy_MLPpca, mlpAUC_PCA);

% Plot ROC
figure('Name','ROC Curves After PCA');
plot(svmX_PCA, svmY_PCA, 'b-', 'LineWidth', 2); hold on;
plot(mlpX_PCA, mlpY_PCA, 'r-', 'LineWidth', 2);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC Curves for SVM and MLP After PCA');
legend(sprintf('SVM PCA (AUC=%.2f)', svmAUC_PCA), sprintf('MLP PCA (AUC=%.2f)', mlpAUC_PCA), 'Location','best');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5. Analyze the effect of PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Discuss:
% - How accuracy changed vs. original
% - Any speed-up due to reduced dimensionality
% - Possibly re-check noise/rotation experiments if desired
