%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PART 1: Classification Using Machine Learning Models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

% 1. Load the dataset
load('hmaxdata.mat', 'XTrain', 'XTest', 'ytrain', 'ytest');

% Transpose data if needed so that rows are observations and columns are features
% (Typical for fitcsvm and patternnet usage).
% Below we assume each row is an example, each column a feature.
XTrain = XTrain';   % Now size(XTrain) => [NumTrainSamples, NumFeatures]
XTest = XTest';

% Convert ytrain and ytest from +1/-1 to categorical or double if needed
% For SVM in MATLAB, numeric labels are acceptable; for patternnet, we might convert to [0 1] or [1 0] encoding.
% Here we keep them numeric for SVM. For MLP, we can handle them differently.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Train an SVM Classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create an SVM model. We'll do a simple linear SVM to start.
svmModel = fitcsvm(XTrain, ytrain, 'KernelFunction', 'linear', ...
    'ClassNames', [ -1, 1 ], ...
    'Standardize', true);

% Predict on the training set (to check training accuracy) - optional
yPredTrain_SVM = predict(svmModel, XTrain);
trainAccuracy_SVM = sum(yPredTrain_SVM == ytrain) / numel(ytrain) * 100;

% Predict on the test set
yPredTest_SVM = predict(svmModel, XTest);
testAccuracy_SVM = sum(yPredTest_SVM == ytest) / numel(ytest) * 100;

fprintf('--- SVM Classification Results ---\n');
fprintf('Training Accuracy: %.2f%%\n', trainAccuracy_SVM);
fprintf('Test Accuracy: %.2f%%\n', testAccuracy_SVM);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. Train an MLP Classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We will use a simple feedforward network (patternnet) with one hidden layer.

% Convert labels from +1/-1 to 1/0 for MLP or keep them as 1/2, etc.
% For demonstration, let's map +1 => [1; 0],  -1 => [0; 1] for a 2-class problem.

ytrainMLP = zeros(2, length(ytrain));
for i = 1:length(ytrain)
    if ytrain(i) == 1
        ytrainMLP(:, i) = [1; 0];
    else
        ytrainMLP(:, i) = [0; 1];
    end
end

ytestMLP = zeros(2, length(ytest));
for i = 1:length(ytest)
    if ytest(i) == 1
        ytestMLP(:, i) = [1; 0];
    else
        ytestMLP(:, i) = [0; 1];
    end
end

% Define MLP architecture
numHiddenNeurons = 10;
net = patternnet(numHiddenNeurons);

% For reproducibility
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio   = 0.2;
net.divideParam.testRatio  = 0;

% Train MLP
[net, tr] = train(net, XTrain', ytrainMLP);

% Evaluate on the training set
yPredTrainMLP = net(XTrain');
[~, yPredTrain_MLP_class] = max(yPredTrainMLP, [], 1);
% Convert 1 => +1, 2 => -1
yPredTrain_MLP_class(yPredTrain_MLP_class == 1) = 1;
yPredTrain_MLP_class(yPredTrain_MLP_class == 2) = -1;
trainAccuracy_MLP = sum(yPredTrain_MLP_class' == ytrain) / numel(ytrain) * 100;

% Evaluate on the test set
yPredTestMLP = net(XTest');
[~, yPredTest_MLP_class] = max(yPredTestMLP, [], 1);
yPredTest_MLP_class(yPredTest_MLP_class == 1) = 1;
yPredTest_MLP_class(yPredTest_MLP_class == 2) = -1;
testAccuracy_MLP = sum(yPredTest_MLP_class' == ytest) / numel(ytest) * 100;

fprintf('--- MLP Classification Results ---\n');
fprintf('Training Accuracy: %.2f%%\n', trainAccuracy_MLP);
fprintf('Test Accuracy: %.2f%%\n', testAccuracy_MLP);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4. Evaluate Performance (Confusion Matrix, ROC, AUC)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Confusion matrix for SVM
figure('Name','SVM Confusion Matrix');
confusionchart(ytest, yPredTest_SVM, 'Title','SVM Confusion Matrix');

% Confusion matrix for MLP
figure('Name','MLP Confusion Matrix');
confusionchart(ytest, yPredTest_MLP_class', 'Title','MLP Confusion Matrix');

% ROC and AUC for SVM and MLP
% We need predicted scores for each class to plot ROC in MATLAB.
% For SVM we can get the score from the decision function:
[~, scoreSVM] = predict(svmModel, XTest);

% The second column of scoreSVM is typically the positive-class score. 
% We consider the labels [ -1, +1 ]; the positive class is +1 (index 2):
[svmX, svmY, svmT, svmAUC] = perfcurve(ytest, scoreSVM(:,2), 1);

% For MLP, the net(...) returns probability-like outputs; pick the probability of class +1:
mlpScores = yPredTestMLP(1,:)';  % Probability associated with first row => class +1
[mlpX, mlpY, mlpT, mlpAUC] = perfcurve(ytest, mlpScores, 1);

% Plot ROC
figure('Name','ROC Curves');
plot(svmX, svmY, 'b-', 'LineWidth', 2); hold on;
plot(mlpX, mlpY, 'r-', 'LineWidth', 2);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC Curves for SVM and MLP');
legend(sprintf('SVM (AUC=%.2f)', svmAUC), sprintf('MLP (AUC=%.2f)', mlpAUC), 'Location','best');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5. Compare SVM vs. MLP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nComparison:\n');
fprintf('SVM Test Accuracy: %.2f%%, AUC: %.2f\n', testAccuracy_SVM, svmAUC);
fprintf('MLP Test Accuracy: %.2f%%, AUC: %.2f\n', testAccuracy_MLP, mlpAUC);

% Plot category-wise accuracy (if we have two categories: Animal vs Non-Animal)
% This requires separating the test samples by their true category.
animal_idx = (ytest == 1);
nonanimal_idx = (ytest == -1);

% SVM category-wise accuracy
svm_animal_acc = sum(yPredTest_SVM(animal_idx) == 1) / sum(animal_idx) * 100;
svm_nonanimal_acc = sum(yPredTest_SVM(nonanimal_idx) == -1) / sum(nonanimal_idx) * 100;

% MLP category-wise accuracy
mlp_animal_acc = sum(yPredTest_MLP_class(animal_idx)' == 1) / sum(animal_idx) * 100;
mlp_nonanimal_acc = sum(yPredTest_MLP_class(nonanimal_idx)' == -1) / sum(nonanimal_idx) * 100;

figure('Name','Category-wise Accuracy');
barData = [svm_animal_acc, svm_nonanimal_acc; mlp_animal_acc, mlp_nonanimal_acc];
bar(barData);
set(gca, 'XTickLabel', {'SVM','MLP'});
legend({'Animal','Non-Animal'}, 'Location','best');
ylabel('Accuracy (%)');
title('Category-wise Accuracy Comparison');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6. Bonus: Test Model Robustness (Noisy and Rotated Images)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Here, we assume XTest are the features for original images.
% To demonstrate, we will craft a scenario of "noisy" features or "rotated" images.
% If you want to re-extract features from physically rotated/noisy images, you
% should apply transformations to the raw images, then re-run HMAX. 
% But as a placeholder, let's demonstrate by artificially modifying XTest.
%
% This is just a demonstration. In a real scenario, you'd have the original
% images, apply imrotate or imnoise, then re-extract HMAX features.

% For demonstration, let's add Gaussian noise to XTest:
XTest_noisy = XTest + 0.05 * randn(size(XTest));

% Evaluate SVM and MLP on noisy data
yPredTest_SVM_noisy = predict(svmModel, XTest_noisy);
testAccuracy_SVM_noisy = sum(yPredTest_SVM_noisy == ytest) / numel(ytest) * 100;

yPredTest_MLP_noisy = net(XTest_noisy');
[~, yPredTest_MLP_noisy_class] = max(yPredTest_MLP_noisy, [], 1);
yPredTest_MLP_noisy_class(yPredTest_MLP_noisy_class == 1) = 1;
yPredTest_MLP_noisy_class(yPredTest_MLP_noisy_class == 2) = -1;
testAccuracy_MLP_noisy = sum(yPredTest_MLP_noisy_class' == ytest) / numel(ytest) * 100;

fprintf('\n--- Robustness to Noise ---\n');
fprintf('SVM Test Accuracy on Noisy Data: %.2f%%\n', testAccuracy_SVM_noisy);
fprintf('MLP Test Accuracy on Noisy Data: %.2f%%\n', testAccuracy_MLP_noisy);

% For rotation, in a real scenario, you would have raw images. Let's just pretend we
% have them and re-run HMAX or somehow approximate rotating these features.
% As an illustration, let's do a random "swap" of certain columns:
XTest_rotated = XTest(:, randperm(size(XTest,2))); % This is not a real rotation, but a placeholder

% Evaluate on 'rotated' (mock) data
yPredTest_SVM_rotated = predict(svmModel, XTest_rotated);
testAccuracy_SVM_rotated = sum(yPredTest_SVM_rotated == ytest) / numel(ytest) * 100;

yPredTest_MLP_rotated = net(XTest_rotated');
[~, yPredTest_MLP_rotated_class] = max(yPredTest_MLP_rotated, [], 1);
yPredTest_MLP_rotated_class(yPredTest_MLP_rotated_class == 1) = 1;
yPredTest_MLP_rotated_class(yPredTest_MLP_rotated_class == 2) = -1;
testAccuracy_MLP_rotated = sum(yPredTest_MLP_rotated_class' == ytest) / numel(ytest) * 100;

fprintf('\n--- Robustness to Rotation ---\n');
fprintf('SVM Test Accuracy on Rotated Data: %.2f%%\n', testAccuracy_SVM_rotated);
fprintf('MLP Test Accuracy on Rotated Data: %.2f%%\n', testAccuracy_MLP_rotated);

