%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PART 3: Exploring Confidence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

% 1. Load data
load('hmaxdata.mat', 'XTrain', 'XTest', 'ytrain', 'ytest');
XTrain = XTrain'; 
XTest = XTest';

% 2. Train an SVM with posterior probabilities
svmModel = fitcsvm(XTrain, ytrain, 'KernelFunction','linear','Standardize',true);
% Fit posterior
svmModelPosterior = fitSVMPosterior(svmModel, XTrain, ytrain);

[~, svmPosterior_train] = predict(svmModelPosterior, XTrain);
[~, svmPosterior_test]  = predict(svmModelPosterior, XTest);

% Posterior probabilities: columns correspond to classes -1 (first) and +1 (second)
p_animal_SVM = svmPosterior_test(:,2);     % Probability of +1 (animal)
p_nonanimal_SVM = svmPosterior_test(:,1);  % Probability of -1 (non-animal)

% Confidence
confidence_SVM = abs(p_animal_SVM - p_nonanimal_SVM);

% 3. Train MLP and get probability outputs
% Convert labels for MLP
ytrainMLP = zeros(2, length(ytrain));
for i = 1:length(ytrain)
    if ytrain(i) == 1
        ytrainMLP(:, i) = [1;0];
    else
        ytrainMLP(:, i) = [0;1];
    end
end

net = patternnet(10);
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio   = 0.2;
net.divideParam.testRatio  = 0;
net = train(net, XTrain', ytrainMLP);

% Probability outputs for MLP
yMLP_test = net(XTest');
p_animal_MLP = yMLP_test(1,:)';     % Probability for class +1
p_nonanimal_MLP = yMLP_test(2,:)';  % Probability for class -1
confidence_MLP = abs(p_animal_MLP - p_nonanimal_MLP);

% 4. Compute and plot mean confidence per category
animal_idx = (ytest == 1);
nonanimal_idx = (ytest == -1);

meanConf_animal_SVM = mean(confidence_SVM(animal_idx));
meanConf_nonanimal_SVM = mean(confidence_SVM(nonanimal_idx));

meanConf_animal_MLP = mean(confidence_MLP(animal_idx));
meanConf_nonanimal_MLP = mean(confidence_MLP(nonanimal_idx));

figure('Name','Mean Confidence');
barData = [meanConf_animal_SVM, meanConf_nonanimal_SVM; meanConf_animal_MLP, meanConf_nonanimal_MLP];
bar(barData);
ylabel('Mean Confidence');
set(gca,'XTickLabel', {'SVM','MLP'});
legend({'Animal','Non-Animal'}, 'Location','best');
title('Mean Confidence Per Category');
grid on;

% 5. (Optional) Apply PCA and recalculate confidence
[coeff, scoreTrain, ~, ~, explained] = pca(XTrain);
cumExplained = cumsum(explained);
numComponents = find(cumExplained >= 95,1);
XTrainPCA = scoreTrain(:,1:numComponents);
scoreTest = (XTest - mean(XTrain,1))*coeff;
XTestPCA = scoreTest(:,1:numComponents);

% Retrain SVM with posterior on PCA data
svmModelPCA = fitcsvm(XTrainPCA, ytrain, 'KernelFunction','linear','Standardize',true);
svmModelPosteriorPCA = fitSVMPosterior(svmModelPCA, XTrainPCA, ytrain);
[~, svmPosterior_testPCA] = predict(svmModelPosteriorPCA, XTestPCA);
p_animal_SVMpca = svmPosterior_testPCA(:,2);
p_nonanimal_SVMpca = svmPosterior_testPCA(:,1);
confidence_SVMpca = abs(p_animal_SVMpca - p_nonanimal_SVMpca);

% MLP with PCA
ytrainMLP = zeros(2, length(ytrain));
for i = 1:length(ytrain)
    if ytrain(i) == 1
        ytrainMLP(:, i) = [1;0];
    else
        ytrainMLP(:, i) = [0;1];
    end
end

netPCA = patternnet(10);
netPCA.divideParam.trainRatio = 0.8;
netPCA.divideParam.valRatio   = 0.2;
netPCA.divideParam.testRatio  = 0;
netPCA = train(netPCA, XTrainPCA', ytrainMLP);

yMLP_testPCA = netPCA(XTestPCA');
p_animal_MLPpca = yMLP_testPCA(1,:)';
p_nonanimal_MLPpca = yMLP_testPCA(2,:)';
confidence_MLPpca = abs(p_animal_MLPpca - p_nonanimal_MLPpca);

% Compare mean confidence
meanConf_animal_SVMpca = mean(confidence_SVMpca(animal_idx));
meanConf_nonanimal_SVMpca = mean(confidence_SVMpca(nonanimal_idx));
meanConf_animal_MLPpca = mean(confidence_MLPpca(animal_idx));
meanConf_nonanimal_MLPpca = mean(confidence_MLPpca(nonanimal_idx));

figure('Name','Mean Confidence After PCA');
barDataPCA = [meanConf_animal_SVMpca, meanConf_nonanimal_SVMpca; meanConf_animal_MLPpca, meanConf_nonanimal_MLPpca];
bar(barDataPCA);
ylabel('Mean Confidence');
set(gca, 'XTickLabel', {'SVM-PCA','MLP-PCA'});
legend({'Animal','Non-Animal'}, 'Location','best');
title('Mean Confidence Per Category After PCA');
grid on;

% 6. Bonus: Confidence under noise / rotation
% In practice, you would re-generate XTest_noisy, XTest_rotated, then compute
% p_animal / p_nonanimal, and re-evaluate confidence. 
