%%% 1. Initial cleanup, add paths and load in data
%%% DON'T CHANGE
clearvars;
close all;
addpath('data');
addpath('helper');
load lab3cardata.mat;

%%% 2. Create helpful variables
mTrain = size(Xtrain, 1); % Total number of training examples
mTest = size(Xtest, 1); % Total number of test examples
n = 4; % Total number of classes

%%% 3. Train non-linear SVM classifiers - one vs all using the training data
% Create cell arrays to store each SVM classifier
% Use the Gaussian kernel function
%%% PLACE YOUR CODE HERE
svm = cell(1,4);
y1=zeros(mTrain,1);
y1(Ytrain==1)=1;

y2=zeros(mTrain,1);
y2(Ytrain==2)=1;

y3=zeros(mTrain,1);
y3(Ytrain==3)=1;

y4=zeros(mTrain,1);
y4(Ytrain==4)=1;

% SVMs Training
svm{1} = fitcsvm(Xtrain, y1, 'ClassNames', [0 1], 'KernelFunction', 'gaussian');
svm{2} = fitcsvm(Xtrain, y2, 'ClassNames', [0 1], 'KernelFunction', 'gaussian');
svm{3} = fitcsvm(Xtrain, y3, 'ClassNames', [0 1], 'KernelFunction', 'gaussian');
svm{4} = fitcsvm(Xtrain, y4, 'ClassNames', [0 1], 'KernelFunction', 'gaussian');

%%% 4. Perform One-Vs-All prediction on the training and test dataset
% Determine which class each of the examples in the test datasets are
% Create score matrices for both the training and test datasets
%%% PLACE YOUR CODE HERE
for i = 1 : 4
    [~,predtrain] = predict(svm{i}, Xtrain);
    pred_train(:,i) = predtrain(:,2);
end
[~,classtrain] = max(pred_train,[],2);
 
for i = 1 : 4
    [~,predtest] = predict(svm{i}, Xtest);
    pred_test(:,i) = predtest(:,2);
end
[~,classtest] = max(pred_test,[],2);

%%% 5. Calculate the classification accuracy for the training and test datasets
%%% PLACE YOUR CODE HERE
accuracy_train= mean(double(classtrain == Ytrain)) * 100;
accuracy_test= mean(double(classtest == Ytest)) * 100;
