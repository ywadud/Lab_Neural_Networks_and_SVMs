%%% 1. Clear variables and close all figures
clearvars;
close all;

%%% 2. Define input training examples
X = [0 1; 1 1; 1 0; 0 0];
y = [1;0;1;0];

%%% 3. Train a non-linear SVM classifier
%%% PLACE YOUR CODE HERE
svm_lin = fitcsvm(X, y, 'ClassNames', [0 1]);
svm_poly = fitcsvm(X, y, 'ClassNames', [0 1], 'KernelFunction', 'polynomial');
svm_gauss = fitcsvm(X,y,'ClassNames', [0,1],'KernelFunction', 'gaussian'); 

%%% 4. Compute decision regions
%%% PLACE YOUR CODE HERE
%%% Hint: Use plot_XOR_and_regions as inspiration
X = [0 1; 1 1; 1 0; 0 0];
[X1,X2] = meshgrid(0:0.001:1, 0:0.001:1);
X1 = X1(:);
X2 = X2(:);
XVALUE = [X1 X2];
pred_lin = predict(svm_lin, XVALUE);
pred_poly = predict(svm_poly, XVALUE);
pred_gauss = predict(svm_gauss, XVALUE);

%%%% Linear
figure; 
hold on;
%Plot the actual training examples
plot([X(1,1) X(3,1)], [X(1,2) X(3,2)], 'rx', 'MarkerSize', 16);
hold on
plot([X(2,1) X(4,1)], [X(2,2) X(4,2)], 'bo', 'MarkerSize', 16);
hold on
%Plot decision regions
gscatter(X1, X2, pred_lin, [0.85 0.325 0.098; 0.9290 0.6940 0.1250]);
axis tight;

%%%% Polynomial
figure; 
hold on;
%Plot the actual training examples
plot([X(1,1) X(3,1)], [X(1,2) X(3,2)], 'rx', 'MarkerSize', 16);
hold on
plot([X(2,1) X(4,1)], [X(2,2) X(4,2)], 'bo', 'MarkerSize', 16);
hold on
%Plot decision regions
gscatter(X1, X2, pred_poly, [0.85 0.325 0.098; 0.9290 0.6940 0.1250]);
axis tight;

%%%% Gaussian
figure; 
hold on;
%Plot the actual training examples
plot([X(1,1) X(3,1)], [X(1,2) X(3,2)], 'rx', 'MarkerSize', 16);
hold on
plot([X(2,1) X(4,1)], [X(2,2) X(4,2)], 'bo', 'MarkerSize', 16);
hold on
%Plot decision regions
gscatter(X1, X2, pred_gauss, [0.85 0.325 0.098; 0.9290 0.6940 0.1250]);
axis tight;