%%% 1. Clear all variables and close all figures
%%% DON'T CHANGE
clearvars;
close all;
addpath('helper');

%%% 2. Input training examples
%%% DON'T CHANGE
X = [0 1; 1 1; 1 0; 0 0];
y = [1;0;1;0];

%%% 3. Initialize weight matrices

%%% Number of input neurons
%%% DON'T CHANGE
input_neurons = 2;

%%% Number of hidden layer neurons
%%% This you can change
hidden_neurons = 8;

%%% Number of output layer neurons
%%% DON'T CHANGE
output_neurons = 1;

%%% DON'T CHANGE
% W1 is a 3 x X matrix - 2 + 1 input neurons, X hidden layer neurons
rng(123);
e_init_1 = sqrt(6) / sqrt(input_neurons + hidden_neurons);
W1 = 2*e_init_1*rand(input_neurons + 1,hidden_neurons) - e_init_1;

% W2 is a (X + 1) x 1 matrix - X + 1 hidden layer neurons, 1 output layer neuron
e_init_2 = sqrt(6) / sqrt(hidden_neurons + output_neurons);
W2 = 2*e_init_2*rand(hidden_neurons + 1,output_neurons) - e_init_1;

%%% 4. Repeat k times
%%% DON'T CHANGE
k = 150;

%%% 5. Some relevant variables
%%% DON'T CHANGE
m = size(X,1);
n = size(X,2);

%%% 6. Initialize cost array
%%% DON'T CHANGE
costs = zeros(k,1);

%%% 7. Set learning rate
%%% DON'T CHANGE
alpha = 5;

%%% 8. Implement Stochastic Gradient Descent
%%% PLACE YOUR CODE HERE
cost_v = zeros(151,1);
X = [ones(m,1) X];

for i1 = 1:k+1
    for i =1:4
        X0 = X(i,:)';
        S = W1'*X0;
        X_1 = [1; sigmoid(S)];     
        s2 = W2'*X_1;
        x_2 = sigmoid(s2);

        %dsigmoid
        dsig2 = (x_2-y(i)).*dsigmoid(s2);
        dsig1 = dsigmoid(S).*(W2(2:end,:)*dsig2);
        dsigw1 = X0*dsig1';
        dsigw2 = X_1*dsig2';

        %weights
        W1 = W1-alpha*dsigw1;
        W2 = W2-alpha*dsigw2;

        %cost
        cost_v (i1) = cost_v (i1) + (1/(2*m))*(x_2-y(i)).^2;
    end
end

%%% 9. Plot the XOR points as well as the decision regions
%%% PLACE YOUR CODE HERE
plot_XOR_and_regions(W1,W2);

%%% 10. Plot the cost per iteration
%%% PLACE YOUR CODE HERE
iteration = 0:1:150;
figure
plot (iteration, cost_v);
ylabel('Hidden Neuron Cost');
xlabel('Iteration');
