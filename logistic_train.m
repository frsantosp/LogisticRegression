function [weights] = logistic_train(data, labels, epsilon, maxiter)
% code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix withn samples and d features, where
% column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"
%
    n = 57;                             % number of features
    l = size(labels);                   % size of sample
    w = zeros(n+1,1);                   % adds bias term
    data = [data ones(l(1),1)];         % adds bias 

    %calculates gradient
    for j=1:maxiter
        de = 0;
                                        % sum of the gradient
        for i=1:l(1)
            y= labels(i);
            x= data(i,:);
            s = sigmoid(x*w);
            de = de + ((y-s)*x');    

        end
        de = de*(1/l(1));               % multiplies gradient by number of samples
        w= w + epsilon*(-1*de);         % updates weights
    end
   [weights]=w;                         %returns weights
end

