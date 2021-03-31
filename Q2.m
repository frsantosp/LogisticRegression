%inspired by https://www.mathworks.com/help/stats/perfcurve.html
M = load('ad_data.mat');
train = M.X_train;
y_train = M.y_train;

test = M.X_test;
y_test =  M.y_test;

par= [0.00000000001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
auc_list = [0,0,0,0,0,0,0,0,0,0];
num_of_features = [0,0,0,0,0,0,0,0,0,0];
for i=1:length(par)
    [w,c] = logistic_l1_train(train,y_train,par(i));

    scores =test*w + c;
    [X,Y,T,AUC]=perfcurve(y_test,scores,1);
    resp = sum(w~=0);  %num of non zero vals      
    %resp1 = sum(w'<0);
    %respT = resp+ resp1;
    %disp(resp);
    %figure;
    %plot(X,Y);
    %disp(AUC);
    auc_list(i) = AUC;
    num_of_features(i) = resp;
end
figure;
plot(par, auc_list);
figure;

plot(par, num_of_features);

function [w, c] = logistic_l1_train(data, labels, par)
% OUTPUT w is equivalent to the first d dimension of weights in logistic train
% c is the bias term, equivalent to the last dimension in weights in logistic train.
% Specify the options (use without modification).
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations.
[w, c] = LogisticR(data, labels, par, opts);

end