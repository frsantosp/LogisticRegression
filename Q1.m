filename = 'data.txt';                 % read file into matrix
M = readmatrix(filename);

filename = fopen('labels.txt','r');    % read labels 
formatSpec = '%f';
L = fscanf(filename,formatSpec);

n = [200,500,800,1000, 1500, 2000];        %loops through n of samples for training
acc_list=[0,0,0,0,0,0];                    %list with accuracy 
for j=1:6
    train = M(1:n(j),:);                   % split matrix
    test = M(2001:4601,:);
    L_train =L(1:n(j),:);
    L_test = L(2001:4601,:);

                                           % states epsilon and maximeter and
                                           % calls function
    epsilon = 1e-5;
    maxiter = 1000;
    [weights] = logistic_train(train,L_train,epsilon,maxiter);

                                            %adds bias term to testing
    test = [test ones(2601,1)];
    z = -1*test*weights;                    %results
                                            
    correct = 0;                            %calculates the accuracy
    for i=1:2601
        if (z(i) > 0) && (L_test(i)== 1)    % if result > 0 and is one in testing labels
            correct= correct + 1;           %    then mark as correct
        elseif (z(i)<0) && (L_test(i)== 0)  % if result < 0 and is zero in testing labels
            correct= correct + 1;           %    then mark as correct
        end
    end

    acc = correct/2601;
    disp(acc);                              %displays accuracy
    acc_list(j)= acc;
end

plot(n, acc_list);
    