function K_out = covSEMKL(hyp, SD, i)

% Squared exponential/RBF multiple kernel learning (MKL) covariance function. The covariance
% different to native GPML covariance functions. Rather than taking vectors
% of data as input, it takes a cell array of n sqaured distance matrices (SD) SD_1, SD_2 ... SD_n 
% which must all have the same size. These SD matrices must be calculated 
% before passing them to the function and can be based on whatever 
% distance measure is desired (Euclidean, L1, Mahalanobis etc). A final
% kernel is contructed as a weighted sum of the exponentiated squared
% distances in SD, with the squared distances divided by ell^2 and then the
% exponentiated scaled squared distances weighted by alphas, and then with a
% final bias term beta. Alpha, ell(gamma) and bias term beta as the 
% hyperparameters stored in hyp.cov. 
% Derivatives are taken from covScale for alphas and covConst for beta.
%
% K_out = alpha_1 * exp((SD_1/ell_1^2)/2) + alpha_2 * exp((SD_2/ell_2^2)/2) + ... + alpha_n * exp((SD_n/ell_n^2)/2) + beta
%
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)  
%         ......
%         log(ell_n)
%         log(sqrt(alpha_1))
%         log(sqrt(alpha_2)
%          .....
%         log(sqrt(alpha_n)
%         log(sqrt(beta)) ]
%

if nargin<2, K_out = '(2*n_K + 1)'; return; end                  % report number of parameters
if nargin<3, i = []; end% make sure, i exists

% get the number of SD matrices
n_SD = numel(SD);
size(SD,1);
% Checks
% check all kernels are the same size
[rows, columns] =  cellfun(@size, SD);
rows = unique(rows);
columns = unique(columns);
% if we have more than one width or length, give an error
if length(rows) > 1 || length(columns) > 1
    disp('all kernels must have the same size!');
    return;
end

% Check the number of hyperparameters is one more than twice the number of kernels
if ~isempty(hyp) && length(hyp) - (2 * n_SD) ~= 1
    disp('Wrong number of hyperparameters. There should be one more hyperparameter than twice the number of squared distance matrices.');
    return;
end

% If the hyperparameter vector is empty, share the weights equally among
% the n_K kernels and set the bias to 1. Set the initial ell values to be
% the median of the corresponding squared distance matrices (see
% http://blog.smola.org/post/940859888/easy-kernel-width-choice)
if isempty(hyp)
    for hyp_ind = 1:n_SD;
       hyp(hyp_ind) = log(sqrt(median(median(SD{hyp_ind})))); 
    end  
    hyp(n_SD + 1:2*n_SD) = log(sqrt(1/n_SD)); 
    hyp((2 * n_SD)+1) = 0;
end
%tic

% split the hyperparameter vector into two: wts (alphas and beta) and ells
ells_vector = exp(hyp(1:n_SD));
wts_vector = exp(2 * hyp(n_SD+1:end));

% reshape SD into a stack of 2d arrays, each of which is one of the original
% squared distance matrices
SD_stack = reshape(cell2mat(SD),rows,columns,n_SD);

% convert the vector of ells into a stack of n_SD 2d arrays, each of 
% which is a rows by colums set of the ell in that layer
ells_stack = repmat(reshape(ells_vector,1,1,n_SD),[rows,columns,1]);

% convert the vector of weights into a stack of n_SD+1 2d arrays, each of 
% which is a rows by colums set of the weight in that layer
wts_stack = repmat(reshape(wts_vector,1,1,n_SD+1),[rows,columns,1]);

scaled_SD_stack = SD_stack./(ells_stack.^2);

% calculate the unweighted kernel stack
K_stack = exp(-scaled_SD_stack/2);

% add a layer of ones to the stack
K_stack(:,:,n_SD+1) = ones(rows,columns);

% calculate K_sum by element-wise multiplying K_stack and wts_stack and
% then summing the result in the 3rd dimension (across weighted original
% subkernels)
K_sum =  sum((wts_stack .* K_stack),3);
%toc

% if is empty, return K_sum
if isempty(i)
    K_out = K_sum;  
else
    % if i is 'diag', return the diagonal of K_sum
    if strcmp(i,'diag')
        if rows == columns
            K_out = diag(K_sum);
        else
            disp('Can return diagonal only for square kernels.');
            return;
        end
    end
    
    % if i is numeric and less than or equal to the number of
    % squared distance matrices, return the derivative wrt the ith value of
    % ell
    if isnumeric(i) && i <= n_SD
        %%%%%%%%%% derivative wrt ells - calculate according to covSEiso line
        % 42
      
        % index of wts corresponding to ith squared distance matrix
        %wts_ind = i + n_SD;
        wts_ind = i;
        K_out = wts_vector(wts_ind) .* K_stack(:,:,i) .* scaled_SD_stack(:,:,i);
      
    end
    
    % if i is numeric and more than the number of squared distance matrices
    % but less than or equal to twice the number of squared distance matrices, return the
    % derative wrt the (i - n_SD)th weight
    if isnumeric(i) && i > n_SD && i <= 2*n_SD
        %%%%%%%%%% derivative wrt wts - caculate according to covSEiso line
        % 44
      
        % index of SD_stack corresponding to ith hyperparameter
        SD_stack_ind = i - n_SD;
        K_out = 2 .* wts_vector(SD_stack_ind) .* K_stack(:,:,SD_stack_ind); 
    end
    
    % if i is numeric and equal to twice the number of squared distance
    % matrices, return the derivative wrt the hyperparameter bias
    % (constant) term
    if isnumeric(i) && i == (2 * n_SD) + 1
        K_out = 2 * wts_vector(n_SD + 1) * ones(rows,columns);
    end
    
end
  
if nargin>3                                                        % derivatives
  error('Unknown hyperparameter')
end