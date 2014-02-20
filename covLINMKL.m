function K_out = covLINMKL(hyp, K, i)

% Linear multiple kernel learning (MKL) covariance function. The covariance
% different to native GPML covariance functions. Rather than taking vectors
% of data as input, it takes a cell array of n kernels (K) K_1, K_2 ... K_n 
% which must all have the same size. These kernels must be calculated 
% before passing them to the function and can be based on whatever 
% similarity measure is desired (linear, RBF, polynomial etc). A final
% kernel is contructed as a weighted sum of the kernels in K, with weights 
% alpha and bias term beta as the hyperparameters stored in hyp.cov. 
% Derivatives are taken from covScale for alphas and covConst for beta.
%
% K_sum = alpha_1 * K_1 + alpha_2 * K_2 + ... + alpha_n * K_n + beta
%
% hyperparameters are:
%
% hyp = [ log(sqrt(alpha_1))
%         log(sqrt(alpha_2)
%          .
%         log(sqrt(alpha_n)
%         log(sqrt(beta)) ]
%


if nargin<2, K_out = 'n_K + 1'; return; end                  % report number of parameters
if nargin<3, i = []; end% make sure, i exists

% get the number of kernels
n_K = numel(K);
size(K,1);
% Checks
% check all kernels are the same size
[rows, columns] =  cellfun(@size, K);
rows = unique(rows);
columns = unique(columns);
% if we have more than one width or length, give an error
if length(rows) > 1 || length(columns) > 1
    disp('all kernels must have the same size!');
    return;
end

% Check the number of hyperparameters is one more than the number of kernels
if ~isempty(hyp) && length(hyp) - n_K ~= 1
    disp('Wrong number of hyperparameters. There should be one more hyperparameter than there are kernels.');
    return;
end

% If the hyperparameter vector is empty, share the weights equally among
% the n_K kernels and set the bias to 1
if isempty(hyp)
    hyp(1:n_K) = log(sqrt(1/n_K));
    hyp(n_K + 1) = 0;
end
%tic
% reshape K into a stack of 2d arrays, each of which is one of the original
% sub-kernels
K_stack = reshape(cell2mat(K),rows,columns,n_K);

% add a layer of ones to the stack
K_stack(:,:,n_K+1) = ones(rows,columns);

% convert the vector of hyperparameters to a vector of weights
wts_vector = exp(2 * hyp);

% convert the vector of weights into a stack of n_k+1 2d arrays, each of 
% which is a rows by colums set of the weight in that layer
wts_stack = repmat(reshape(wts_vector,1,1,n_K+1),[rows,columns,1]);

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
  % subkernels, return the derivative wrt the hyperparameter for the weight
  % on that subkernel
  if isnumeric(i) && i <= n_K
      K_out = 2 * wts_vector(i) * K{i};
  end
  % if i is numeric and less than or equal to the number of
  % subkernels, return the derivative wrt the hyperparameter bias
  % (constant) term
  if isnumeric(i) && i == n_K + 1
      K_out = 2 * wts_vector(i) * ones(rows,columns);
  end
end

if nargin>3                                                        % derivatives
  error('Unknown hyperparameter')
end