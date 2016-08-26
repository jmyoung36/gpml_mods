function A = meanConst_K(hyp, K, i)

% Constant mean function for use with precomputed kernels. As by definition 
% a constant mean does not depend on the input data this can be used 
% without input data as an argument; however take kernel matrix K to get 
% length of mean vector or derivative vector. This is in fact
% mathematically identical to original meanConst but it is desirable to
% emphasise the use of this function for precomputed kernel problems.
%
% The mean function is parameterized as:
%
% m(K) = c
%
% The hyperparameter is:
%
% hyp = [ c ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-08-04.
%
% See also MEANFUNCTIONS.M.

if nargin<2, A = '1'; return; end             % report number of hyperparameters 
if numel(hyp)~=1, error('Exactly one hyperparameter needed.'), end
c = hyp;
if nargin==2
  A = c*ones(size(K{1},1),1);                                       % evaluate mean
else
  if i==1
    A = ones(size(K{1},1),1);                                          % derivative
  else
    A = zeros(size(K{1},1),1);
  end
end