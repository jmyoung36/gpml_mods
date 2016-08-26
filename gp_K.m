function [varargout] = gp_K(hyp, inf, cov, lik, mean, K_t, y, K_p, K_tp, ys)
% Gaussian Process inference and prediction. The gp function provides a
% flexible framework for Bayesian inference and prediction with Gaussian
% processes for scalar targets, i.e. both regression and binary
% classification. The prior is Gaussian process, defined through specification
% of its mean and covariance function. The likelihood function is also
% specified. Both the prior and the likelihood may have hyperparameters
% associated with them.
%
% This modified version takes precomputed kernels and means rather than raw input
% data and hyperparameters. As this assumes the hyperparameters have
% already been calculated and used to generate the kernels and means, it
% cannot (currently) be used for training. Using it for training will
% result in an error. Also, this must be used with a modified version of
% the inference method.
%
%   training: [nlZ dnlZ          ] = gp(hyp, inf, cov, lik, mean, K_t, y);CURRENTLY NOT IMPLEMENTED. Use for
%   prediction only.
% prediction: [ymu ys2 fmu fs2   ] = gp(hyp, inf, cov, lik, mean, K_t, y, K_p, K_tp);
%         or: [ymu ys2 fmu fs2 lp] = gp(hyp, inf, cov, lik, mean, K_t, y, K_p, K_tp, ys);
%
% where:
%
%   hyp      column vector of hyperparameters
%   inf      function specifying the inference method 
%   lik      likelihood function
%   mean     mean function (currently set to meanConst_K)
%   cov      prior covariance function
%   K_t      cell array of subkernels for self covariance of training data
%   K_p      cell array of subkernels for self covariance of prediction data
%   K_tp     cell array of subkernels for cross covariance of training and 
%            prediction data
%   y        column vector of length n of training targets
%   ys       column vector of length nn of test targets
%
%   nlZ      returned value of the negative log marginal likelihood
%   dnlZ     column vector of partial derivatives of the negative
%               log marginal likelihood w.r.t. each hyperparameter
%   ymu      column vector (of length ns) of predictive output means
%   ys2      column vector (of length ns) of predictive output variances
%   fmu      column vector (of length ns) of predictive latent means
%   fs2      column vector (of length ns) of predictive latent variances
%   lp       column vector (of length ns) of log predictive probabilities
%
%   post     struct representation of the (approximate) posterior
%            3rd output in training mode and 6th output in prediction mode
% 
% See also covFunctions.m, infMethods.m, likFunctions.m, meanFunctions.m.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18

if nargin<7 || nargin>10
  disp('Usage: [nlZ dnlZ          ] = gp(hyp, inf, cov, lik, mean, K_t, y); (Training not supported for precomputed kernels yet)')
  disp('   or: [ymu ys2 fmu fs2   ] = gp(hyp, inf, cov, lik, mean, K_t, y, K_p, K_tp);')
  disp('   or: [ymu ys2 fmu fs2 lp] = gp(hyp, inf, cov, lik, mean, K_t, y, K_p, K_tp ys);')
  return
end

if isempty(inf),  inf = @infExact; else                        % set default inf
  if iscell(inf), inf = inf{1}; end                      % cell input is allowed
  if ischar(inf), inf = str2func(inf); end        % convert into function handle
end
% if isempty(mean), mean = {@meanZero}; end                     % set default mean
% if ischar(mean) || isa(mean, 'function_handle'), mean = {mean}; end  % make cell
% if isempty(cov), error('Covariance function cannot be empty'); end  % no default
% if ischar(cov)  || isa(cov,  'function_handle'), cov  = {cov};  end  % make cell
% cov1 = cov{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
% if strcmp(cov1,'covFITC'); inf = @infFITC; end       % only one possible inf alg
% if isempty(lik),  lik = @likGauss; else                        % set default lik
%   if iscell(lik), lik = lik{1}; end                      % cell input is allowed
%   if ischar(lik), lik = str2func(lik); end        % convert into function handle
% end

% check we are using a precomputed kernel version of the inference
% function
inf_str = func2str(inf);
if ~strcmp(inf_str(end-1:end),'_K')
    error('Precomputed kernel version of inference functions must be used with gp_K. infEP becomes infEP_K, infExact becomes infExact_K &c...')
end

% check we are using a precomputed kernel version of the inference
% function
cov_str = func2str(cov);
if ~ (strcmp(cov_str,'covLINMKL') || strcmp(cov_str,'covSEMKL'))
    error('The precomputed kernel version of GPML is currently only compatible with the covLINMKL covariance function. Please use this or revert to using the original GPML.')
end

% get number of sukernels
n_K = numel(K_t);

% for the means, currently hard-code in a call to meanConst_K. Replace
% with call to mean function as argument when other mean functions
% become available. Could theoretically use any mean that is not a function
% of data in the input space.
mean = @meanConst_K;

if ~isfield(hyp,'mean'), hyp.mean = []; end      
if eval(feval(mean)) ~= numel(hyp.mean)
   error('Number of mean function hyperparameters disagree with mean function'); 
end

if ~isfield(hyp,'cov'), hyp.cov = []; end
if eval(feval(cov)) ~= numel(hyp.cov)
   error('Number of cov function hyperparameters disagree with cov function')
end

if ~isfield(hyp,'lik'), hyp.lik = []; end
if eval(feval(lik)) ~= numel(hyp.lik)
  error('Number of lik function hyperparameters disagree with lik function')
end

try                                                  % call the inference method
  % issue a warning if a classification likelihood is used in conjunction with
  % labels different from +1 and -1
  if strcmp(func2str(lik),'likErf') || strcmp(func2str(lik),'likLogistic')
    uy = unique(y);
    if any( uy~=+1 & uy~=-1 )
      warning('You attempt classification using labels different from {+1,-1}\n')
    end
  end
  if nargin>7   % compute marginal likelihood and its derivatives only if needed
    
    % check all 3 kernel cell arrays contain the same number of subkernels
    n_Kp = numel(K_p);
    n_Ktp = numel(K_tp);
    if n_K ~= n_Kp || n_K ~= n_Ktp
        error('Kernel cell arrays contain different numbers of subkernels.') ; 
    end  
      
    post = inf(hyp, cov, lik, mean, K_t, y);
    %post.alpha
    %post.L
    %post.sW
  else
    % training mode not supported yet - give an error message and exit
    % error('Use of gp_K for training not yet supported.');
    
    % change these when I get round to implementing training with
    % precomputed kernels
    
    % done now I think!!
    if nargout==1
       [post nlZ] = inf(hyp, cov, lik, mean, K_t, y); dnlZ = {};
    else
       [post nlZ dnlZ] = inf(hyp, cov, lik, mean, K_t, y);
    end
  end
catch
  msgstr = lasterr;
  if nargin > 7, error('Inference method failed [%s]', msgstr); 
  else 
    % again, sort this out when implementing training
    % error('Use of gp_K for training not yet supported.');
     warning('Inference method failed [%s] .. attempting to continue',msgstr)
     dnlZ = struct('cov',0*hyp.cov, 'mean',0*hyp.mean, 'lik',0*hyp.lik);
     varargout = {NaN, dnlZ}; return                    % continue with a warning
  end
end

if nargin==7                                     % if no test cases are provided
    % again, sort this out when implementing training. For the moment just
    % give a message and quit
    % error('Use of gp_K for training not yet supported.');
    
    % should be working now!
    varargout = {nlZ, dnlZ, post};    % report -log marg lik, derivatives and post
else
    % not training - implement this properly!
    % first of all check all subkernel cell arrays have the same number of
    % subkernels
    n_subs = [numel(K_t) numel(K_p) numel(K_tp)];
    if length(unique(n_subs))~=1
        error('Subkernel cell arrays K_t, K_p and K_tp do not all contain the same number of subkernels.');
    end   
  alpha = post.alpha; 
  L = post.L; 
  sW = post.sW;
  if issparse(alpha)                  % handle things for sparse representations
    nz = alpha ~= 0;                                % determine nonzero indices
    if issparse(L), L = full(L(nz,nz)); end      % convert L and sW if necessary
    if issparse(sW), sW = full(sW(nz)); end
  else nz = true(size(alpha)); end  % non-sparse representation
  if numel(L)==0                      % in case L is not provided, we compute it
    % generate K by evaluating the covariance function with the training self covariance and taking a subset of the results   
    K = feval(cov, hyp.cov, K_t);
    K = K(nz,nz);
    L = chol(eye(sum(nz))+sW*sW'.*K);
  end
  Ltril = all(all(tril(L,-1)==0));            % is L an upper triangular matrix?
  % we don't have xs any more, use K_p instead
  %ns = size(xs,1);                                       % number of data points
  ns = size(K_p{1},1);
  nperbatch = 1000;                       % number of data points per mini batch
  nact = 0;                       % number of already processed test data points
  ymu = zeros(ns,1); ys2 = ymu; fmu = ymu; fs2 = ymu; lp = ymu;   % allocate mem
  while nact<ns               % process minibatches of test cases to save memory
    id = (nact+1):min(nact+nperbatch,ns);               % data points to process
    % again, generate kss and Ks by evaluating cov with the K_p and K_tp
    % subkernel cell arrays respectively. Then take the correct subset.
    %kss = feval(cov{:}, hyp.cov, xs(id,:), 'diag');              % self-variance
    %Ks  = feval(cov{:}, hyp.cov, x(nz,:), xs(id,:));             % cross-covariances
    %tic
    kss = feval(cov, hyp.cov, K_p, 'diag');              % self-variance
    %toc
    %tic
    kss = kss(id);
    %toc
    
    Ks  = feval(cov, hyp.cov, K_tp);         % cross-covariances                                      % self-variance
    Ks  = Ks(nz,id);                                         % cross-covariances
     
    
    % generate ms with a call to the mean function which is currently
    % harded coded as being meanConst_K
    ms = feval(mean, hyp.mean, K_p);
    ms = ms(id);
    
    fmu(id) = ms + Ks'*full(alpha(nz));                       % predictive means
    if Ltril           % L is triangular => use Cholesky parameters (alpha,sW,L)
      V  = L'\(repmat(sW,1,length(id)).*Ks);
      fs2(id) = kss - sum(V.*V,1)';                       % predictive variances
    else                % L is not triangular => use alternative parametrisation
      fs2(id) = kss + sum(Ks_id.*(L*Ks_id),1)';                 % predictive variances
    end
    fs2(id) = max(fs2(id),0);   % remove numerical noise i.e. negative variances
    
    % one extra input for prediction now
    % if nargin<9
    if nargin<10
      [lp(id) ymu(id) ys2(id)] = lik(hyp.lik, [], fmu(id), fs2(id));
    else        
      [lp(id) ymu(id) ys2(id)] = lik(hyp.lik, ys(id), fmu(id), fs2(id));
    end
    nact = id(end);          % set counter to index of last processed data point
  end
  if nargin<6
    varargout = {ymu, ys2, fmu, fs2, [], post};        % assign output arguments
  else
    varargout = {ymu, ys2, fmu, fs2, lp, post};
  end
end
