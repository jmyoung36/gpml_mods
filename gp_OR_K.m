function [varargout] = gp_OR_K(hyp, inf, mean, cov, lik,  K_t, y, K_p, K_tp, ys)
%A modified version of the gp.m by Carl Edward Rasmussen and Hannes
%Nickisch for ordinal regression using Gaussian processes with precomputed
% kernels.
%
% This modified version takes precomputed kernels rather than raw input
% data. As this means it does not have access to data in the data
% space, it assumes the hyperparameters must live in the kernel space.
% Therefore it cannot be used with kernel covariance functions which 
% violate this assumption, e.g. ARD kernels. It must be used with modified 
% versions of covariance functions and modified inference methods.
%
% Two modes are possible: training or prediction: if no test cases are
% supplied, then the negative log marginal likelihood and its partial
% derivatives w.r.t. the hyperparameters is computed; this mode is used to fit
% the hyperparameters. If test cases are given, then the test set predictive
% probabilities are returned. Usage:
%
%   training: [nlZ dnlZ          ] = gp_OR_K(hyp, inf, cov, lik, mean, K_t, y);
% prediction: [ymu ys2 fmu fs2   ] = gp_OR_K(hyp, inf, cov, lik, mean, K_t, y, K_p, K_tp);
%         or: [ymu ys2 fmu fs2 lp] = gp_OR_K(hyp, inf, cov, lik, mean, K_t, y, K_p, K_tp ys);
%
% where:
%
%   hyp         column vector of hyperparameters
%   inf         function specifying the inference method 
%   cov         prior covariance function (see below)
%   mean        prior mean function (currently set to meanConst_K)
%   lik         likelihood function
%   K_t         cell array of subkernels for self covariance of training data
%   K_p         cell array of subkernels for self covariance of prediction data
%   K_tp        cell array of subkernels for cross covariance of training and 
%               prediction data
%   y           column vector of length n of training targets
%   ys          column vector of length nn of test targets
%
%   nlZ         returned value of the negative log marginal likelihood
%   dnlZ        column vector of partial derivatives of the negative
%               log marginal likelihood w.r.t. each hyperparameter

%   do I want to keep these next two? Easily done outside of gp function, 
%   may not really belong here
%   zero_one_err the number of instances for which the test is assigned an
%                incorrect label divided by the total number of test
%                cases
%   y_pred       the class prediction assigned to a test case

%   p           the probability that a test case belongs to a particular
%               class
%   fmu         predictive latent means
%   fs2         predictive latent variances
%   
%
%   post        struct representation of the (approximate) posterior
%               3rd output in training mode and 6th output in prediction mode
% 
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18
%
% Modified by Orla Doyle May 2013 to ensure compatibility with ordinal
% regression
%
% Modified a second time by Jonathan Young Jan 2014 to also ensure
% compatibility with precomputed kernel convariance functions



if nargin<7 || nargin>10
  disp('Usage: [nlZ dnlZ          ] = gp(hyp, inf, mean, cov, lik, K_t, y);')
  disp('   or: [ymu ys2 fmu fs2   ] = gp(hyp, inf, mean, cov, lik, K_t, y, K_p, K_tp);')
  disp('   or: [ymu ys2 fmu fs2 lp] = gp(hyp, inf, mean, cov, lik, K_t, y, K_p, K_tp ys);')
  return
end

if isempty(inf),  inf = @infExact_K; else                        % set default inf
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
if ~strcmp(cov_str,'covLINMKL')
    error('The precomputed kernel version of ORGP is currently only compatible with the covLINMKL covariance function. Please use this or revert to using the original ORGP.')
end

% get number of subkernels
n_K = numel(K_t);

% for the means, currently hard-code in a call to meanConst_K. Replace
% with call to mean function as argument when other mean functions
% become available. Could theoretically use any mean that is not a function
% of data in the input space.
mean = @meanConst_K;

% similarly, hard-code likErf_OR_K as the likelihood function
lik = @likErf_OR_K;

% if training kernel K_t is wrong way around flip it. Kernels must be 1 by
% n cell arrays
if size(K_t,1) > size(K_t,2)
    K_t = K_t';
end

% check the hyp specification
if ~isfield(hyp,'mean'), hyp.mean = []; end      
if eval(feval(mean)) ~= numel(hyp.mean)
   error('Number of mean function hyperparameters disagree with mean function'); 
end

if ~isfield(hyp,'cov'), hyp.cov = []; end
if eval(feval(cov)) ~= numel(hyp.cov)
   error('Number of cov function hyperparameters disagree with cov function')
end

% if ~isfield(hyp,'lik'), hyp.lik = []; end
% eval(feval(lik))
% if eval(feval(lik)) ~= numel(hyp.lik)
%   error('Number of lik function hyperparameters disagree with lik function')
% end

try
% check all 3 kernel cell arrays contain the same number of subkernels
%     n_Kp = numel(K_p);
%     n_Ktp = numel(K_tp);
%     if n_K ~= n_Kp || n_K ~= n_Ktp
%         error('Kernel cell arrays contain different numbers of subkernels.') ; 
%     end 

% call the inference method

if nargin>7   % compute marginal likelihood and its derivatives only if needed
    %disp('nargin > 7');
    
    % kernels K_p and K_tp must exist. Take this opportunity to check them
    % too
    if size(K_p,1) > size(K_p,2)
        K_p = K_p';
    end
    if size(K_tp,1) > size(K_tp,2)
        K_tp = K_tp';
    end
    
    
    post = inf(hyp, mean, cov, lik, K_t, y);
else
    if nargout==1
        %disp('nargin <=7, nargout = 1');
        [post nlZ] = inf(hyp, mean, cov, lik, K_t, y); dnlZ = {};
%         post.alpha % debug
%         post.sW % debug
%         post.L % debug
    else
        %disp('nargin <=7, nargout ~= 1');
        [post nlZ dnlZ] = inf(hyp, mean, cov, lik, K_t, y);
    end
end
catch
     msgstr = lasterr;
  if nargin > 7, error('Inference method failed [%s]', msgstr); 
  else 
     warning('Inference method failed [%s] .. attempting to continue',msgstr)
     dnlZ = struct('cov',0*hyp.cov, 'mean',0*hyp.mean, 'lik',0*hyp.lik);
     varargout = {NaN, dnlZ}; return                    % continue with a warning
  end
end


if nargin==7                                    % if no test cases are provided
  varargout = {nlZ, dnlZ, post};    % report -log marg lik, derivatives and post
else
    % not training - implement this properly!
    % first of all check all subkernel cell arrays have the same number of
    % subkernels
    n_subs = [numel(K_t) numel(K_p) numel(K_tp)];
    if length(unique(n_subs))~=1
        error('Subkernel cell arrays K_t, K_p and K_tp do not all contain the same number of subkernels.');
    end
    
    % continue with calculations
    alpha = post.alpha; 
    L = post.L; 
    sW = post.sW; 
    
    % deal with sparse representations
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
    
    % number of prediction data points
    ns = size(K_p{1},1);
    nperbatch = 1000;                       % number of data points per mini batch
    nact = 0;                       % number of already processed test data points
  	ymu = zeros(ns,1); ys2 = ymu; fmu = ymu; fs2 = ymu; lp = ymu;   % allocate mem
    while nact<ns               % process minibatches of test cases to save memory - probably not necessary as normally ns is small...
        id = (nact+1):min(nact+nperbatch,ns);               % data points to process   
        % again, generate kss and Ks by evaluating cov with the K_p and K_tp
        % subkernel cell arrays respectively. Then take the correct subset.
        kss = feval(cov, hyp.cov, K_p, 'diag');              % self-variance
        kss = kss(id);
        Ks  = feval(cov, hyp.cov, K_tp);         % cross-covariances
        Ks  = Ks(nz,id); 
               
        % generate ms with a call to the mean function which is currently
        % harded coded as being meanConst_K
        ms = feval(mean, hyp.mean, K_p);
        ms = ms(id);
        % fmu(id) = ms + Ks'*full(alpha(nz));                       % predictive means 
        fmu(id) = Ks'*full(alpha(nz));                       % predictive means 
           
        if Ltril           % L is triangular => use Cholesky parameters (alpha,sW,L)
            V  = L'\(repmat(sW,1,length(id)).*Ks);
            
            fs2(id) = kss - sum(V.*V,1)';                       % predictive variances
        else                % L is not triangular => use alternative parametrisation
            fs2(id) = kss + sum(Ks.*(L*Ks),1)';                 % predictive variances
        end
        fs2 = max(fs2,0);   % remove numerical noise i.e. negative variances
        num_class = length(hyp.lik); 
        yord = 1:num_class; %the range of ordinal classes
    
        %compute the probability per class
        for i =1:num_class  
            [p(id,i)] = lik(hyp.lik, yord(i), fmu, fs2); 
        end
        
        %assign the test case to the class with the largest probability
        %for i = 1:size(Ks,1)
            %[blah, y_pred(i)] = max(p(i,:)); 
        %end
 
        
        % better - avoids loop and respects range of id
        [blah(id), y_pred(id)] = max(p(id,:),[],2); 
        
        nact = id(end);          % set counter to index of last processed data point
      
    % end while loop
    end
    %calculate the zero one error
    zero_one_err = length(find(y_pred' ~=ys))/length(ys);
    varargout = {zero_one_err,y_pred,p,fmu,fs2};
end
