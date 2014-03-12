function [varargout] = likErf_OR_K(hyp, y, mu, s2, inf)

% likErf_OR - Likelihood for ordinal regression which is based on a generalization of the Error function.

% In general, care is taken to avoid numerical issues when the likelihood begins to apprroach zero.
% The numerical checks used here were taken from Wei Chu's ORGP code
% available at: http://www.gatsby.ucl.ac.uk/~chuwei/ordinalregression.html

%Copyright (c) by Orla Doyle May 2013

% Modified by Jonathan Young Jan 2014 to add compatibility with 
% infLaplace_OR_K inference function

if nargin<2,
    varargout = {'0'};
    return;
end   % report number of hyperparameters
if numel(y)==0, y = 1; end

if nargin<5                              % prediction mode if inf is not present
    y = y.*ones(size(mu));                                       % make y a vector
    s2zero = 1; if nargin>3, if norm(s2)>0, s2zero = 0; end, end         % s2==0 ?
    if s2zero                                         % log probability evaluation
        [p,lp] = cumGauss(y,mu);
    else                                                              % prediction
        lp = likErf_OR(hyp, y, mu, s2, 'infLaplace_OR'); p = exp(-lp);
    end
    ymu = 0; ys2 = 0;
  
    varargout = {p,ymu,ys2};
else                                                            % inference mode
    switch inf
        case 'infLaplace_OR'
            if nargin<6                                             % no derivative mode
                z = mu; dlZ = {}; d2lZ = {}; d3lZ={};       
                if isempty(s2)
                    s2 = zeros(length(z),1);
                end
                %compute z value pairs z_i and z_i-1
                num_class = length(hyp);
                sigma = exp(hyp(1));               %noise hyperparam
                %sigma = hyp(1);
                b(1) = -1/eps;
                for i = 2:(num_class)           %start at two as hyp.lik(1) is sigma
                    if i ==2
                        b(i) = hyp(2);
                    else
                        b(i) = b(2) + sum(exp(hyp(3:i)));
                    end
                end
                b(num_class+1)=1/eps;
                ind1 = y+1;  %b is indexed beginning at 1, not zero
                byi = b(ind1);
                byi_1 = b(y);
                z1 = (byi' - mu)./sqrt(repmat(sigma^2,length(mu),1) + s2.^2);
                z2 = (byi_1' - mu)./sqrt(repmat(sigma^2,length(mu),1) + s2.^2);
                [Z,lZ,ind_e] = cumGauss_OR(z1,z2);
                varargout = {-lZ};
                if nargout>1
                    if numel(y)==0, y=1; end
                    s2=0;
                    [dlZ, d2lZ, d3lZ] = derivs_of_Z_OR(Z,hyp,z1, z2,s2,ind_e,num_class,y);
                    varargout = {-lZ,-dlZ,-d2lZ,d3lZ};
                end
                
            else                                                       % derivative mode
                lp_dhyp = abs(y-mu)/b - 1;    % derivative of log likelihood w.r.t. hypers
                d2lp_dhyp = zeros(size(mu));        % and also of the second mu derivative
                varargout = {-lp_dhyp,-d2lp_dhyp};
            end
        
        % must also recognise the precomputed kernel version of infLaplace_OR: infLaplace_OR_K
        % don't actually do anything differently than for infLaplace_OR. 
        case 'infLaplace_OR_K'
            if nargin<6                                             % no derivative mode
                z = mu; dlZ = {}; d2lZ = {}; d3lZ={};       
                if isempty(s2)
                    s2 = zeros(length(z),1);
                end
                %compute z value pairs z_i and z_i-1
                num_class = length(hyp);
                sigma = exp(hyp(1));               %noise hyperparam
                %sigma = hyp(1);
                b(1) = -1/eps;
                for i = 2:(num_class)           %start at two as hyp.lik(1) is sigma
                    if i ==2
                        b(i) = hyp(2);
                    else
                        b(i) = b(2) + sum(exp(hyp(3:i)));
                    end
                end
                b(num_class+1)=1/eps;
                ind1 = y+1;  %b is indexed beginning at 1, not zero
                byi = b(ind1);
                byi_1 = b(y);
                z1 = (byi' - mu)./sqrt(repmat(sigma^2,length(mu),1) + s2.^2);
                z2 = (byi_1' - mu)./sqrt(repmat(sigma^2,length(mu),1) + s2.^2);
                [Z,lZ,ind_e] = cumGauss_OR(z1,z2);
                varargout = {-lZ};
                if nargout>1
                    if numel(y)==0, y=1; end
                    s2=0;
                    [dlZ, d2lZ, d3lZ] = derivs_of_Z_OR(Z,hyp,z1, z2,s2,ind_e,num_class,y);
                    varargout = {-lZ,-dlZ,-d2lZ,d3lZ};
                end
                
            else                                                       % derivative mode
                lp_dhyp = abs(y-mu)/b - 1;    % derivative of log likelihood w.r.t. hypers
                d2lp_dhyp = zeros(size(mu));        % and also of the second mu derivative
                varargout = {-lp_dhyp,-d2lp_dhyp};
            end    
            
    end
end

function [Z,lZ,ind_e] = cumGauss_OR(z1,z2)

Z1 = (erfc(-z1/sqrt(2)))/2;     % phi(z1)
Z2 = -1*(2-erfc(z2/sqrt(2)))/2; % minus phi(z2)
Z = Z1+Z2;  
%upper bounds for erfc, phi(z1) = 0.5*erfc(-z1/sqrt(2))
%-phi(z2) = -0.5(1-erfc(z2/sqrt(2))
%bound acoording to Abramowitz&Stegun 7.1.13, with a bound per phi(z)
u_t1 = 1./(sqrt((z1.^2)/2 +4/pi) - z1/sqrt(2));
u_t2 = 1./(sqrt((z2.^2)/2 +4/pi) + z2/sqrt(2));

ind_b1 = find(z1<=-7);      %apply bound for phi(z1)
indb1_interp = find(z1>-7 & z1<=-6);   %interp between bound and regular implementation
Z1(ind_b1) = (1/sqrt(pi)).*(exp(-(z1(ind_b1).^2)/2).*u_t1(ind_b1));
interp = -6-z1(indb1_interp);
Z1(indb1_interp)=((1-interp).*Z1(indb1_interp)) + (interp.*(1/sqrt(pi))).*(exp(-(z1(indb1_interp).^2)/2).*u_t1(indb1_interp));
Z=Z1+Z2;

ind_b2 = find(z2>=7);      %apply bound for phi(z1)
indb2_interp = find(z2<7 & z2>=6);   %interp between bound and regular implementation
Z2(ind_b2) = -1 + (1/sqrt(pi)).*(exp(-(z2(ind_b2).^2)/2).*u_t2(ind_b2));
%lik
Z(ind_b2) = Z1(ind_b2) + -1*ones(size(Z2(ind_b2))) + (1/sqrt(pi)).*(exp(-(z2(ind_b2).^2)/2).*u_t2(ind_b2)); %we need to do this in one line to avoid Z = 0, instead of Z ~ exp-13

interp2 = z2(indb2_interp)-6;
Z2(indb2_interp)=(1-interp2).*Z2(indb2_interp) + interp2.*(-1*ones(size(Z2(indb2_interp))) + (1/sqrt(pi)).*(exp(-(z2(indb2_interp).^2)/2).*u_t2(indb2_interp)));
Z(indb2_interp) = Z1(indb2_interp) + Z2(indb2_interp); 

%more numerical checks
ind_e = find(Z<eps);
lZ = log(Z);
for i = 1:length(ind_e)
    lZ(ind_e(i))  = log(eps);
end


function [dlZ, d2lZ, d3lZ] = derivs_of_Z_OR(Z,hyp,z1, z2,s2,ind_e,num_class,y)

sigma = exp(hyp(1)); 

dlZ = (1./sqrt(s2 + sigma.^2)).*(normpdf(z1) - normpdf(z2))./(Z);
beta = (1./(2.*(s2 + sigma.^2))).*(z1.*normpdf(z1) - z2.*normpdf(z2))./Z;
d2lZ = (dlZ.*dlZ) + 2.*beta; 
ind_b = find((z1 <= -6) | (z2 >= 6)); %check bounds
d2lZ(ind_b) = 0.99;
d3lZ = (3/sigma).*beta+ (1/sigma.^2).*dlZ + beta.*dlZ + (1./sigma)*dlZ;
%Checks from the implementation of Chu et al. 2005
for i = 1:length(ind_e)
    j=ind_e(i);
    if y(j) == 1 
        dlZ(j) = -z1(j)/sigma;
        d2lZ(j) = 1/sigma.^2;
    elseif y(j) == num_class
        dlZ(j) = -z2(j)/sigma;
        d2lZ(j) = 1/sigma.^2;
    else
        if (normpdf(z1(j)) - normpdf(z2(j))) > 100*eps 
            dlZ(j) = -(z1(j)*exp((-0.5*z1(j)*z1(j))+(0.5*z2(j)*z2(j)))-z2(j))/(sigma*(exp(-0.5*z1(j)*z1(j)+0.5*z2(j)*z2(j))-1.0));
            d2lZ(j) = 1/sigma^2 + dlZ(j)^2 - (z1(j)*z1(j)*exp(-0.5*z1(j)*z1(j)+0.5*z2(j)*z2(j))-z2(j)*z2(j))/(exp(-0.5*z1(j)*z1(j)+0.5*z2(j)*z2(j))-1.0)/sigma^2;
        else
            dlZ(j) = 0;
            d2lZ(j)=1/sigma^2;
        end
    end
end
 

