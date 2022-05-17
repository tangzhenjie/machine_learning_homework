%   X: data matrix, each row is one observation, each column is one feature
%      column mean=0 
%   d: reduced dimension
%   Y: dimensionanlity-reduced data
%   Warning: This function is not optimized for very high dimensional data!
function [Y, eigVector, eigValue]=PCAFunction(X,d)

%% eigenvalue analysis
Sx=cov(X);
[V,D]=eig(Sx);
eigValue=diag(D);
[eigValue,IX]=sort(eigValue,'descend');
eigVector=V(:,IX);

%% normailization
norm_eigVector=sqrt(sum(eigVector.^2));
eigVector=eigVector./repmat(norm_eigVector,size(eigVector,1),1);

%% dimensionality reduction
eigVector=eigVector(:,1:d);
Y=X*eigVector;