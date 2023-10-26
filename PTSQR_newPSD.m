% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PARALLEL SEQUENTIAL TALL-SKINNY QR DECOMPOSITION
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
close all
clear all

% Load data
% load('y_test.mat')
load('Data_porting.mat')
y = yA;

% Set maximum number of chunks (equal to the number of partitions in the
% 1st step)
Nchunk = 4;

% Set AR/ARMA model order
model_order = 16;
out = y(1:400,1);
Nsmp = size(out,1);

%% Create regression matrix
% Round number of samples to an integer multiple of Nchunks
No = floor((Nsmp-model_order)/Nchunk)*Nchunk;

% Prepare regression matrix Phi (the integer one to be decomposed)
for i = model_order+(1:No)
    
    b = -flip(out(i-model_order:i-1));        % y response
    phi(:,i-model_order) = b;                 % phi value
    
end

%% Start looking at the binary tree
Ni = Nchunk;
for iIter = 1:log2(Nchunk)+1 %1:Nchunk-1 
    
    % 1st round
    if (iIter == 1)
        phiA = phi';
        Nsmp_block = No/Nchunk;
        outi = out(model_order+(1:No));
    else 
        phiA = R;
        Ni = floor(Ni/2);
        Nsmp_block = size(phiA,1)/Ni;
        outi = Yi;
    end
    
    % Update R, Q and Y values at each iteration
    R = [];
    Yi = [];
    Qtemp = zeros(Ni*Nsmp_block,Ni*model_order);
    
    % Start processing considering the available chunks as a power of 2 
    for iCh = 1:2^(log2(Nchunk)-iIter+1)
        phi_i = phiA((iCh-1)*Nsmp_block+1:iCh*Nsmp_block,:);
        [Qi,Ri] = qr(phi_i,0);
        
        % Update vector coefficient
        Yi((iCh-1)*size(Qi,2)+1:iCh*size(Qi,2),1) = Qi'*outi((iCh-1)*Nsmp_block+1:iCh*Nsmp_block);
        % Concatenate new R matrix
        R = [R; Ri];
        % Concatenate new Q matrix
        Qtemp((iCh-1)*Nsmp_block+1:iCh*Nsmp_block,(iCh-1)*model_order+1:iCh*model_order) = Qi;
    end
    
    % Compute total Q matrix at each step to be used for comparison at the
    % end
    Q{iIter} = Qtemp;
    
end

% Compute overall Q
for i = 1:numel(Q)
    
    if (i == 1)
        QT = Q{i};
    else
        QT = QT*Q{i};
    end
end

%% Compare with built-in Matlab decomposition
% Built-in Matlab
[QMat,RMat] = qr(phi',0);

% Resize coefficient vector
out = out(model_order+(1:No));
BMat = QMat'*out;
BPTSQR = Yi;

theta = RMat\BMat;

%% SPECTRUM ESTIMATION via regressive form (PSD)
% Number of model parameters
i = [1:1:model_orderAR];
% Number of points in the FRF
N = 2048;
nn = [0:1:N-1];
for ii = 1:N
    f = nn(ii)/N;
    %%% ---> ARMA SPECTRUM 
    A = 1 + sum(theta(1:model_orderAR)'.*cos(2*pi*f.*i));
    B = sum(theta(1:model_orderAR)'.*sin(2*pi*f.*i));
    
    C = 1 + sum(theta(model_orderAR+1:end)'.*cos(2*pi*f.*i));
    D = sum(theta(model_orderAR+1:end)'.*sin(2*pi*f.*i));
    
    S_ARMA(ii) = sigma^2/(C^2+D^2)^2*((A*C+B*D)^2+(-A*D+B*C)^2);
    
    %%% ---> AR SPECTRUM
    C = 1 + sum(theta(1:model_orderAR)'.*cos(2*pi*f.*i));
    D = sum(theta(1:model_orderAR)'.*sin(2*pi*f.*i));
    
    S_AR(ii) = sigma^2/(C^2+D^2)^2;
end
% Plot equivalent spectrum
% Frequency vector
freq = linspace(0,1/Ts,N);
semilogx(freq,mag2db(S));
xlabel('Frequency [Hz]'), ylabel('|S_y(f)| [dB]')