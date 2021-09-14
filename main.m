clear all; close all;

training_data = 'dataset/64mono.wav';
source = 'dataset/70mono.wav';

info = audioinfo(training_data);
[x,F] = audioread(training_data,'native') ;
fprintf('\n');
fprintf('Sampling frequency:      F = %d',F); fprintf(' [Hz] \n');
fprintf('Resolution:              nbits = %d',info.BitsPerSample); fprintf(' [bit] \n');

% if info.BitsPerSample ~= 16
%     fprintf('EXIT: source samples must have a resolution of 16 bit\n');
%     return;
% end

L = 2;       % dimension of each vector in the codebook (#samples for each block)
R = 4;       % RATE specified
K = 2^(L*R); % cardinality of the codebook: K = 2^(LR)
eps = 0.0001;
delta = 0.001;

T = build_training_set(x,L);
[codebook,counters] = LBG_split(T,L,R,eps,delta);

% COMPUTING the Signal-To-Noise Ratio
[z,F] = audioread(source,'native') ;
Z = build_training_set(z,L);

Q = zeros(size(Z,1),1);
for i=1:size(Z,1)
    argmin = 0;
    min_dist = realmax;
    % look for the nearest codevector
    for j=1:K
        %temp_dist = sum((training_set(i,:) - codebook(j,:)).^2);
        temp_dist = double(0);
        for y = 1:L
            temp_dist = temp_dist + (Z(i,y) - codebook(j,y))^2;
        end
        if (temp_dist < min_dist)
            min_dist = temp_dist;
            argmin = j;
        end
        Q(i,1) = argmin;
    end
end

var_input = var(Z);
err = Z - codebook(Q,:);
var_err = var(err);
SNR_db = 10 * log10 (var_input / var_err);
fprintf("SNR = %f\n", SNR_db);

% figure;
% scatter(codebook(:,1), codebook(:,2), 15, 'red', 'filled');
% 
% figure;
% scatter(T(:,1), T(:,2), 5, 'blue');




function T = build_training_set(x,L)
    T = zeros(floor(size(x,1)/L), L, 'double');
    for i=1: floor(size(x,1)/L)
        for j=1:L
            T(i,j) = x( ((i-1)*L) + j ,1);
        end
    end
end