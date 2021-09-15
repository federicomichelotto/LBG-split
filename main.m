clear all; close all;

training_data = 'dataset/49mono.wav';
source = 'dataset/49mono.wav';

info_t = audioinfo(training_data);
[t,F_t] = audioread(training_data,'native') ;

info_x = audioinfo(source);
[x,F_x] = audioread(source,'native') ;

fprintf('\n');
fprintf('Sampling frequency:      F = %d',F_x); fprintf(' [Hz] \n');
fprintf('Resolution:              nbits = %d',info_x.BitsPerSample); fprintf(' [bit] \n');

if info_t.BitsPerSample ~= 16 || info_x.BitsPerSample ~= 16
    fprintf('EXIT: training samples and source samples must have a resolution of 16 bit\n');
    return;
end

tic
%%%%%%%%%%%%%%%%% PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R = 2;          % desired RATE
L = 4;          % dimension of each vector in the codebook (#samples for each block)
K = 2^(L*R);    % cardinality of the codebook: K = 2^(LR)
eps = 0.001;    % amount of pertubation during the splitting phase
delta = 0.0001; % minimim relative distortion improvement to continue the LBG alg.

%%%%%%%%%% COMPUTE THE CODEBOOK and QUANTIZE THE SOURCE SIGNAL %%%%%%%%%%%%
T = build_training_set(t,L);
codebook = LBG_split(T,L,R,eps,delta);
reproduction_idx = code(codebook,x);
quantized_signal = decode(codebook, reproduction_idx);

%%%%%%%%%%%%%%%%% COMPUTING the Signal-To-Noise Ratio %%%%%%%%%%%%%%%%%%%%%
var_input = var(double(x));
err = x - quantized_signal;
var_err = var(double(err));
SNR_db = 10 * log10 (var_input / var_err);
fprintf("SNR = %f db\n", SNR_db);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
toc
%audiowrite(strcat(source, string(L)),quantized_signal,F,'BitsPerSample',16);

% figure;
% scatter(codebook(:,1), codebook(:,2), 15, 'red', 'filled');
% 
% figure;
% scatter(T(:,1), T(:,2), 5, 'blue');


%%%%%%%%%%%%%% AUXILIARY FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function T = build_training_set(x,L)
    T = zeros(floor(size(x,1)/L), L, 'double');
    for i=1: floor(size(x,1)/L)
        for j=1:L
            T(i,j) = x( ((i-1)*L) + j ,1);
        end
    end
end


function reproduction_idx = code(codebook, signal)
    K = size(codebook,1);
    L = size(codebook,2);
    if ( K == 0 || L == 0 )
       fprintf('ERROR: malformed codebook given to the function reproduction_idx()');
    end
    S = build_training_set(signal,L);
    reproduction_idx = zeros(size(S,1),1,'int16');
    
    for i=1:size(S,1)
        argmin = 0;
        min_dist = realmax;
        % look for the nearest codevector
        for j=1:K
            temp_dist = double(0);
            for y = 1:L
                temp_dist = temp_dist + (S(i,y) - double(codebook(j,y)))^2;
            end
            if (temp_dist < min_dist)
                min_dist = temp_dist;
                argmin = j;
            end
        end
        reproduction_idx(i,1) = argmin;
    end
    
end

function quant_signal = decode(codebook, reproduction_idx)
    K = size(codebook,1);
    L = size(codebook,2);
    blocks = size(reproduction_idx,1); % #vectors to decode -> signal length = blocks * L
    if ( K == 0 || L == 0 )
       fprintf('ERROR: malformed codebook given to the function code()');
    end
    quant_signal = zeros(blocks*L, 1, 'int16');
    for i = 1:blocks
        for j = 1:L
            quant_signal((i-1)*L + j, 1) = int16(codebook(reproduction_idx(i,1),j));
        end
    end
end
