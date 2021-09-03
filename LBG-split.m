% Source Coding
%
%
% test_splitting.m:  LBG-split algorithm

clear all; close all;

filename = 'dataset/70mono.wav';

info = audioinfo(filename);
[x,F] = audioread(filename,'native') ;
fprintf('\n');
fprintf('Sampling frequency:      F = %d',F); fprintf(' [Hz] \n');
fprintf('Resolution:              nbits = %d',info.BitsPerSample); fprintf(' [bit] \n');

if info.BitsPerSample ~= 16
    fprintf('EXIT: source samples must have a resolution of 16 bit\n');
    return;
end

L = 4;       % dimension of each vector in the codebook (#samples for each block)
R = 2;       % RATE specified
K = 2^(L*R); % cardinality of the codebook: K = 2^(LR)
eps = 0.1;

training_set = zeros(floor(size(x,1)/L), L, 'double');
for i=1: floor(size(x,1)/L)
    for j=1:L
        training_set(i,j) = x( ((i-1)*L) + j ,1);
    end
end

% codebook
codebook = zeros(K, L, 'double');
codebook(1,:) = mean(training_set); % initial centroid: mean of the training set
N = 1; % vectors in the codebook so far

distorsion_curr = double(0);
for i=1:size(training_set,1)
    dist_i = double(0);
    for k = 1:L
        dist_i = dist_i + (training_set(i,k) - codebook(1,k))^2;
    end
    distorsion_curr = distorsion_curr + dist_i;
end
distorsion_curr = distorsion_curr/size(training_set,1);
fprintf('Distortion[%5d] = %12.3f\n', N, distorsion_curr);

distorsion_prev = realmax;
d_err = (distorsion_prev - distorsion_curr)/distorsion_prev;
it = 1;

while( N<K || d_err > 0.0001)
    if (2*N <= K)
        % splitting
        for i=1:N
            curr_centr = codebook(i,:);
            codebook(i,:) = curr_centr;
            codebook(N+i,:) = (1+eps)*curr_centr;
            %fprintf('codebook %d = (%f,%f)\n', i, codebook(i,1), codebook(i,2));
            %fprintf('codebook %d = (%f,%f)\n', N+i, codebook(N+i,1), codebook(N+i,2));
        end
        N = 2*N;
        % LBG optimization
    end
    fprintf("\n Iteration %d: ",it);
    distorsion_prev = distorsion_curr;
    % compute the decision cells
    %I = zeros(N,size(training_set,1));
    counters = zeros(N,1, 'double');
    sum_vec = zeros(N,L, 'double');
    
    max_min_dist = double(0);
    max_sample = 0;
    max_codevec = 0;
    % find the nearest centroid for each training sample
    for i=1:size(training_set,1)
        argmin = 1; % at the beginning, set the first codevector as the nearest
        min_dist = double(0);
        for y = 1:L
            min_dist = min_dist + (training_set(i,y) - codebook(1,y))^2;
        end
        % look for the nearest codevector
        for j=2:N
            temp_dist = double(0);
            for y = 1:L
                temp_dist = temp_dist + (training_set(i,y) - codebook(j,y))^2;
            end
            if (temp_dist<min_dist)
                min_dist = temp_dist;
                argmin = j;
            end
        end
        % count how many samples there are in each cluster
        counters(argmin,1) = counters(argmin,1) + 1;
        % and sum the vectors in order to compute the mean at the end
        sum_vec(argmin,:) = sum_vec(argmin,:) + training_set(i,:);
        distorsion_curr = distorsion_curr + min_dist;
        if (min_dist > max_min_dist)
            max_min_dist = min_dist;
            max_sample = i;
            max_codevec = argmin;
        end
    end
    % check if there is a codevector with no samples associated to it
    % if it exists, substitute this "void" codevector with the sample that contributes the most to
    % the distortion measure
    for w=1:N
      if (counters(w,1) == 0)
          fprintf("*** %d, max_min_dist = %f *** ",w, max_min_dist/size(training_set,1));
          sum_vec(max_codevec,:) = sum_vec(max_codevec,:) - training_set(max_sample,:);
          counters(max_codevec,1) = counters(max_codevec,1) - 1;

          distorsion_curr = distorsion_curr - max_min_dist;
          sum_vec(w,:) = training_set(max_sample,:);
          counters(w,1) = 1;

          break;
      end
    end
    
    distorsion_curr = distorsion_curr/size(training_set,1);
    fprintf('Distortion[%5d] = %12.3f  ', N, distorsion_curr);
    % compute the optimal centroids
    for i=1:N
        if (counters(i,1) > 0)
            codebook(i,:) = sum_vec(i,:)/counters(i,1);
        end
    end
    d_err = (distorsion_prev - distorsion_curr)/distorsion_prev;
    fprintf("d_err = %f\n", d_err);
    it = it+1;
end






