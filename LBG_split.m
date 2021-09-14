function [codebook,counters] = LBG_split(T,L,R,eps,delta)
    K = 2^(L*R);
    codebook = zeros(K, L, 'double');
    codebook(1,:) = mean(T); % initial centroid: mean of the training set
    
    N = 1; % actual #codevectors in the codebook

    distorsion_curr = double(0);
    for i=1:size(T,1)
        dist_i = double(0);
        for k = 1:L
            dist_i = dist_i + (T(i,k) - codebook(1,k))^2;
        end
        distorsion_curr = distorsion_curr + dist_i;
    end
    fprintf('\nInitial Distortion (K = 1): %f\n', distorsion_curr);
    
    distorsion_prev = realmax;
    rel_improv = (distorsion_prev - distorsion_curr)/distorsion_prev;
    iteration = 1;
    
    while( N < K || rel_improv > delta)
        if (N == K)
            %return;
            fprintf("\nIteration %d:\n", iteration);
            iteration = iteration + 1; 
        end
        
        if (2*N <= K)
            % splitting
            fprintf("\nInitialization (K = %d):\n", 2*N);
     
            for i=1:N
                curr_centr = codebook(i,:);
                codebook(i,:) = (1-eps)*curr_centr;
                codebook(N+i,:) = (1+eps)*curr_centr;
            end
            N = 2*N;
        end
        
        % LBG optimization
        distorsion_prev = distorsion_curr;
        counters = zeros(N,1, 'double');
        sum_vec = zeros(N,L, 'double');
        
        max_min_dist = double(0);
        max_sample = 0;
        max_codevec = 0;
        % find the nearest centroid for each training sample
        tic
        for i=1:size(T,1)
            % look for the nearest codevector
            argmin = 0;
            min_dist = realmax;
            for j=1:N
                %temp_dist = norm((T(i,:) - codebook(j,:)))^2;
                temp_dist = double(0);
                for y = 1:L
                    temp_dist = temp_dist + (T(i,y) - codebook(j,y))^2;
                end
                if (temp_dist < min_dist)
                    min_dist = temp_dist;
                    argmin = j;
                end
            end
            % keep track of how many samples there are in each cluster
            counters(argmin,1) = counters(argmin,1) + 1;
            % and sum the vectors in order to compute the mean at the end
            sum_vec(argmin,:) = sum_vec(argmin,:) + T(i,:);
            distorsion_curr = distorsion_curr + min_dist;
            if (min_dist > max_min_dist)
                max_min_dist = min_dist;
                max_sample = i;
                max_codevec = argmin;
            end
        end
        toc
        % check if there is a codevector with no samples associated to it
        % if it exists, substitute this "void" codevector with the sample that contributes the most to
        % the distortion measure
        for w=1:N
            if (counters(w,1) == 0)
                fprintf("*** %d, max_min_dist = %f *** ",w, max_min_dist/size(T,1));
                sum_vec(max_codevec,:) = sum_vec(max_codevec,:) - T(max_sample,:);
                counters(max_codevec,1) = counters(max_codevec,1) - 1;
                
                distorsion_curr = distorsion_curr - max_min_dist;
                sum_vec(w,:) = T(max_sample,:);
                counters(w,1) = 1;
                break;
            end
        end
        distorsion_curr = distorsion_curr/size(T,1);
        fprintf('\tDistortion = %.3f\n', distorsion_curr);
        % compute the optimal centroids
        for i=1:N
            if (counters(i,1) > 0)
                codebook(i,:) = sum_vec(i,:)/counters(i,1);
            end
            % if (counters(i,1) == 0 ): keep the "old" codevector that in
            % the next iterations will be replaced
        end
        rel_improv = (distorsion_prev - distorsion_curr)/distorsion_prev;
        fprintf("\tImprovement = %f %%\n", rel_improv*100);
    end
end
