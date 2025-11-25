% ASV Demo Code 
% Version: 1 (2016/3/11)
% 
% 
% All rights reserved.
% -----------------------------------------------------------------------------------------------
% Code author: Tsun-Yi Yang 
% Email: shamangary@hotmail.com
% Project page: http://shamangary.logdown.com/posts/587520
% Paper: [CVPR16] Accumulated Stability Voting: A Robust Descriptor from Descriptors of Multiple Scales
% 
% If you use the code, please cite the paper.
% If any bug is found, please email me.
% You may use the code for academic study.
% However, using the provided code for commercial purpose is forbidden.
% -----------------------------------------------------------------------------------------------
% Tested platform:
% Ubuntu 12.04 LTS (I do not test the code on Windows or Mac.)
% Matlab R2012b
% 
% -----------------------------------------------------------------------------------------------

% FUNCTION: vl_asvcovdet
% PURPOSE: Compute ASV (Accumulated Stability Voting) descriptors from multi-scale SIFT/LIOP/patch descriptors
%
% INPUTS:
%   im          - Input image (grayscale or color)
%   opt         - Options structure containing:
%                 .nr       - Number of rotation samples
%                 .rc_min   - Minimum rotation angle (radians)
%                 .rc_max   - Maximum rotation angle (radians)
%                 .ns       - Number of scale samples
%                 .sc_min   - Minimum scale factor
%                 .sc_max   - Maximum scale factor
%   frames_ori  - Original keypoint frames (6×N matrix where each column is a frame)
%                 Row 1-2: x,y position
%                 Row 3-6: Affine transformation matrix (2×2 reshaped to 4×1)
%   des         - Descriptor type: 'sift' (128-dim), 'liop' (144-dim), or 'patch' (1681-dim)
%   isInter     - Whether to use interpolation between scales (1=yes, 0=no)
%
% OUTPUT:
%   D_out       - ASV descriptors (dim × nf matrix where dim depends on nr, ns, and descriptor type)
%
% ALGORITHM:
%   1. For each rotation sample:
%      - Rotate the keypoint frames
%      - For each scale sample:
%        * Extract descriptors at that scale
%      - Apply stability voting across scales
%   2. Stack results from all rotations
function [D_out] = vl_asvcovdet(im, opt, frames_ori,des,isInter)

% Extract parameters from options structure
nr = opt.nr;           % Number of rotation samples
rc_min = opt.rc_min;   % Minimum rotation angle
rc_max = opt.rc_max;   % Maximum rotation angle

% Initialize output descriptor matrix
D_out = [];

% Copy original frames for rotation manipulation
frames = frames_ori;
nf = size(frames_ori, 2);  % Number of keypoints (frames)

% ===== ROTATION LOOP =====
% Sample rotations uniformly between rc_min and rc_max
for rc = linspace(rc_min, rc_max, nr)
    
    % Keep position (x,y) the same
    frames([1 2], :) = frames_ori([1 2], :);
    
    % Rotate the affine transformation matrix for each frame
    % The 2×2 affine matrix is stored as rows 3-6 (reshaped from 2×2 to 4×1)
    for rf = 1:nf
        % Create rotation matrix [cos(rc) -sin(rc); sin(rc) cos(rc)]
        % Apply rotation to the affine transformation matrix
        frames(3:6, rf) = reshape([cos(rc) -sin(rc);sin(rc) cos(rc)] * reshape(frames_ori(3:6,rf),2,2),4,1);
    end
    
    % ===== SCALE SAMPLING =====
    ns = opt.ns;           % Number of scale samples
    sc_min = opt.sc_min;   % Minimum scale factor
    sc_max = opt.sc_max;   % Maximum scale factor
    
    % Pre-allocate frames for all scales
    % f is 6 × nf × ns (6 frame params, nf keypoints, ns scales)
    f = zeros(6, nf, ns);
    
    cnt = 0;
    % Sample scales uniformly between sc_min and sc_max
    for sc = linspace(sc_min, sc_max, ns)
        cnt = cnt + 1;
        % Keep position unchanged
        f([1 2], :, cnt) = frames([1 2], :);
        % Scale the affine transformation matrix
        f(3:6, :, cnt) = sc * frames(3:6,:);
    end
    
    % ===== DESCRIPTOR EXTRACTION =====
    % Determine descriptor dimension based on type
    if strcmp(des,'sift') == 1
        dim = 128;   % SIFT: 128-dimensional
    elseif strcmp(des,'liop') == 1
        dim = 144;   % LIOP: 144-dimensional
    elseif strcmp(des,'patch') == 1
        dim = 1681;  % Raw patch: 41×41 = 1681-dimensional
    end
    
    % Pre-allocate descriptor matrix
    % D is dim × nf × ns (descriptor_dim, num_keypoints, num_scales)
    D = zeros(dim,nf,ns);

    % Extract descriptors at each scale
    for i = 1:ns
        [~,d] = extract(im,des,f(:,:,i));  % External function to extract descriptors
        D(:,:,i) = d;
    end
    
    % Rearrange to dim × ns × nf (descriptor_dim, num_scales, num_keypoints)
    % This makes it easier to process each keypoint independently
    D = permute(D,[1,3,2]);
    
    % ===== OPTIONAL INTERPOLATION =====
    % Add interpolated descriptors between consecutive scales
    % This increases the number of scale samples for smoother voting
    if isInter == 1
        for in = 1:size(D,2)-1
            % Average of consecutive scales
            tempIntep = (D(:,in,:) + D(:,in+1,:))/2;
            % Concatenate interpolated descriptors
            D = cat(2,D,tempIntep);
        end
    end
    
    % ===== ACCUMULATED STABILITY VOTING =====
    % For each keypoint, vote on which descriptor dimensions are stable across scales
    d_out = [];
    for i_f = 1:nf  % For each keypoint
        % Initialize accumulation vector (votes for each descriptor dimension)
        accVec = zeros(dim,1);
        
        % Compare each scale with all subsequent scales
        for c = 1:size(D,2)-1
            % Reference descriptor at scale c
            temp = D(:,c,i_f);
            
            % Compute absolute differences with all later scales
            % M is dim × (num_remaining_scales)
            M = bsxfun(@minus,D(:,(c+1):end,i_f),temp);
            M = abs(M);
            
            %% MEDIAN THRESHOLDING (1st stage)
            % For each comparison, compute median difference across all dimensions
            % This gives a stability threshold for that scale pair
            m = median(M,1);  % 1 × num_comparisons
            
            % Vote: for each dimension, if difference <= median, increment vote
            % This means the dimension is stable (below threshold)
            for t = 1:size(m,2)
                accVec = accVec + (M(:,t) <= m(t));
            end
            
        end
        
        % Convert accumulated votes to double precision
        accVec = double(accVec);
        % Append this keypoint's ASV descriptor
        d_out = [d_out,accVec];
    end
    
    % Stack descriptors from this rotation with previous rotations
    % Final D_out is (dim*nr) × nf
    D_out = [D_out;d_out];
end

end