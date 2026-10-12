%% ==========================================
% ROAD DATASET: AMBIENT CONCATENATOR (9 FEATURES - INCLUDES ID)
% ==========================================
clear; clc;

% 1. Folder holding ROAD's ambient_*.log captures (edit to match your copy).
%    The .mat file is written to the current folder.
ambient_dir = fullfile('road dataset', 'ambient');

files = dir(fullfile(ambient_dir, 'ambient_*.log'));
num_files = length(files);

if num_files < 12
    error('CRITICAL FAILURE: Found %d files. Make sure path is correct.', num_files);
end

all_benign_data = [];
fprintf('Found %d ambient files. Starting 9-feature extraction...\n', num_files);

for i = 1:num_files
    fname = files(i).name;
    fprintf('Parsing [%d/%d]: %s... ', i, num_files, fname);
    
    fid = fopen(fullfile(ambient_dir, fname), 'r');
    raw = textscan(fid, '%*s %*s %s', 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
    fclose(fid);
    
    can_data = raw{1};
    n = numel(can_data);
    
    % NEW: 9 columns (1 for ID, 8 for Data Bytes)
    X_data = zeros(n, 9);
    
    for j = 1:n
        parts = split(string(can_data{j}), '#');
        if numel(parts) == 2
            % 1. Extract ID (Before the #)
            id_str = char(parts(1));
            X_data(j, 1) = hex2dec(id_str);
            
            % 2. Extract Data (After the #)
            data_str = char(parts(2));
            len = length(data_str);
            if len < 16
                data_str = [data_str, repmat('0', 1, 16 - len)];
            end
            for b = 1:8
                X_data(j, b+1) = hex2dec(data_str(2*b-1 : 2*b));
            end
        end
    end
    
    % Take 30% chronologically
    target_rows = floor(0.30 * n);
    file_30_percent = X_data(1:target_rows, :);
    
    all_benign_data = [all_benign_data; file_30_percent];
    fprintf('Done. Added %d rows.\n', target_rows);
end

fprintf('\n--- EXTRACTION COMPLETE ---\n');
ambient_30_percent = all_benign_data;
% id_vae_road.m loads the 9-column matrix from this file name
save('new_ambient_diverse_30pct.mat', 'ambient_30_percent', '-v7.3');
fprintf('SUCCESS: 9-Column Matrix saved to new_ambient_diverse_30pct.mat\n');
