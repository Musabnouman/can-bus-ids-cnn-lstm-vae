%% ==========================================
% ROAD DATASET: AMBIENT CONCATENATOR (PROPER 30% SAMPLING)
% ==========================================
clear; clc;

% 1. FORCE MATLAB TO LOOK IN THE RIGHT FOLDER
cd('C:\Users\musab\Desktop\road dataset\ambient');

% 2. Find all ambient log files in the directory
files = dir('ambient_*.log');
num_files = length(files);

if num_files < 12
    error('CRITICAL FAILURE: Still only found %d files. MATLAB is stuck.', num_files);
end

all_benign_data = [];
fprintf('Found %d ambient files. Starting extraction...\n', num_files);

% 3. Loop through and parse each file
for i = 1:num_files
    fname = files(i).name;
    fprintf('Parsing [%d/%d]: %s... ', i, num_files, fname);
    
    fid = fopen(fname, 'r');
    raw = textscan(fid, '%*s %*s %s', 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
    fclose(fid);
    
    can_data = raw{1};
    n = numel(can_data);
    X_data = zeros(n, 8);
    
    % Extract the 8 bytes
    for j = 1:n
        parts = split(string(can_data{j}), '#');
        if numel(parts) == 2
            data_str = char(parts(2));
            len = length(data_str);
            if len < 16
                data_str = [data_str, repmat('0', 1, 16 - len)];
            end
            for b = 1:8
                X_data(j, b) = hex2dec(data_str(2*b-1 : 2*b));
            end
        end
    end
    
    % --- THE FIX: Take 30% of THIS specific file chronologically ---
    target_rows = floor(0.30 * n);
    file_30_percent = X_data(1:target_rows, :);
    
    % Concatenate the 30% slice to the master matrix
    all_benign_data = [all_benign_data; file_30_percent];
    fprintf('Done. Added %d rows.\n', target_rows);
end

fprintf('\n--- EXTRACTION COMPLETE ---\n');
fprintf('Total rows in your diverse 30%% dataset: %d\n', size(all_benign_data, 1));

% 4. Save as a compiled MATLAB binary
save_filename = 'ambient_diverse_30pct.mat';
ambient_30_percent = all_benign_data;
save(save_filename, 'ambient_30_percent', '-v7.3');

fprintf('SUCCESS: Matrix saved to %s\n', save_filename);
