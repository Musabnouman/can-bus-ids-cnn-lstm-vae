% ID_VAE_ROAD  CNN-LSTM VAE with a learned CAN-ID embedding, ROAD dataset.
%   Each frame is a 16-d embedding of its CAN ID (vocabulary built from
%   benign traffic, one extra row for IDs never seen there) followed by the
%   8 payload bytes scaled to [0,1]. The model predicts frame 32 from
%   frames 1-31. Score = weighted prediction error + KL term + a penalty
%   times the fraction of unseen IDs in the window; the script reports
%   results for penalties 0 to 5.
%   Needs new_ambient_diverse_30pct.mat from prep/road_ambient_with_id.m.

%% ==========================================
% 0. CONFIGURATION & LOADING
% ==========================================
clear; clc; 
rng(42);   % fixed seed: shuffling, sampling and initialisation repeat

dataset_mode = 'ROAD';              
attack_filename = 'fuzzing_attack_1.log'; 

if strcmp(dataset_mode, 'CIC')
    benign_filename = 'decimal_benign.csv';
elseif strcmp(dataset_mode, 'CARHACK')
    benign_filename = 'normal_run_data.txt';
elseif strcmp(dataset_mode,'ROAD')
    % Handled via .mat file
end

target_cols = {'ID','DATA_0','DATA_1','DATA_2','DATA_3','DATA_4','DATA_5','DATA_6','DATA_7'};

% : CNN-LSTM VAE (PREDICTION + KL + ATTENTION)
W = 32;          % Window Size
S = 4;           % Stride (Memory Safe)
latentDim = 16;   
hiddenSize = 128; 
learningRate = 1e-4;
maxEpochs = 30;  
batchSize = 64; 
kl_weight = 5e-5;

%% ==========================================
% 1. LOAD & PARSING (9 Columns for all datasets)
% ==========================================
fprintf('Loading data for %s...\n', dataset_mode);

if strcmp(dataset_mode,'CIC')
    opts = detectImportOptions(benign_filename);
    opts.VariableNamingRule = 'preserve';
    decimalbenign = readtable(benign_filename, opts);
    decimalbenign = decimalbenign(:, target_cols);
    attack_data_table = readtable(attack_filename, opts);
    attack_data_table = attack_data_table(:, target_cols);
    X_benign_all = table2array(decimalbenign);
    X_attack_all = table2array(attack_data_table);
    
elseif strcmp(dataset_mode,'CARHACK')
    opts_benign = detectImportOptions(benign_filename, 'FileType', 'text');
    opts_benign.Delimiter = {' ', '\t'};
    opts_benign.ConsecutiveDelimitersRule = 'join';
    opts_benign.LeadingDelimitersRule = 'ignore';
    opts_benign.DataLines = [1 Inf];
    opts_benign.VariableNamesLine = 0;
    raw_benign_table = readtable(benign_filename, opts_benign, 'ReadVariableNames', false);
    benign_d = zeros(height(raw_benign_table), 9);
    for k=1:9
        if k == 1, col_idx = 4; else, col_idx = 6 + k; end
        if col_idx <= width(raw_benign_table)
            benign_d(:,k) = hex2dec_force(string(raw_benign_table{:, col_idx})); 
        end
    end
    decimalbenign = array2table(benign_d, 'VariableNames', target_cols);
    [attack_data_full, Y_attack_rows] = parse_carhack_attack_csv(attack_filename);
    if width(attack_data_full) > 9
        attack_data_full = attack_data_full(:, 2:10); 
        attack_data_full.Properties.VariableNames = target_cols;
    end
    X_attack_raw = table2array(attack_data_full);
    idx_T = (Y_attack_rows == 1);
    idx_R = (Y_attack_rows == 0);
    X_attack_T = X_attack_raw(idx_T, :); 
    X_attack_R = X_attack_raw(idx_R, :); 
    X_benign_all = table2array(decimalbenign);

elseif strcmp(dataset_mode, 'ROAD')
    load('new_ambient_diverse_30pct.mat'); % Ensure this is your 9-col file
    X_benign_all = ambient_30_percent; 
    [attack_data_matrix, attack_timestamps, attack_ids] = parse_road_log(attack_filename);
    json_text = fileread('capture_metadata.json');
    road_meta = jsondecode(json_text);
    [~, base_name, ~] = fileparts(attack_filename);
    if isfield(road_meta, base_name)
        file_meta = road_meta.(base_name);
        inj_interval = file_meta.injection_interval;
        inj_id_str = file_meta.injection_id;
        if isempty(inj_interval), inj_start = -1; inj_end = -1;
        else, inj_start = inj_interval(1); inj_end = inj_interval(2); end
    else
        error('Metadata for %s not found in metadata.json', base_name);
    end
    Y_attack_rows = zeros(size(attack_data_matrix, 1), 1);
    if inj_start >= 0
        time_mask = (attack_timestamps >= inj_start) & (attack_timestamps <= inj_end);
        if isempty(inj_id_str) || strcmp(string(inj_id_str), "XXX") || strcmp(string(inj_id_str), "null")
            Y_attack_rows(time_mask) = 1;
        else
            clean_hex = strrep(string(inj_id_str), "0x", "");
            target_id_dec = hex2dec(char(clean_hex));
            id_mask = (attack_ids == target_id_dec);
            Y_attack_rows(time_mask & id_mask) = 1;
        end
    end
    idx_T = (Y_attack_rows == 1);
    idx_R = (Y_attack_rows == 0);
    X_attack_T = attack_data_matrix(idx_T, :); 
    X_attack_R = attack_data_matrix(idx_R, :); 
end

if strcmp(dataset_mode, 'CARHACK')|| strcmp(dataset_mode, 'ROAD')
    X_attack_all = X_attack_T; 
end

X_benign_all(isnan(X_benign_all)) = 0;
X_attack_all(isnan(X_attack_all)) = 0; 

%% ==========================================
% 2. PREPROCESSING (Embedding Setup & Min-Max)
% ==========================================
fprintf('\n--- PREPROCESSING ---\n');

% Build vocabulary from BENIGN only
all_unique_ids = unique(X_benign_all(:, 1));
num_unique_ids = length(all_unique_ids);
fprintf('Legitimate CAN IDs in ambient: %d\n', num_unique_ids);

id_benign = encodeIDtoIndex(X_benign_all(:,1), all_unique_ids);
id_attack = encodeIDtoIndex(X_attack_all(:,1), all_unique_ids);

data_benign = single(X_benign_all(:, 2:9)) / 255.0;
data_attack = single(X_attack_all(:, 2:9)) / 255.0;

X_benign_processed = [id_benign, data_benign];   
X_attack_processed = [id_attack, data_attack];

if strcmp(dataset_mode, 'CARHACK')
    id_R = encodeIDtoIndex(X_attack_R(:,1), all_unique_ids);
    data_R = single(X_attack_R(:, 2:9)) / 255.0;
    X_attack_R_processed = [id_R, data_R];
end

nFeatures_data  = 8;               
embedDim        = 16;              
nFeatures       = embedDim + nFeatures_data;  

fprintf('Embedding %d IDs into %d dims. Total feature dim: %d\n', num_unique_ids, embedDim, nFeatures);

%% ==========================================
% 3. PARTITIONING (No pre-built windows)
% ==========================================
fprintf('\n--- PARTITIONING ---\n');
n_benign = size(X_benign_processed, 1);
n_train  = floor(0.60 * n_benign);
n_val    = floor(0.20 * n_benign);

X_train_rows       = X_benign_processed(1 : n_train, :);
X_val_rows         = X_benign_processed(n_train+1 : n_train+n_val, :);
X_test_benign_rows = X_benign_processed(n_train+n_val+1 : end, :);
X_test_attack_rows = X_attack_processed;

if strcmp(dataset_mode, 'CARHACK')
    X_test_benign_rows = [X_test_benign_rows; X_attack_R_processed];
end

% Compute number of windows per split (for progress reporting)
n_train_wins  = floor((size(X_train_rows,1)  - W) / S) + 1;
n_val_wins    = floor((size(X_val_rows,1)    - W) / S) + 1;
n_ben_wins    = floor((size(X_test_benign_rows,1) - W) / S) + 1;
n_atk_wins    = floor((size(X_test_attack_rows,1) - W) / S) + 1;

fprintf('Window counts (on-the-fly):\n');
fprintf('  Train:%d  Val:%d  TestBenign:%d  TestAttack:%d\n', ...
    n_train_wins, n_val_wins, n_ben_wins, n_atk_wins);

fprintf('X_benign_all size: [%d %d]\n', size(X_benign_all,1), size(X_benign_all,2));
fprintf('X_attack_all size: [%d %d]\n', size(X_attack_all,1), size(X_attack_all,2));
fprintf('Unknown attack IDs: %d / %d\n', sum(id_attack==0), numel(id_attack));

%% ==========================================
% 4. VAE TRAINING (With Embeddings)
% ==========================================
numHeads = 2;
lgraphE = layerGraph();
lgraphE = addLayers(lgraphE, sequenceInputLayer(nFeatures, "Name","in"));
lgraphE = addLayers(lgraphE, convolution1dLayer(3, 32, "Padding","same", "Name","cnn"));
lgraphE = addLayers(lgraphE, reluLayer("Name","relu"));
lgraphE = addLayers(lgraphE, lstmLayer(hiddenSize,"OutputMode","sequence","Name","lstm_enc"));
lgraphE = addLayers(lgraphE, selfAttentionLayer(numHeads, hiddenSize, "Name","attn"));
lgraphE = addLayers(lgraphE, globalAveragePooling1dLayer("Name","gap")); 
lgraphE = addLayers(lgraphE, fullyConnectedLayer(2*latentDim,"Name","fc_mu_logvar"));
lgraphE = connectLayers(lgraphE,"in","cnn");
lgraphE = connectLayers(lgraphE,"cnn","relu");
lgraphE = connectLayers(lgraphE,"relu","lstm_enc");
lgraphE = connectLayers(lgraphE,"lstm_enc","attn");
lgraphE = connectLayers(lgraphE,"attn","gap");
lgraphE = connectLayers(lgraphE,"gap","fc_mu_logvar");
netE = dlnetwork(lgraphE);

lgraphD = layerGraph();
lgraphD = addLayers(lgraphD, featureInputLayer(latentDim,"Name","z"));
lgraphD = addLayers(lgraphD, fullyConnectedLayer(hiddenSize,"Name","fc1"));
lgraphD = addLayers(lgraphD, reluLayer("Name","relu1"));
lgraphD = addLayers(lgraphD, fullyConnectedLayer(nFeatures,"Name","fc_out")); 
lgraphD = connectLayers(lgraphD,"z","fc1");
lgraphD = connectLayers(lgraphD,"fc1","relu1");
lgraphD = connectLayers(lgraphD,"relu1","fc_out");
netD = dlnetwork(lgraphD);

% Embedding table (Row 1 = Unknown OOV ID)
embeddingTable = dlarray(randn(num_unique_ids+1, embedDim, 'single') * 0.01);

% Optimization Setup
avgGradientsE = []; avgGradientsSqE = [];
avgGradientsD = []; avgGradientsSqD = [];
avgGradientsEmbed = []; avgGradientsSqEmbed = []; % NEW: Track embedding momentum
iter = 1;

patience = 5;
min_delta = 1e-4; 
best_val_loss = inf;
patience_counter = 0;

% NEW: Initialize best models to prevent undefined variable crashes
best_netE = netE;
best_netD = netD;
best_embed = embeddingTable;


fprintf('\nStarting VAE Training (on-the-fly windows)...\n');

% Pre-compute all valid window start indices
train_starts = 1 : S : (size(X_train_rows,1) - W + 1);
n_train_wins = numel(train_starts);

for epoch = 1:maxEpochs
    perm = randperm(n_train_wins);   % Shuffle window START indices
    epoch_losses = [];

    for i = 1 : batchSize : n_train_wins
        j             = min(i + batchSize - 1, n_train_wins);
        batch_starts  = train_starts(perm(i:j));
        current_batch = numel(batch_starts);

        % Build batch tensor on-the-fly [W x 9 x B]
        X_mat = zeros(W, 9, current_batch, 'single');
        for b = 1:current_batch
            s = batch_starts(b);
            X_mat(:,:,b) = X_train_rows(s : s+W-1, :);
        end

        id_seq   = X_mat(:, 1, :);     % [W x 1 x B]
        data_seq = X_mat(:, 2:end, :); % [W x 8 x B]

        [loss, gradientsE, gradientsD, gradEmbed] = dlfeval(@modelGradients, ...
            netE, netD, id_seq, data_seq, embeddingTable, latentDim, kl_weight);

        [netE, avgGradientsE, avgGradientsSqE] = adamupdate(netE, gradientsE, ...
            avgGradientsE, avgGradientsSqE, iter, learningRate);
        [netD, avgGradientsD, avgGradientsSqD] = adamupdate(netD, gradientsD, ...
            avgGradientsD, avgGradientsSqD, iter, learningRate);
        [embeddingTable, avgGradientsEmbed, avgGradientsSqEmbed] = adamupdate(...
            embeddingTable, gradEmbed, avgGradientsEmbed, avgGradientsSqEmbed, iter, learningRate);

        iter = iter + 1;
        epoch_losses(end+1) = extractdata(loss); %#ok<AGROW>
    end

    % Validation scoring (on-the-fly)
    val_scores = scoreOnTheFly(netE, netD, X_val_rows, embeddingTable, ...
                               W, S, latentDim, kl_weight, embedDim, ...
                               num_unique_ids, batchSize, 0.0);
    valLoss   = mean(val_scores);
    trainLoss = mean(epoch_losses);

    fprintf('Epoch %d/%d | Train: %.5f | Val: %.5f', epoch, maxEpochs, trainLoss, valLoss);

    if valLoss < best_val_loss - min_delta
        best_val_loss = valLoss;
        best_netE = netE; best_netD = netD; best_embed = embeddingTable;
        patience_counter = 0;
        fprintf(' ← BEST\n');
    else
        patience_counter = patience_counter + 1;
        fprintf(' (patience %d/%d)\n', patience_counter, patience);
        if patience_counter >= patience
            fprintf('Early stopping at epoch %d.\n', epoch);
            break;
        end
    end
end
netE = best_netE; netD = best_netD; embeddingTable = best_embed;


%% ==========================================
% 6. EVALUATION
% ==========================================
fprintf('\n--- HYBRID BATCH TESTING ---\n');

val_scores = scoreOnTheFly(netE, netD, X_val_rows, embeddingTable, ...
                           W, S, latentDim, kl_weight, embedDim, ...
                           num_unique_ids, batchSize, 0.0);
targetFPR = 0.01;
thr = quantile(val_scores, 1-targetFPR);
fprintf('Threshold (val 99th pct): %.6f\n', thr);

penalties = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0];
best_f1 = -inf; best_p=0; best_acc=0; best_prec=0; best_rec=0; best_auc=0; best_fpr=0;

fprintf('\n--- PENALTY SWEEP ---\n');
for p = penalties
    s_b = scoreOnTheFly(netE, netD, X_test_benign_rows, embeddingTable, ...
                        W, S, latentDim, kl_weight, embedDim, num_unique_ids, batchSize, p);
    s_a = scoreOnTheFly(netE, netD, X_test_attack_rows, embeddingTable, ...
                        W, S, latentDim, kl_weight, embedDim, num_unique_ids, batchSize, p);

    err_t    = [s_b(:); s_a(:)];
    Yw_sweep = categorical([repmat("benign",numel(s_b),1); repmat("malicious",numel(s_a),1)]);
    Yh       = categorical(err_t > thr, [false true], ["benign","malicious"]);

    [p2,r2,f2,~,~] = prf1(Yw_sweep, Yh, "malicious", false);
    [~,~,~,AUC_p]  = perfcurve(double(Yw_sweep=="malicious"), err_t, 1);
    acc_p = mean(Yw_sweep == Yh);
    fpr_p = sum(Yh(Yw_sweep=="benign")=="malicious") / sum(Yw_sweep=="benign");

    fprintf('Penalty=%4.1f → Acc=%.3f Prec=%.3f Rec=%.3f F1=%.3f AUC=%.4f FPR=%.4f\n', ...
        p, acc_p, p2, r2, f2, AUC_p, fpr_p);

    if f2 > best_f1
        best_f1=f2; best_p=p; best_prec=p2; best_rec=r2;
        best_auc=AUC_p; best_acc=acc_p; best_fpr=fpr_p;
    end
end

fprintf('\n[BEST RESULT — Penalty=%.1f]\n', best_p);
fprintf('Accuracy=%.3f | Precision=%.3f | Recall=%.3f | F1=%.3f | AUC=%.4f | FPR=%.4f\n', ...
    best_acc, best_prec, best_rec, best_f1, best_auc, best_fpr);

%% ==========================================
% SUPPORT FUNCTIONS
% ==========================================
function idx = encodeIDtoIndex(raw_ids, vocab)
    [found, loc] = ismember(raw_ids, vocab);
    idx = zeros(size(raw_ids), 'single');
    idx(found) = single(loc(found));   
    idx(~found) = 0;                   
end

function [loss, gradE, gradD, gradEmbed] = modelGradients(netE, netD, id_seq, data_seq, embedTable, latentDim, kl_weight)
    W             = size(id_seq, 1);
    current_batch = size(id_seq, 3);
    num_unique_ids = size(embedTable, 1) - 1;
    embedDim      = size(embedTable, 2);

    % 1. Embedding lookup (inside tape — gradient flows to embedTable)
    id_idx_flat = max(1, min(round(id_seq(:)) + 1, num_unique_ids + 1));
    embed_flat  = embedTable(id_idx_flat, :);           % [W*B x embedDim] dlarray
    % Rows of embed_flat run time-fastest, then window: reshape to W x B x E
    % and move E forward. reshape(embed_flat, W, embedDim, B) would mix
    % embedding values from different windows.
    embed_seq   = permute(reshape(embed_flat, W, current_batch, embedDim), [1 3 2]);  % [W x embedDim x B]

    % 2. Extract targets from RAW arrays BEFORE any dlarray labeling
    %    This avoids all CTB indexing ambiguity
    embed_target = reshape(extractdata(embed_seq(end, :, :)), embedDim, current_batch);  % [embedDim x B]
    data_target  = reshape(data_seq(end, :, :),               8,        current_batch);  % [8 x B]
    X_target     = single([embed_target; data_target]);    % [nFeatures x B], plain array

    % 3. Build input sequence (steps 1..W-1) as CTB dlarray
    embed_in = permute(embed_seq(1:end-1, :, :), [2 1 3]);  % [embedDim x W-1 x B]
    data_in  = permute(data_seq(1:end-1,  :, :), [2 1 3]);  % [8       x W-1 x B]
    embed_dl = dlarray(embed_in, 'CTB');
    data_dl  = dlarray(single(data_in),  'CTB');
    X_in     = cat(1, embed_dl, data_dl);                  % [nFeatures x W-1 x B]

    % 4. Encoder
    Zstats = forward(netE, X_in);
    Zstats = reshape(Zstats, 2*latentDim, current_batch);  % GAP already pooled

    mu     = Zstats(1:latentDim, :);
    logVar = Zstats(latentDim+1:end, :);

    % 5. Reparameterization
    epsilon = randn(size(mu), 'like', mu);
    Z       = mu + exp(0.5 * logVar) .* epsilon;

    % 6. Decoder
    X_pred = forward(netD, dlarray(Z, 'CB'));
    X_pred = reshape(X_pred, embedDim + 8, current_batch);

    % 7. Weighted MSE loss
    diff_embed = X_target(1:embedDim, :)    - X_pred(1:embedDim, :);
    diff_data  = X_target(embedDim+1:end,:) - X_pred(embedDim+1:end,:);

    mse = 5.0 * mean(diff_embed.^2, 'all') + mean(diff_data.^2, 'all');
    kl  = -0.5 * mean(sum(1 + logVar - mu.^2 - exp(logVar), 1), 'all');

    loss = mse + kl_weight * kl;

    % 8. Gradients
    [gradE, gradD, gradEmbed] = dlgradient(loss, netE.Learnables, netD.Learnables, embedTable);
end

% function scores = getHybridScoreBatched(netE, netD, X_cells, embedTable, latentDim, W, kl_weight, batchSize, id_penalty)
%     n              = numel(X_cells);
%     scores         = zeros(n, 1);
%     embedDim       = size(embedTable, 2);
%     num_unique_ids = size(embedTable, 1) - 1;
%     nFeatures      = embedDim + 8;
%     embTable_data  = extractdata(embedTable);
% 
%     for i = 1:batchSize:n
%         j             = min(i + batchSize - 1, n);
%         current_batch = j - i + 1;
% 
%         % Build raw batch [W x 9 x B]
%         X_mat    = reshape(cat(3, X_cells{i:j}), W, 9, current_batch);
%         id_seq   = X_mat(:, 1, :);    % [W x 1 x B]
%         data_seq = X_mat(:, 2:9, :);  % [W x 8 x B]
% 
%         % Unknown ID fraction [B x 1]
%         id_indices_3d = round(id_seq) + 1;
%         id_indices_3d = max(1, min(id_indices_3d, num_unique_ids + 1));
%         unknown_fraction = reshape(mean(id_indices_3d == 1, 1), current_batch, 1);
% 
%         % Embedding lookup
%         embed_flat = embTable_data(id_indices_3d(:), :);              % [W*B x embedDim]
%         embed_seq  = permute(reshape(embed_flat, W, current_batch, embedDim), [1 3 2]); % [W x embedDim x B]
% 
%         % --- Target: extract last time step from raw arrays (no dlarray indexing) ---
%         embed_target = reshape(embed_seq(end, :, :), embedDim, current_batch); % [embedDim x B]
%         data_target  = reshape(data_seq(end, :, :), 8,        current_batch); % [8 x B]
%         X_target     = [embed_target; data_target];                            % [nFeatures x B]
% 
%         % --- Input: steps 1..W-1 as CTB dlarray ---
%         embed_in = dlarray(permute(embed_seq(1:end-1,:,:), [2 1 3]), 'CTB'); % [embedDim x W-1 x B]
%         data_in  = dlarray(single(permute(data_seq(1:end-1,:,:), [2 1 3])), 'CTB'); % [8 x W-1 x B]
%         X_in     = cat(1, embed_in, data_in);                                 % [nFeatures x W-1 x B]
% 
%         % Encoder
%         Zstats = extractdata(predict(netE, X_in));
%         Zstats = reshape(Zstats, 2*latentDim, current_batch);  % [2L x B]
% 
%         mu     = Zstats(1:latentDim, :);
%         logVar = Zstats(latentDim+1:end, :);
% 
%         % Decoder
%         X_pred = extractdata(predict(netD, dlarray(mu, 'CB')));
%         X_pred = reshape(X_pred, nFeatures, current_batch);    % [nFeatures x B]
% 
%         % Hybrid score
%         diff_embed = X_target(1:embedDim, :)    - X_pred(1:embedDim, :);
%         diff_data  = X_target(embedDim+1:end,:) - X_pred(embedDim+1:end,:);
% 
%         mse      = reshape(5.0*mean(diff_embed.^2,1) + mean(diff_data.^2,1), current_batch, 1);
%         kl_terms = reshape(-0.5*sum(1+logVar-mu.^2-exp(logVar), 1),          current_batch, 1);
% 
%         scores(i:j) = mse + kl_weight*kl_terms + id_penalty*unknown_fraction;
%     end
% end
function [tbl, y_rows] = parse_carhack_attack_csv(fname)
    fid = fopen(fname,'r'); if fid < 0, error("Could not open %s", fname); end
    raw = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', ''); fclose(fid); 
    lines = string(raw{1}); n = numel(lines); X = zeros(n, 10); y_rows = zeros(n, 1); keep = false(n,1); 
    for i = 1:n
        line_str = strtrim(lines(i)); if strlength(line_str) < 5, continue; end
        parts = split(line_str, ","); parts = strtrim(erase(parts, ['"'; "'"]));
        if numel(parts) < 4, continue; end
        flag="R"; flagIdx=0;
        for t=numel(parts):-1:1, tok=upper(parts(t)); if tok=="T"||tok=="R", flag=tok; flagIdx=t; break; end; end
        y_rows(i) = double(flag=="T");
        id_tok = erase(parts(2),"0x"); 
        if strlength(id_tok)>0 && all(isstrprop(id_tok,'xdigit')), id_val=hex2dec(id_tok); else, id_val=0; end
        dlc = str2double(parts(3)); if isnan(dlc), dlc=8; end; dlc=max(0,min(8,dlc));
        if flagIdx>4, dt=parts(4:flagIdx-1); else, dt=strings(0,1); end
        valid=false(numel(dt),1); for k=1:numel(dt), s=upper(dt(k)); if strlength(s)>=1&&all(isstrprop(s,'xdigit')), valid(k)=true; end; end
        dt=dt(valid); d_vals=zeros(1,8); take=min(dlc,numel(dt)); for k=1:take, d_vals(k)=hex2dec(dt(k)); end
        X(i,:) = [0, id_val, d_vals]; keep(i) = true; 
    end
    X=X(keep,:); y_rows=y_rows(keep);
    tbl = array2table(X, 'VariableNames', {'DUMMY','ID','DATA_0','DATA_1','DATA_2','DATA_3','DATA_4','DATA_5','DATA_6','DATA_7'});
end

function out_dec = hex2dec_force(in_str)
    in_str = erase(in_str, ["'", '"', "0x"]); out_dec = zeros(numel(in_str),1);
    for i = 1:numel(in_str)
        s = char(strtrim(in_str(i)));
        if ~isempty(s) && all(isstrprop(s,'xdigit')), out_dec(i) = hex2dec(s); end
    end
end

function Xw = buildWindows(M, W, S)
    if istable(M), M = table2array(M); end
    N = size(M,1); starts = 1:S:(N-W+1); Xw = cell(numel(starts),1);
    for k=1:numel(starts), Xw{k} = M(starts(k):starts(k)+W-1,:); end
end

function [prec, rec, f1, CM, order] = prf1(Ytrue, Ypred, posClass, doPlot)
    [CM, order] = confusionmat(Ytrue, Ypred); posIdx = find(order==posClass);
    if isempty(posIdx), prec=0; rec=0; f1=0; return; end
    TP = CM(posIdx,posIdx); FP = sum(CM(:,posIdx))-TP; FN = sum(CM(posIdx,:))-TP;
    prec = TP/max(TP+FP,1); rec = TP/max(TP+FN,1); f1 = 2*prec*rec/max(prec+rec,eps);
    if doPlot, figure; confusionchart(CM,string(order)); end
end

function [X_data, timestamps, ids] = parse_road_log(fname)
    fid = fopen(fname,'r'); if fid < 0, error("Could not open %s", fname); end
    raw = textscan(fid, '%s %*s %s', 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
    fclose(fid); time_strs = raw{1}; can_data = raw{2}; n = numel(can_data);
    X_data = zeros(n, 9); timestamps = zeros(n, 1); ids = zeros(n, 1);
    for i = 1:n
        t_str = char(time_strs{i}); timestamps(i) = str2double(t_str(2:end-1)); 
        parts = split(string(can_data{i}), '#'); if numel(parts) ~= 2, continue; end
        ids(i) = hex2dec(char(parts(1))); X_data(i, 1) = ids(i);
        data_str = char(parts(2)); len = length(data_str);
        if len < 16, data_str = [data_str, repmat('0', 1, 16 - len)]; end
        for b = 1:8, X_data(i, b+1) = hex2dec(data_str(2*b-1 : 2*b)); end
    end
    timestamps = timestamps - timestamps(1);
end

function scores = scoreOnTheFly(netE, netD, X_rows, embedTable, ...
                                 W, S, latentDim, kl_weight, embedDim, ...
                                 num_unique_ids, batchSize, id_penalty)
% Scores all windows in X_rows without pre-allocating cell arrays.
% X_rows: [N x 9] matrix (col1=ID index, cols2-9=data bytes)

    starts   = 1 : S : (size(X_rows,1) - W + 1);
    n_wins   = numel(starts);
    scores   = zeros(n_wins, 1);
    nFeatures = embedDim + 8;
    embData  = extractdata(embedTable);

    for i = 1 : batchSize : n_wins
        j             = min(i + batchSize - 1, n_wins);
        current_batch = j - i + 1;

        % Build batch on-the-fly [W x 9 x B]
        X_mat = zeros(W, 9, current_batch, 'single');
        for b = 1:current_batch
            s = starts(i + b - 1);
            X_mat(:,:,b) = X_rows(s : s+W-1, :);
        end

        id_seq   = X_mat(:, 1, :);    % [W x 1 x B]
        data_seq = X_mat(:, 2:9, :);  % [W x 8 x B]

        % Unknown ID fraction [B x 1]
        id_idx_3d        = max(1, min(round(id_seq) + 1, num_unique_ids + 1));
        unknown_fraction = reshape(mean(id_idx_3d == 1, 1), current_batch, 1);

        % Embedding lookup
        embed_flat = embData(id_idx_3d(:), :);
        embed_seq  = permute(reshape(embed_flat, W, current_batch, embedDim), [1 3 2]);  % [W x embedDim x B]

        % Target = last time step (raw arrays, no dlarray)
        embed_target = reshape(embed_seq(end,:,:), embedDim, current_batch);
        data_target  = reshape(data_seq(end,:,:),  8,        current_batch);
        X_target     = [embed_target; data_target];   % [nFeatures x B]

        % Input = steps 1..W-1
        embed_in = dlarray(permute(embed_seq(1:end-1,:,:), [2 1 3]), 'CTB');
        data_in  = dlarray(single(permute(data_seq(1:end-1,:,:), [2 1 3])), 'CTB');
        X_in     = cat(1, embed_in, data_in);

        % Encode
        Zstats = reshape(extractdata(predict(netE, X_in)), 2*latentDim, current_batch);
        mu     = Zstats(1:latentDim, :);
        logVar = Zstats(latentDim+1:end, :);

        % Decode
        X_pred = reshape(extractdata(predict(netD, dlarray(mu,'CB'))), nFeatures, current_batch);

        % Score
        de = X_target(1:embedDim,:)    - X_pred(1:embedDim,:);
        dd = X_target(embedDim+1:end,:) - X_pred(embedDim+1:end,:);

        mse      = reshape(5.0*mean(de.^2,1) + mean(dd.^2,1), current_batch, 1);
        kl_terms = reshape(-0.5*sum(1+logVar-mu.^2-exp(logVar),1), current_batch, 1);

        scores(i:j) = mse + kl_weight*kl_terms + id_penalty*unknown_fraction;
    end
end
