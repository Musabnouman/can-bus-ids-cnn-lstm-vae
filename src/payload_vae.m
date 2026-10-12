%% ==========================================
% 0. CONFIGURATION & LOADING
% ==========================================
clear; clc; 
rng(42);
dataset_mode = 'CIC';              
attack_filename = 'decimal_DoS.csv'; 
canUSEGPU=false;
if strcmp(dataset_mode, 'CIC')
    benign_filename = 'decimal_benign.csv';
elseif strcmp(dataset_mode, 'CARHACK')
    benign_filename = 'normal_run_data.txt';
elseif strcmp(dataset_mode, 'OTIDS')
    benign_filename = 'Attack_free_dataset.txt'; 
end

target_cols = {'DATA_0','DATA_1','DATA_2','DATA_3','DATA_4','DATA_5','DATA_6','DATA_7'};

% : CNN-LSTM VAE (PREDICTION + KL + ATTENTION)
W = 32;          % Window Size
S = 1;           % Stride
latentDim = 16;   
hiddenSize = 128; 
learningRate = 1e-4;
maxEpochs = 30;  
batchSize = 256; 
kl_weight = 5e-5;

%% ==========================================
% 1. LOAD & PARSING
% ==========================================
fprintf('Loading data...\n');

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
    % CARHACK LOADER
    opts_benign = detectImportOptions(benign_filename, 'FileType', 'text');
    opts_benign.Delimiter = {' ', '\t'};
    opts_benign.ConsecutiveDelimitersRule = 'join';
    opts_benign.LeadingDelimitersRule = 'ignore';
    opts_benign.DataLines = [1 Inf];
    opts_benign.VariableNamesLine = 0;
    
    raw_benign_table = readtable(benign_filename, opts_benign, 'ReadVariableNames', false);
    
    benign_d = zeros(height(raw_benign_table), 8);
    for k=1:8
        col_idx = 7 + k;
        if col_idx <= width(raw_benign_table)
            benign_d(:,k) = hex2dec_force(string(raw_benign_table{:, col_idx})); 
        end
    end
    decimalbenign = array2table(benign_d, 'VariableNames', target_cols);

    [attack_data_full, Y_attack_rows] = parse_carhack_attack_csv(attack_filename);
    
    % Ensure columns match 
    if width(attack_data_full) > 8
        attack_data_full = attack_data_full(:, 2:9); 
        attack_data_full.Properties.VariableNames = target_cols;
    end
    
    % SEPARATE Mixed Attack File (R=Benign, T=Attack)
    X_attack_raw = table2array(attack_data_full);
    
    % 'T' implies Attack (1), 'R' implies Benign (0)
    idx_T = (Y_attack_rows == 1);
    idx_R = (Y_attack_rows == 0);
    
    X_attack_T = X_attack_raw(idx_T, :); % True Attacks
    X_attack_R = X_attack_raw(idx_R, :); % Normal messages hidden in attack file
    
    X_benign_all = table2array(decimalbenign);
    
    fprintf('CarHack: Separated %d Normal (R) and %d Injected (T) from attack file.\n', ...
        sum(idx_R), sum(idx_T));

elseif strcmp(dataset_mode, 'OTIDS')
    % 1. Load OTIDS files (9 columns: [ID, D0..D7])
    [X_benign_full, ~, ~] = parse_otids_txt(benign_filename);
    [X_attack_full, ~, T_attack] = parse_otids_txt(attack_filename);
    
    % 2. Generate Ground Truth Labels using the ID (Column 1)
    Y_attack_rows = zeros(size(X_attack_full, 1), 1);
    
    if contains(lower(attack_filename), 'dos')
        Y_attack_rows = (X_attack_full(:, 1) == 0);
    elseif contains(lower(attack_filename), 'impersonation')
        Y_attack_rows = (X_attack_full(:, 1) == hex2dec('164')) & (T_attack >= 250);
    else
        % benign_ids = unique(X_benign_full(:, 1));
        % is_unknown_id = ~ismember(X_attack_full(:, 1), benign_ids);
        Y_attack_rows =(T_attack >= 250);
    end
    
    % 3. Separate streams
    idx_T = (Y_attack_rows == 1);
    idx_R = (Y_attack_rows == 0);
    
    % 4. STRIP THE ID COLUMN! (Keep only Cols 2-9 for DATA_0 to DATA_7)
    X_attack_T = X_attack_full(idx_T, 2:9); 
    X_attack_R = X_attack_full(idx_R, 2:9); 
    X_benign_all = X_benign_full(:, 2:9);
    X_attack_all = X_attack_full(:, 2:9); 
    
    fprintf('OTIDS: Separated %d Normal (R) and %d Injected (T)\n', sum(idx_R), sum(idx_T));
end

if strcmp(dataset_mode, 'CARHACK') || strcmp(dataset_mode, 'OTIDS')
    X_attack_all = X_attack_T; % Use only the T part as the "Attack Source"
end

% Handle missing
X_benign_all(isnan(X_benign_all)) = 0;
X_attack_all(isnan(X_attack_all)) = 0;


%% ==========================================
% IMPROVED FEATURE SELECTION
% ==========================================
fprintf('\n--- MUTUAL INFORMATION FEATURE SELECTION ---\n');

% Prepare labeled sample for feature selection
sample_size = min(20000, size(X_benign_all, 1));
benign_sample_idx = randperm(size(X_benign_all, 1), sample_size);
attack_sample_idx = randperm(size(X_attack_all, 1), min(sample_size, size(X_attack_all, 1)));

X_fs = [X_benign_all(benign_sample_idx, :); 
        X_attack_all(attack_sample_idx, :)];
Y_fs = [zeros(sample_size, 1); ones(length(attack_sample_idx), 1)];

% Compute Mutual Information for each feature
nFeatures_raw = size(X_fs, 2);
mi_scores = zeros(nFeatures_raw, 1);


for i = 1:nFeatures_raw
    [~, edges] = discretize(X_fs(:, i), 10);
    X_discrete = discretize(X_fs(:, i), edges);
    mi_scores(i) = mutualInformation(X_discrete, Y_fs);
end

[mi_sorted, mi_idx] = sort(mi_scores, 'descend');
top_k = min(8, nFeatures_raw); 
keep_idx = mi_idx(1:top_k);

fprintf('Top %d features by Mutual Information:\n', top_k);
feat_names = target_cols;
for k = 1:length(keep_idx)
    fprintf('  %s (MI=%.4f)\n', feat_names{keep_idx(k)}, mi_sorted(k));
end

% Apply selection
X_benign_all = X_benign_all(:, keep_idx);
X_attack_all = X_attack_all(:, keep_idx);
if strcmp(dataset_mode, 'CARHACK')
    X_attack_R = X_attack_R(:, keep_idx);
end

global_mu  = mean(X_benign_all, 1);       
global_sig = std(X_benign_all, 0, 1);     
global_sig(global_sig==0) = 1; 
nFeatures = length(keep_idx);





% %% ==========================================
% % 3. PARTITIONING (TRAIN / VAL / OPT / TEST)
% % ==========================================
% % 1. Split Benign Source
% n_benign = size(X_benign_all, 1);
% %idx_b = randperm(n_benign);
% 
% %Train: 60%, Val(Loss): 10%, Opt(Thresh): 10%, Test: 20%
% n_train = floor(0.60 * n_benign);
% n_val   = floor(0.20 * n_benign); 
% %n_opt   = floor(0.10 * n_benign); 
% n_test  = n_benign - n_train - n_val; %- n_opt;
% 
% X_train_rows      = X_benign_all(idx_b(1:n_train), :);
% X_val_rows        = X_benign_all(idx_b(n_train+1 : n_train+n_val), :);
% %X_opt_benign_rows = X_benign_all(idx_b(n_train+n_val+1 : n_train+n_val+n_opt), :);
% X_test_benign_rows= X_benign_all(idx_b(n_train+n_val+1 : end), :);
% 
% % For CARHACK, add the 'R' rows (Normal from attack file) to the Benign Test Set
% if strcmp(dataset_mode, 'CARHACK')
%     X_test_benign_rows = [X_test_benign_rows; X_attack_R];
% end
% 
% % 2. Split Attack Source (Purely for Testing now)
% X_test_attack_rows = X_attack_all;
% 
% % 3. Build Windows
% fprintf('Building Windows...\n');
% Xw_train      = buildWindows(X_train_rows, W, S);
% Xw_val        = buildWindows(X_val_rows, W, S); 
% %Xw_opt_benign = buildWindows(X_opt_benign_rows, W, S);
% 
% Xw_test_benign = buildWindows(X_test_benign_rows, W, S);
% Xw_test_attack = buildWindows(X_test_attack_rows, W, S);
% 
% fprintf('Data Splits:\n');
% fprintf('  Train: %d\n  Val(Loss): %d\n  Opt(Thresh): %d (Benign Only)\n  Test(Benign): %d\n  Test(Attack): %d\n', ...
%     numel(Xw_train), numel(Xw_val),... %numel(Xw_opt_benign), ...
%     numel(Xw_test_benign), numel(Xw_test_attack));

% without randperm new
%% ==========================================
% 3. PARTITIONING (TRAIN / VAL / OPT / TEST)
% ==========================================
% 1. Split Benign Source
n_benign = size(X_benign_all, 1);

% Strict Chronological Split (No idx_b, No randperm)
n_train = floor(0.60 * n_benign);
n_val   = floor(0.20 * n_benign); 

X_train_rows       = X_benign_all(1 : n_train, :);
X_val_rows         = X_benign_all(n_train+1 : n_train+n_val, :);
X_test_benign_rows = X_benign_all(n_train+n_val+1 : end, :);

% For CARHACK, add the 'R' rows (Normal from attack file) to the Benign Test Set
if strcmp(dataset_mode, 'CARHACK')
    X_test_benign_rows = [X_test_benign_rows; X_attack_R];
end

% 2. Split Attack Source (Purely for Testing now)
X_test_attack_rows = X_attack_all;

% 3. Build Windows
fprintf('Building Windows...\n');

Xw_train       = buildWindows(X_train_rows, W, S);
Xw_val         = buildWindows(X_val_rows, W, S); 
Xw_test_benign = buildWindows(X_test_benign_rows, W, S);
Xw_test_attack = buildWindows(X_test_attack_rows, W, S);

fprintf('Data Splits:\n');
fprintf('  Train: %d\n  Val(Loss): %d\n  Test(Benign): %d\n  Test(Attack): %d\n', ...
    numel(Xw_train), numel(Xw_val), numel(Xw_test_benign), numel(Xw_test_attack));


%% ==========================================
% 4. VAE TRAINING
% ==========================================
% ENCODER
numHeads = 4;
lgraphE = layerGraph();
lgraphE = addLayers(lgraphE, sequenceInputLayer(nFeatures, "Name","in"));
lgraphE = addLayers(lgraphE, convolution1dLayer(3, 32, "Padding","same", "Name","cnn"));
lgraphE = addLayers(lgraphE, reluLayer("Name","relu"));
lgraphE = addLayers(lgraphE, lstmLayer(hiddenSize,"OutputMode","sequence","Name","lstm_enc"));
lgraphE = addLayers(lgraphE, selfAttentionLayer(numHeads, hiddenSize, "Name","attn"));
lgraphE = addLayers(lgraphE, globalAveragePooling1dLayer("Name","gap")); 
lgraphE = addLayers(lgraphE, fullyConnectedLayer(2*latentDim,"Name","fc_mu_logvar"));

lgraphE = connectLayers(lgraphE, "in",        "cnn");
lgraphE = connectLayers(lgraphE, "cnn",       "relu");
lgraphE = connectLayers(lgraphE, "relu",      "lstm_enc");
lgraphE = connectLayers(lgraphE, "lstm_enc",  "attn");
lgraphE = connectLayers(lgraphE, "attn",      "gap");
lgraphE = connectLayers(lgraphE, "gap",       "fc_mu_logvar");

netE = dlnetwork(lgraphE);

% DECODER (Predicts Next Step)
lgraphD = layerGraph();
lgraphD = addLayers(lgraphD, featureInputLayer(latentDim,"Name","z"));
lgraphD = addLayers(lgraphD, fullyConnectedLayer(hiddenSize,"Name","fc1"));
lgraphD = addLayers(lgraphD, reluLayer("Name","relu1"));
lgraphD = addLayers(lgraphD, fullyConnectedLayer(nFeatures,"Name","fc_out")); % Outputs 1xFeatures

lgraphD = connectLayers(lgraphD,"z","fc1");
lgraphD = connectLayers(lgraphD,"fc1","relu1");
lgraphD = connectLayers(lgraphD,"relu1","fc_out");

netD = dlnetwork(lgraphD);

% Optimization
avgGradientsE = []; avgGradientsSqE = [];
avgGradientsD = []; avgGradientsSqD = [];
iter = 1;

% Early stopping parameters
patience = 5;
min_delta = 1e-4; 
best_val_loss = inf;
patience_counter = 0;
best_netE = netE;
best_netD = netD;

fprintf('\nStarting VAE Training (Next-Step Prediction)...\n');

for epoch = 1:maxEpochs
    % Training loop
    idx = randperm(numel(Xw_train));
    epoch_losses = [];
    
    for i = 1:batchSize:numel(Xw_train)
        j = min(i+batchSize-1, numel(Xw_train));
        batch_indices = idx(i:j);
        
        raw_batch = Xw_train(batch_indices);
        X_mat = cat(3, raw_batch{:});
        X_mat = (X_mat - reshape(global_mu, [1, nFeatures, 1])) ./ ...
                reshape(global_sig, [1, nFeatures, 1]);
        X_dl = dlarray(permute(X_mat, [2, 1, 3]), 'CTB');
        
         [loss, gradientsE, gradientsD] = dlfeval(@modelGradients, netE, netD, ...
                                                 X_dl, latentDim, W, kl_weight);
        
        [netE, avgGradientsE, avgGradientsSqE] = adamupdate(netE, gradientsE, ...
                                                            avgGradientsE, avgGradientsSqE, iter, learningRate);
        [netD, avgGradientsD, avgGradientsSqD] = adamupdate(netD, gradientsD, ...
                                                            avgGradientsD, avgGradientsSqD, iter, learningRate);
        iter = iter + 1;
        epoch_losses = [epoch_losses; extractdata(loss)];
    end
    
    % Validation (On pure benign data)
    err_val = getAnomalyScoreBatched(netE, netD, Xw_val, latentDim, W, ...
                                     kl_weight, 512, global_mu, global_sig);
    valLoss = mean(err_val);
    trainLoss = mean(epoch_losses);
    
    fprintf('Epoch %d/%d | Train: %.4f | Val: %.4f', ...
            epoch, maxEpochs, trainLoss, valLoss);
    
    % Early stopping check
    if valLoss < (best_val_loss - min_delta)
        best_val_loss = valLoss;
        best_netE = netE;
        best_netD = netD;
        patience_counter = 0;
        fprintf(' ← BEST\n');
    else
        patience_counter = patience_counter + 1;
        fprintf(' (patience: %d/%d)\n', patience_counter, patience);
        
        if patience_counter >= patience
            fprintf('\n⚠ Early stopping triggered at epoch %d\n', epoch);
            break;
        end
    end
end

% Restore best model
netE = best_netE;
netD = best_netD;
fprintf('\nRestored best model (Val Loss: %.4f)\n', best_val_loss);





%% ==========================================
% 6. THRESHOLD OPTIMIZATION (BENIGN ONLY)
% ==========================================
fprintf('\n--- THRESHOLD OPTIMIZATION (No Attack Data) ---\n');

% Calculate scores for Benign Optimization set
%scores_opt_benign = getAnomalyScoreBatched(netE, netD, Xw_opt_benign, latentDim, W, kl_weight, 512, global_mu, global_sig);

% Define Target False Positive Rate (FPR)
% We want to accept e.g. 99% of benign data as Normal.
%target_FPR = 0.01; 

% Sort benign scores to find the cutoff
%sorted_scores = sort(scores_opt_benign, 'ascend');
%n_samples = numel(sorted_scores);

% Find index corresponding to (1 - target_FPR)
%cutoff_idx = floor(n_samples * (1 - target_FPR));
%cutoff_idx = max(1, min(n_samples, cutoff_idx));

%best_thr = sorted_scores(cutoff_idx);
% 1. Validation Scores
err_val = getAnomalyScoreBatched(netE, netD, Xw_val, latentDim, W, kl_weight, 512, global_mu, global_sig);

% 2. Thresholding
targetFPR = 0.01;
thr = quantile(err_val, 1 - targetFPR);
fprintf('Threshold (Val FPR 1%%): %.6f\n', thr);

fprintf('Target FPR: %.2f%%\n', targetFPR * 100);
%fprintf('Optimization complete. Selected Threshold: %.6f\n', best_thr);


%% ==========================================
% 7. EVALUATION (BATCH TESTING)
% ==========================================
fprintf('\n--- BATCH TESTING (Separated Streams) ---\n');

% 1. Run Benign Batch
fprintf('Testing on Benign Stream... ');
scores_benign = getAnomalyScoreBatched(netE, netD, Xw_test_benign, latentDim, W, kl_weight, 512, global_mu, global_sig);
fp_count = sum(scores_benign > thr);
tn_count = sum(scores_benign <= thr);
fpr = fp_count / (fp_count + tn_count);
fprintf('Done.\n');

% 2. Run Attack Batch
fprintf('Testing on Attack Stream... ');
scores_attack = getAnomalyScoreBatched(netE, netD, Xw_test_attack, latentDim, W, kl_weight, 512, global_mu, global_sig);
tp_count = sum(scores_attack > thr);
fn_count = sum(scores_attack <= thr);
detection_rate = tp_count / (tp_count + fn_count); % Recall
fprintf('Done.\n');


scores_benign = scores_benign(:);
scores_attack = scores_attack(:);

err_test = [scores_benign; scores_attack];

Yw_test = [
    repmat("benign",   numel(scores_benign), 1);
    repmat("malicious",numel(scores_attack), 1)
];

Yw_test = categorical(Yw_test);

% 4. Metrics
Yhat = categorical(err_test > thr, [false true], ["benign","malicious"]);

[prec, rec, f1, CM, order] = prf1(Yw_test, Yhat, "malicious", true);
acc = mean(Yw_test == Yhat);

isBen = (Yw_test=="benign");
fpr_benign = sum(Yhat(isBen)=="malicious") / max(sum(isBen),1);

[FPR, TPR, ~, AUC] = perfcurve(double(Yw_test=="malicious"), err_test, 1);

fprintf('\n[TEST RESULTS]\n Accuracy=%.3f Precision=%.3f Recall=%.3f F1=%.3f | AUC=%.4f | Benign FPR=%.4f\n', ...
    acc, prec, rec, f1, AUC, fpr_benign);

% 3. Aggregate Metrics
fprintf('\n[FINAL RESULTS]\n');
fprintf('Threshold:       %.6f\n', thr);
fprintf('Benign Samples:  %d\n', numel(scores_benign));
fprintf('Attack Samples:  %d\n', numel(scores_attack));
fprintf('-----------------------\n');
fprintf('False Positives: %d\n', fp_count);
fprintf('True Negatives:  %d\n', tn_count);
fprintf('FPR (False Alarm): %.4f\n', fpr);
fprintf('-----------------------\n');
fprintf('True Positives:  %d\n', tp_count);
fprintf('False Negatives: %d\n', fn_count);
fprintf('Recall (DR):       %.4f\n', detection_rate);

total_samples = numel(scores_benign) + numel(scores_attack);
accuracy = (tp_count + tn_count) / total_samples;
precision = tp_count / (tp_count + fp_count + eps);
f1_final = 2 * (precision * detection_rate) / (precision + detection_rate + eps);

fprintf('-----------------------\n');
fprintf('Accuracy:   %.4f\n', accuracy);
fprintf('Precision:  %.4f\n', precision);
fprintf('F1 Score:   %.4f\n', f1_final);



%% Plots for result section
%% SAVE: Precision-Recall Curve
[rec_curve, prec_curve, ~, AUPR] = perfcurve(...
    double(Yw_test == "malicious"), ...
    err_test, 1, ...
    'XCrit', 'reca', ...
    'YCrit', 'prec');

figure('Position', [100 100 600 550]);

plot(rec_curve, prec_curve, 'r-', 'LineWidth', 2, ...
     'DisplayName', sprintf('PR Curve (AUPR = %.4f)', AUPR));
hold on;

% Baseline: random classifier line (= proportion of positives)
baseline = sum(Yw_test == "malicious") / numel(Yw_test);
yline(baseline, 'k:', 'LineWidth', 1.2, ...
      'DisplayName', sprintf('Baseline (%.4f)', baseline));

xlabel('Recall', 'FontSize', 11);
ylabel('Precision', 'FontSize', 11);
title(['Precision-Recall Curve — ' dataset_mode ' — ' ...
       strrep(attack_filename, '_', ' ')], 'FontSize', 11);
legend('Location', 'southwest', 'FontSize', 10);
grid on; box on;
xlim([0 1]); ylim([0 1]);

plot_name = ['PR_' dataset_mode '_' attack_filename(1:end-4)];
saveas(gcf, [plot_name '.png']);
saveas(gcf, [plot_name '.pdf']);
fprintf('Saved: %s\n', plot_name);
close;
% AUC-ROC curve
figure('Position', [100 100 600 550]);

plot(FPR, TPR, 'b-', 'LineWidth', 2, ...
     'DisplayName', sprintf('ROC Curve (AUC = %.4f)', AUC));
hold on;
plot([0 1], [0 1], 'k:', 'LineWidth', 1.2, ...
     'DisplayName', 'Random Classifier');

xlabel('False Positive Rate', 'FontSize', 11);
ylabel('True Positive Rate', 'FontSize', 11);
title(['AUC-ROC Curve '])% dataset_mode ' — ' ...
       %strrep(attack_filename, '_', ' ')], 'FontSize', 11);
legend('Location', 'southeast', 'FontSize', 10);
grid on; box on;
xlim([0 1]); ylim([0 1]);

xlim([0.00 1.00])
ylim([0.00 1.00])

 plot_name = ['ROC_' dataset_mode '_' attack_filename(1:end-4)];
 saveas(gcf, [plot_name '.png']);
 saveas(gcf, [plot_name '.pdf']);
 fprintf('Saved: %s\n', plot_name);
 close;
%%  Raw scores for combined plots later
save_name = ['rawscores_' dataset_mode '_' attack_filename(1:end-4) '.mat'];
save(save_name, 'scores_benign', 'scores_attack', 'thr', ...
                'FPR', 'TPR', 'AUC', 'Yw_test', 'Yhat');
fprintf('Raw scores saved: %s\n', save_name);


%% ==========================================
% SUPPORT FUNCTIONS
% ==========================================
function [loss, gradE, gradD] = modelGradients(netE, netD, X, latentDim, W, kl_weight)
    % TRUE PREDICTION SETUP:
    % Input:  Steps 1 to W-1
    % Target: Step W (The immediate next value)
    
    % A formatted dlarray is stored in 'SCBTU' order, so this 'CTB' batch is
    % held as C x B x T. Time is the 3rd dimension; indexing the 2nd one
    % would pick windows of the batch instead of time steps.
    X_in     = X(:, :, 1:end-1);              % C x B x (W-1): steps 1..W-1
    X_target = stripdims(X(:, :, end));       % C x B: step W
    
    Zstats = forward(netE, X_in);
    Zstats = reshape(Zstats, 2*latentDim, []);
    mu     = Zstats(1:latentDim, :);
    logVar = Zstats(latentDim+1:end, :);
    
    epsilon = randn(size(mu), 'like', X);
    sigma = exp(0.5 * logVar);
    Z = mu + sigma .* epsilon;
    
    % Decoder predicts the NEXT step based on latent Z
    X_pred = stripdims(forward(netD, dlarray(Z,'CB')));   % C x B
    
    mse = mean((X_target - X_pred).^2, 'all');
    kl = -0.5 * mean(sum(1 + logVar - mu.^2 - exp(logVar), 1), 'all');
    
    loss = mse + kl_weight * kl;
    
    [gradE, gradD] = dlgradient(loss, netE.Learnables, netD.Learnables);
end

function scores = getAnomalyScoreBatched(netE, netD, X_cells, latentDim, W, kl_weight, batchSize, g_mu, g_sig)
    n = numel(X_cells);
    scores = nan(n,1);   % NaN, not 0: a window that is never scored must not pass as benign
    nFeatures = size(X_cells{1},2);

    for i = 1:batchSize:n
        j = min(i+batchSize-1, n);
        
        X_mat = cat(3, X_cells{i:j});                 
        X_mat = (X_mat - reshape(g_mu, [1, nFeatures, 1])) ./ ...
                          reshape(g_sig, [1, nFeatures, 1]);

        % Split input and target on the raw W x C x B array, before any
        % dlarray labelling ('CTB' is stored as C x B x T, see modelGradients).
        X_target = reshape(X_mat(end, :, :), nFeatures, []);              % C x B: step W
        X_in_dl  = dlarray(permute(X_mat(1:end-1, :, :), [2 1 3]), 'CTB'); % steps 1..W-1

        % Encoder -> mu, logVar
        Zstats = extractdata(predict(netE, X_in_dl));
        Zstats = reshape(Zstats, 2*latentDim, []);
        mu     = Zstats(1:latentDim,:);
        logVar = Zstats(latentDim+1:end,:);
        
        % Decoder -> Prediction (Next Step)
        X_pred_dl = predict(netD, dlarray(mu,'CB'));    
        X_pred = extractdata(X_pred_dl);             
        X_pred = reshape(X_pred, nFeatures, []);        

        % Loss Calculation
        diff = X_target - X_pred;                       
        mse  = mean(diff.^2, 1);                        

        kl_terms = -0.5 * sum(1 + logVar - mu.^2 - exp(logVar), 1); 
        
        batch_scores = mse + kl_weight * kl_terms;
        
        assert(numel(batch_scores) == j-i+1, ...
            'Scored %d windows in a batch of %d.', numel(batch_scores), j-i+1);
        scores(i:j) = batch_scores(:);
    end
end

function [tbl, y_rows] = parse_carhack_attack_csv(fname)
    fid = fopen(fname,'r');
    if fid < 0, error("Could not open %s", fname); end
    raw = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
    fclose(fid);
    lines = string(raw{1}); n = numel(lines);
    X = zeros(n, 9); y_rows = zeros(n, 1); keep = false(n,1);
    for i = 1:n
        line_str = strtrim(lines(i));
        if strlength(line_str) < 5, continue; end
        parts = split(line_str, ",");
        parts = strtrim(erase(parts, ['"'; "'"]));
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
        X(i,:) = [id_val, d_vals]; keep(i) = true;
    end
    X=X(keep,:); y_rows=y_rows(keep);
    tbl = array2table(X, 'VariableNames', {'ID','DATA_0','DATA_1','DATA_2','DATA_3','DATA_4','DATA_5','DATA_6','DATA_7'});
end

function out_dec = hex2dec_force(in_str)
    in_str = erase(in_str, ["'", '"', "0x"]);
    out_dec = zeros(numel(in_str),1);
    for i = 1:numel(in_str)
        s = char(strtrim(in_str(i)));
        if ~isempty(s) && all(isstrprop(s,'xdigit')), out_dec(i) = hex2dec(s); end
    end
end

function Xw = buildWindows(M, W, S)
    if istable(M), M = table2array(M); end
    N = size(M,1); starts = 1:S:(N-W+1);
    Xw = cell(numel(starts),1);
    for k=1:numel(starts), Xw{k} = M(starts(k):starts(k)+W-1,:); end
end

function [prec, rec, f1, CM, order] = prf1(Ytrue, Ypred, posClass, doPlot)
    [CM, order] = confusionmat(Ytrue, Ypred);
    posIdx = find(order==posClass);
    if isempty(posIdx), prec=0; rec=0; f1=0; return; end
    TP = CM(posIdx,posIdx); FP = sum(CM(:,posIdx))-TP; FN = sum(CM(posIdx,:))-TP;
    prec = TP/max(TP+FP,1); rec = TP/max(TP+FN,1);
    f1 = 2*prec*rec/max(prec+rec,eps);
    if doPlot, figure; confusionchart(CM,string(order)); end
end
% road parser

function [X_matrix, y_rows, timestamps] = parse_otids_txt(fname)
    fid = fopen(fname, 'r');
    if fid < 0, error("Could not open %s", fname); end
    
    raw = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    
    lines = string(raw{1});
    n = numel(lines);
    
    X_matrix = zeros(n, 9);
    y_rows = zeros(n, 1);
    timestamps = zeros(n, 1); 
    keep = false(n, 1);
    
    for i = 1:n
        str = strtrim(lines(i));
        if strlength(str) < 10, continue; end
        
        parts = split(str);
        
        if numel(parts) >= 7 && parts(3) == "ID:" && parts(6) == "DLC:"
            timestamps(i) = str2double(parts(2));
            
            id_hex = char(parts(4));
            id_val = hex2dec(id_hex);
            
            dlc = str2double(parts(7));
            dlc = max(0, min(8, dlc));
            
            data_vals = zeros(1, 8);
            for d = 1:dlc
                if 7+d <= numel(parts)
                    data_vals(d) = hex2dec(char(parts(7+d)));
                end
            end
            
            X_matrix(i, :) = [id_val, data_vals];
            keep(i) = true;
        end
    end
    
    X_matrix = X_matrix(keep, :);
    y_rows = y_rows(keep);
    timestamps = timestamps(keep); 
end
% Mutual Information helper function
function mi = mutualInformation(X, Y)
    valid = ~isnan(X) & ~isnan(Y);
    X = X(valid); Y = Y(valid);
    if numel(unique(X)) < 2, mi = 0; return; end
    [~, edges] = discretize(X, 10);
    X_discrete = discretize(X, edges);
    pX = histcounts(X_discrete, 'Normalization', 'probability');
    pY = histcounts(Y, 'Normalization', 'probability');
    [~, ~, pXY] = histcounts2(X_discrete, Y, 'Normalization', 'probability');
    if isempty(pXY) || any(size(pXY)==0), mi = 0; return; end
    mi = 0;
    for i = 1:length(pX)
        for j = 1:length(pY)
            if i <= size(pXY,1) && j <= size(pXY,2)
                if pXY(i,j) > 0 && pX(i) > 0 && pY(j) > 0
                    mi = mi + pXY(i,j) * log2(pXY(i,j) / (pX(i) * pY(j)));
                end
            end
        end
    end
end
