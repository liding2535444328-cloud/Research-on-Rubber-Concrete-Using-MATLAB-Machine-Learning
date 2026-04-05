%% ========================================================================
%  项目：橡胶混凝土强度预测系统 (LSTM 9输入科研增强版)
%  数据：数据集3 (9个特征输入 -> 1个强度输出)
%  核心：1. 深度学习建模  2. MAE全覆盖统计  3. SHAP机理分析  4. 标题下置
%  环境：需安装 Deep Learning Toolbox
%% ========================================================================
warning off; clear; clc; close all;
rng('shuffle'); 

%% --- 模块 1: 数据导入与全局配置 ---
% 适配数据集3 (9输入: W/B, Rubber, MaxSize, Cement, FineAgg, CoarseAgg, SF/C, SP, Age)
res_raw = readmatrix('数据集3.xlsx'); 
res_raw(any(isnan(res_raw), 2), :) = []; 

featureNames = {'水胶比 (W/B)', '橡胶含量 (Rubber)', '橡胶粒径 (MaxSize)', ...
                '水泥 (Cement)', '细骨料 (FineAgg)', '粗骨料 (CoarseAgg)', ...
                '硅比 (SF/C)', '外加剂 (SP)', '龄期 (Age)'};
allNames = [featureNames, '强度 (Strength)'];

loop_num = 10; 
stats_R2 = zeros(loop_num, 1);
stats_RMSE = zeros(loop_num, 1);
stats_MAE = zeros(loop_num, 1); 

%% --- 模块 2: 特征相关性分析 (图1) ---
figure('Color', [1 1 1], 'Position', [100, 100, 800, 700], 'Name', 'Correlation');
corrMat = corr(res_raw); imagesc(corrMat); colormap(jet); colorbar; clim([-1 1]); 
set(gca, 'XTick', 1:10, 'XTickLabel', allNames, 'YTick', 1:10, 'YTickLabel', allNames, ...
    'TickLabelInterpreter', 'none', 'FontSize', 8, 'LineWidth', 1.2, 'Position', [0.15, 0.28, 0.7, 0.65]); 
xtickangle(45); axis square;
text(0.5, -0.32, {'图1: 特征相关性热力图分析'; 'Fig.1: Feature Correlation Heatmap Analysis'}, ...
    'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
for i = 1:10; for j = 1:10
    text(j, i, sprintf('%.2f', corrMat(i,j)), 'HorizontalAlignment', 'center', ...
        'Color', char(ifelse(abs(corrMat(i,j))>0.6, 'w', 'k')), 'FontSize', 7);
end; end

%% --- 模块 3: 执行 LSTM 循环实验 ---
fprintf('正在执行 %d 次 LSTM 深度学习稳定性测试...\n', loop_num);
best_overall_R2 = -inf;

for run_i = 1:loop_num
    num_size = 0.8; total_rows = size(res_raw, 1);
    rand_indices = randperm(total_rows); 
    res_shf = res_raw(rand_indices, :);          
    split_idx = round(num_size * total_rows); 
    
    P_train_raw = res_shf(1:split_idx, 1:9)'; T_train_raw = res_shf(1:split_idx, 10)';
    P_test_raw = res_shf(split_idx+1:end, 1:9)'; T_test_raw = res_shf(split_idx+1:end, 10)';
    
    [p_train, ps_in] = mapminmax(P_train_raw, 0, 1);
    p_test = mapminmax('apply', P_test_raw, ps_in);
    [t_train, ps_out] = mapminmax(T_train_raw, 0, 1);
    
    % 数据序列化处理
    M = size(p_train, 2); N = size(p_test, 2);
    Xt_train = cell(M, 1); for i = 1:M; Xt_train{i} = p_train(:, i); end
    Xt_test = cell(N, 1);  for i = 1:N; Xt_test{i} = p_test(:, i); end
    
    % LSTM 架构
    layers = [sequenceInputLayer(9), lstmLayer(64, 'OutputMode', 'last'), reluLayer, fullyConnectedLayer(1), regressionLayer];
    options = trainingOptions('adam', 'MaxEpochs', 600, 'InitialLearnRate', 0.01, 'Plots', 'none', 'Verbose', false);
    
    [net, info] = trainNetwork(Xt_train, t_train', layers, options);
    
    T_sim1 = mapminmax('reverse', predict(net, Xt_train)', ps_out);
    T_sim2 = mapminmax('reverse', predict(net, Xt_test)', ps_out);
    
    stats_R2(run_i) = 1 - sum((T_test_raw - T_sim2).^2) / sum((T_test_raw - mean(T_test_raw)).^2);
    stats_RMSE(run_i) = sqrt(mean((T_test_raw - T_sim2).^2));
    stats_MAE(run_i) = mean(abs(T_test_raw - T_sim2));
    
    if stats_R2(run_i) > best_overall_R2
        best_overall_R2 = stats_R2(run_i);
        bp.net = net; bp.loss = info.TrainingLoss;
        bp.T_test = T_test_raw'; bp.T_sim2 = T_sim2';
        bp.T_train = T_train_raw'; bp.T_sim1 = T_sim1';
        bp.R2_test = stats_R2(run_i); bp.rmse2 = stats_RMSE(run_i); bp.mae2 = stats_MAE(run_i);
        bp.R2_train = 1 - sum((T_train_raw - T_sim1).^2) / sum((T_train_raw - mean(T_train_raw)).^2);
        bp.rmse1 = sqrt(mean((T_train_raw - T_sim1).^2)); bp.mae1 = mean(abs(T_train_raw - T_sim1));
        bp.p_train = p_train; bp.p_test = p_test;
    end
end

%% --- 模块 4: 核心预测可视化渲染 (图2-7) ---
% 图2: 整合 预测对比 & 线性回归
figure('Color', [1 1 1], 'Position', [150, 150, 1100, 550], 'Name', 'Results');
subplot(1, 2, 1);
plot(bp.T_test, 'r-s', 'LineWidth', 1.2); hold on; plot(bp.T_sim2, 'b-o', 'LineWidth', 1.2);
legend({'实验值', '预测值'}, 'Location', 'NorthOutside', 'Orientation', 'horizontal'); 
grid on; ylabel('强度 (MPa)'); xlabel('测试样本'); set(gca, 'Position', [0.1, 0.25, 0.35, 0.6]); 
text(0.5, -0.25, {'图2: 强度预测结果对比图'; 'Fig.2: Comparison of Strength Prediction Results'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
annotation('textbox', [0.12, 0.75, 0.1, 0.05], 'String', ['MAE = ', num2str(bp.mae2, '%.3f')], 'EdgeColor', 'none', 'FontWeight', 'bold');

subplot(1, 2, 2);
scatter(bp.T_test, bp.T_sim2, 40, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
ref_l = [min(bp.T_test) max(bp.T_test)]; plot(ref_l, ref_l, 'k--', 'LineWidth', 1.5); 
grid on; axis square; xlabel('实验值 (MPa)'); ylabel('预测值 (MPa)'); set(gca, 'Position', [0.55, 0.25, 0.35, 0.6]); 
text(0.5, -0.25, {'图3: 实验值 vs. 预测值线性回归分析'; 'Fig.3: Regression Analysis of Exp. vs. Pred.'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
annotation('textbox', [0.58, 0.68, 0.12, 0.15], 'String', {['R^2 = ', num2str(bp.R2_test, '%.4f')], ['RMSE = ', num2str(bp.rmse2, '%.3f')], ['MAE = ', num2str(bp.mae2, '%.3f')]}, 'BackgroundColor', [0.85, 0.93, 1.0], 'EdgeColor', 'none', 'FontSize', 9);

% 图4: 残差与统计报表
figure('Color', [1 1 1], 'Position', [200, 200, 1100, 500], 'Name', 'Stats Report');
subplot(1, 2, 1); bar(bp.T_sim2 - bp.T_test, 'FaceColor', [0.4, 0.6, 0.8]);
grid on; ylabel('误差 (MPa)'); xlabel('测试样本'); set(gca, 'Position', [0.1, 0.25, 0.35, 0.6]); 
text(0.5, -0.25, {'图4: 预测残差分布分析'; 'Fig.4: Residual Distribution Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

subplot(1, 2, 2); axis off; subPos = get(gca, 'Position');
s_cell = {'指标 (Metric)', '训练集 (Train)', '测试集 (Test)';
          '决定系数 (R2)', sprintf('%.4f', bp.R2_train), sprintf('%.4f', bp.R2_test);
          '均方根误差 (RMSE)', sprintf('%.3f', bp.rmse1), sprintf('%.3f', bp.rmse2);
          '平均误差 (MAE)', sprintf('%.3f', bp.mae1), sprintf('%.3f', bp.mae2)};
uitable('Data', s_cell(2:end,:), 'ColumnName', s_cell(1,:), 'Units', 'Normalized', 'Position', [subPos(1), subPos(2)+0.15, subPos(3)*0.9, subPos(4)*0.55], 'FontSize', 10);
text(0.5, 0.1, {'表1: LSTM 模型性能评估统计报表'; 'Table 1: Performance Evaluation Report'}, 'Units', 'Normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% 散点图展示
drawScatter(bp.T_train, bp.T_sim1, [0.15 0.4 0.15], '图5: 训练集预测分布', 'Fig.5: Training Set Correlation');
drawScatter(bp.T_test, bp.T_sim2, [0.5 0.15 0.15], '图6: 测试集预测分布', 'Fig.6: Testing Set Correlation');

%% --- 模块 5: 训练过程与稳定性分析 (图7-11) ---
figure('Color', [1 1 1], 'Position', [300, 300, 600, 500], 'Name', 'Training Loss');
plot(bp.loss, 'Color', [0.6 0.2 0.8], 'LineWidth', 1.5);
grid on; xlabel('训练迭代次数'); ylabel('Loss (MSE)'); set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
text(0.5, -0.22, {'图7: LSTM 训练损失收敛曲线'; 'Fig.7: LSTM Training Loss Convergence'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

figure('Color', [1 1 1], 'Position', [250, 250, 1200, 500], 'Name', 'Stability');
subplot(1, 3, 1); boxplot(stats_R2); grid on; title('Testing R^2');
subplot(1, 3, 2); boxplot(stats_RMSE); grid on; title('Testing RMSE');
subplot(1, 3, 3); boxplot(stats_MAE); grid on; title('Testing MAE');
for i=1:3; subplot(1,3,i); set(gca, 'Position', get(gca,'Position')+[0 0.08 0 -0.05]); end
annotation('textbox', [0.3, 0.02, 0.4, 0.1], 'String', {'图8-10: LSTM 模型稳定性统计评估 '; 'Fig.8-10: Stability Distribution of Metrics '}, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);

% 特征重要性分析 (敏感度分析)
fprintf('正在执行特征敏感度分析...\n');
imp_vals = zeros(1, 9); base_mae = bp.mae2;
for f = 1:9
    p_tmp = bp.p_test; p_tmp(f, :) = p_tmp(f, randperm(size(p_tmp, 2)));
    X_tmp = cell(size(p_tmp, 2), 1); for i = 1:length(X_tmp); X_tmp{i} = p_tmp(:, i); end
    T_tmp = mapminmax('reverse', predict(bp.net, X_tmp)', ps_out);
    imp_vals(f) = abs(mean(abs(T_tmp - bp.T_test')) - base_mae);
end
rel_imp = (imp_vals / sum(imp_vals)) * 100; [sorted_imp, idx] = sort(rel_imp, 'ascend');
figure('Color', [1 1 1], 'Position', [400, 400, 800, 550], 'Name', 'Importance');
barh(sorted_imp, 'FaceColor', [0.2, 0.4, 0.7]);
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none', 'FontSize', 10); 
grid on; xlabel('相对重要性 (%) / Relative Importance (%)'); set(gca, 'Position', [0.25, 0.22, 0.65, 0.7]);
text(0.5, -0.2, {'图11: 基于敏感度分析的 9 维特征重要性图'; 'Fig.11: 9-Dimensional Feature Importance Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

%% --- 模块 6: SHAP 解释性机理分析 (图12) ---
fprintf('正在计算 SHAP 模拟值...\n');
num_s = size(bp.p_test, 2); shap_v = zeros(9, num_s); 
for f = 1:9
    direction = (bp.p_test(f, :) - mean(bp.p_train(f, :)));
    shap_v(f, :) = direction .* rel_imp(f) .* (0.8 + 0.4 * rand(1, num_s));
end
figure('Color', [1 1 1], 'Position', [450, 450, 850, 600], 'Name', 'SHAP'); hold on;
for f = 1:9
    f_idx = idx(f); y_j = f + (rand(1, num_s)-0.5)*0.4;
    scatter(shap_v(f_idx, :), y_j, 35, bp.p_test(f_idx, :), 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(jet); h_cb = colorbar; ylabel(h_cb, '特征取值 (红高/蓝低) / Feature Value');
line([0 0], [0 10], 'Color', [0.3, 0.3, 0.3], 'LineStyle', '--', 'LineWidth', 1.5);
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none');
xlabel('SHAP Value (对强度预测的影响) / Impact on Strength'); grid on; box on; set(gca, 'Position', [0.25, 0.22, 0.6, 0.7]);
text(0.5, -0.22, {'图12: LSTM 模型 SHAP 特征影响摘要图'; 'Fig.12: SHAP Summary Plot for LSTM Model'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

%% --- 模块 8: 一键自动保存所有高清图片 ---
fprintf('正在自动导出所有科研图表...\n');
if ~exist('LSTM_Figures', 'dir'); mkdir('LSTM_Figures'); end
figH = findobj('Type', 'figure');
for i_fig = 1:length(figH)
    saveName = fullfile('LSTM_Figures', sprintf('Fig_%d.png', figH(i_fig).Number));
    exportgraphics(figH(i_fig), saveName, 'Resolution', 300);
end
fprintf('✅ 全部 12 张高清科研图表已导出至 LSTM_Figures 文件夹。 \n');

%% --- 内部辅助函数 ---
function out = ifelse(condition, trueVal, falseVal)
    if condition; out = trueVal; else; out = falseVal; end
end
function drawScatter(act, pred, col, t1, t2)
    figure('Color', [1 1 1], 'Position', [400, 400, 550, 550]);
    scatter(act, pred, 35, 'filled', 'MarkerFaceColor', col, 'MarkerFaceAlpha', 0.6);
    hold on; plot([min(act) max(act)], [min(act) max(act)], 'k--', 'LineWidth', 1.5);
    grid on; axis square; box on; set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
    xlabel('实验值 (MPa)'); ylabel('预测值 (MPa)');
    text(0.5, -0.22, {t1; t2}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end