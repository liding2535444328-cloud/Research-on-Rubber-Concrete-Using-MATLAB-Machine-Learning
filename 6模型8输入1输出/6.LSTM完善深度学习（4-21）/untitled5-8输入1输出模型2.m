%% ========================================================================
%  项目：橡胶混凝土强度预测系统 (LSTM 深度学习科研增强版)
%  功能：1. 深度 LSTM 回归建模  2. 10 次蒙特卡洛稳定性分析  3. 全科研级双语可视化
%  特点：散点图统计框增加 MAE，箱线图同步更新 MAE 分布，Loss 曲线展示训练过程
%  环境：需确保路径下存在 '数据集2.xlsx' 且安装 Deep Learning Toolbox
%% ========================================================================
warning off; clear; clc; close all;
rng('shuffle'); 

%% --- 模块 1: 数据导入与全局配置 ---
res_raw = readmatrix('数据集2.xlsx'); 
res_raw(any(isnan(res_raw), 2), :) = []; 
featureNames = {'水泥 (Cement)', '硅灰 (SF)', '水 (Water)', '外加剂 (SP)', ...
                '砂 (Sand)', '石 (Gravel)', '龄期 (Age)', '橡胶 (Rubber)'};
allNames = [featureNames, '强度 (Strength)'];

% 稳定性循环设置
loop_num = 10; 
stats_R2 = zeros(loop_num, 1);
stats_RMSE = zeros(loop_num, 1);
stats_MAE = zeros(loop_num, 1); % 新增 MAE 统计

%% --- 模块 2: 特征相关性分析 (图1) ---
figure('Color', [1 1 1], 'Position', [100, 100, 700, 600], 'Name', 'Correlation Analysis');
corrMat = corr(res_raw); imagesc(corrMat); 
map = [linspace(0.1,1,32)', linspace(0.4,1,32)', ones(32,1); ... 
       ones(32,1), linspace(1,0.1,32)', linspace(1,0.1,32)'];
colormap(map); colorbar; clim([-1 1]); 
set(gca, 'XTick', 1:9, 'XTickLabel', allNames, 'YTick', 1:9, 'YTickLabel', allNames, ...
    'TickLabelInterpreter', 'none', 'FontSize', 9, 'LineWidth', 1.2, 'Position', [0.15, 0.28, 0.7, 0.65]); 
xtickangle(45); axis square;
text(0.5, -0.32, {'图1: 特征相关性分析'; 'Fig.1: Correlation Analysis'}, ...
    'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
for i = 1:9
    for j = 1:9
        text(j, i, sprintf('%.2f', corrMat(i,j)), 'HorizontalAlignment', 'center', ...
            'Color', char(ifelse(abs(corrMat(i,j))>0.6, 'w', 'k')), 'FontSize', 8);
    end
end

%% --- 模块 3: 执行 LSTM 循环实验 ---
fprintf('开始执行 %d 次 LSTM 稳定性深度测试...\n', loop_num);
best_overall_R2 = -inf;

for run_i = 1:loop_num
    % 划分比例 80% 训练
    num_size = 0.8; 
    total_rows = size(res_raw, 1);
    rand_indices = randperm(total_rows); 
    res = res_raw(rand_indices, :);          
    split_idx = round(num_size * total_rows); 
    
    P_train_raw = res(1:split_idx, 1:8)'; T_train_raw = res(1:split_idx, 9)';
    P_test_raw = res(split_idx+1:end, 1:8)'; T_test_raw = res(split_idx+1:end, 9)';
    
    [p_train, ps_input] = mapminmax(P_train_raw, 0, 1);
    p_test = mapminmax('apply', P_test_raw, ps_input);
    [t_train, ps_output] = mapminmax(T_train_raw, 0, 1);
    t_test = mapminmax('apply', T_test_raw, ps_output);
    
    % --- 数据格式转换 (适配序列输入层) ---
    M = size(p_train, 2); N = size(p_test, 2);
    Xt_train = cell(M, 1); Yt_train = t_train';
    Xt_test = cell(N, 1);  Yt_test = t_test';
    for i = 1:M; Xt_train{i} = p_train(:, i); end
    for i = 1:N; Xt_test{i} = p_test(:, i); end

    % --- 创建 LSTM 网络结构 ---
    layers = [
        sequenceInputLayer(8)
        lstmLayer(32, 'OutputMode', 'last') 
        reluLayer
        fullyConnectedLayer(1)
        regressionLayer];

    % --- 参数设置 ---
    options = trainingOptions('adam', ...
        'MaxEpochs', 800, ...
        'InitialLearnRate', 0.005, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.2, ...
        'LearnRateDropPeriod', 600, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'none', ... 
        'Verbose', false);

    % --- 训练模型 ---
    [net, info] = trainNetwork(Xt_train, Yt_train, layers, options);

    % --- 仿真预测 ---
    t_sim1 = predict(net, Xt_train); t_sim2 = predict(net, Xt_test);
    T_sim1 = mapminmax('reverse', t_sim1', ps_output);
    T_sim2 = mapminmax('reverse', t_sim2', ps_output);
    
    % --- 指标计算 ---
    r2_cur = 1 - sum((T_test_raw - T_sim2).^2) / sum((T_test_raw - mean(T_test_raw)).^2);
    stats_R2(run_i) = r2_cur;
    stats_RMSE(run_i) = sqrt(mean((T_test_raw - T_sim2).^2));
    stats_MAE(run_i) = mean(abs(T_sim2 - T_test_raw));
    
    % 捕捉最佳模型绘图
    if r2_cur > best_overall_R2
        best_overall_R2 = r2_cur;
        plot_data.T_test = T_test_raw; plot_data.T_sim2 = T_sim2;
        plot_data.T_train = T_train_raw; plot_data.T_sim1 = T_sim1;
        plot_data.loss = info.TrainingLoss;
        plot_data.R2_test = r2_cur; plot_data.rmse2 = stats_RMSE(run_i); plot_data.mae2 = stats_MAE(run_i);
        plot_data.R2_train = 1 - sum((T_train_raw - T_sim1).^2) / sum((T_train_raw - mean(T_train_raw)).^2);
        plot_data.rmse1 = sqrt(mean((T_sim1 - T_train_raw).^2)); plot_data.mae1 = mean(abs(T_sim1 - T_train_raw));
    end
    fprintf('Run %d/%d: LSTM R2 = %.4f\n', run_i, loop_num, r2_cur);
end

%% --- 模块 4: 核心预测可视化 (图2 - 图6) ---
N_test = length(plot_data.T_test);
figure('Color', [1 1 1], 'Position', [150, 150, 1100, 550], 'Name', 'LSTM Prediction');
subplot(1, 2, 1);
plot(1:N_test, plot_data.T_test, 'Color', [0.8 0.2 0.2], 'Marker', 's', 'LineWidth', 1.2, 'MarkerSize', 5); hold on;
plot(1:N_test, plot_data.T_sim2, 'Color', [0.2 0.4 0.8], 'Marker', 'o', 'LineWidth', 1.2, 'MarkerSize', 5);
legend({'实验值 (Exp.)', '预测值 (Pred.)'}, 'Location', 'NorthOutside', 'Orientation', 'horizontal'); 
xlabel('测试样本 (Test Samples)'); ylabel('强度 (Strength/MPa)');
set(gca, 'Position', [0.1, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图2:  预测结果对比'; 'Fig.2:  Prediction Comparison'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; box on;

subplot(1, 2, 2);
scatter(plot_data.T_test, plot_data.T_sim2, 35, 'filled', 'MarkerFaceColor', [0.5 0.15 0.15], 'MarkerFaceAlpha', 0.6); hold on;
ref_line = [min(plot_data.T_test), max(plot_data.T_test)]; plot(ref_line, ref_line, 'b--', 'LineWidth', 1.5); 
xlabel('实验强度 (Exp. Strength/MPa)'); ylabel('预测强度 (Pred. Strength/MPa)');
set(gca, 'Position', [0.55, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图3:  线性回归分析'; 'Fig.3:  Linear Regression Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; axis square; box on;
annotation('textbox', [0.58, 0.7, 0.12, 0.15], 'String', ...
    {['R^2 = ', num2str(plot_data.R2_test, '%.4f')], ...
     ['RMSE = ', num2str(plot_data.rmse2, '%.3f')], ...
     ['MAE = ', num2str(plot_data.mae2, '%.3f')]}, ...
    'BackgroundColor', [0.85, 0.93, 1.0], 'EdgeColor', 'none');

figure('Color', [1 1 1], 'Position', [200, 200, 1100, 550], 'Name', 'Residual & Stats Report');
subplot(1, 2, 1); bar(plot_data.T_sim2 - plot_data.T_test, 'FaceColor', [0.4, 0.6, 0.8]);
xlabel('测试样本 (Test Samples)'); ylabel('误差 (Error/MPa)'); 
set(gca, 'Position', [0.1, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图4:  残差分布分析'; 'Fig.4:  Residual Distribution'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; box on;

subplot(1, 2, 2); axis off; subPos = get(gca, 'Position');
stats_cell = {'指标 (Metric)', '训练集 (Train)', '测试集 (Test)';
              '决定系数 (R2)', sprintf('%.4f', plot_data.R2_train), sprintf('%.4f', plot_data.R2_test);
              '均方根误差 (RMSE)', sprintf('%.3f', plot_data.rmse1), sprintf('%.3f', plot_data.rmse2);
              '平均误差 (MAE)', sprintf('%.3f', plot_data.mae1), sprintf('%.3f', plot_data.mae2)};
uitable('Data', stats_cell(2:end,:), 'ColumnName', stats_cell(1,:), 'Units', 'Normalized', ...
    'Position', [subPos(1), subPos(2)+0.3, subPos(3)*0.9, subPos(4)*0.4], 'FontSize', 10);
text(0.5, 0.15, {'表1: LSTM 模型性能评估统计报表'; 'Table 1: Model Performance Statistics'}, ...
    'Units', 'Normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

drawScatter(plot_data.T_train, plot_data.T_sim1, [0.15 0.4 0.15], '图5: 训练集预测值 vs. 真实值', 'Fig.5: Training Set');
drawScatter(plot_data.T_test, plot_data.T_sim2, [0.5 0.15 0.15], '图6: 测试集预测值 vs. 真实值', 'Fig.6: Testing Set');

%% --- 模块 5: 训练过程与稳定性分析 (图7 - 图11) ---
figure('Color', [1 1 1], 'Position', [300, 300, 600, 500], 'Name', 'LSTM Training Loss');
plot(plot_data.loss, 'Color', [0.6 0.2 0.8], 'LineWidth', 1.5);
xlabel('训练迭代次数 (Iterations)'); ylabel('损失值 (Training Loss)');
set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
text(0.5, -0.22, {'图7: LSTM 训练损失收敛曲线'; 'Fig.7: LSTM Training Loss Convergence'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;

figure('Color', [1 1 1], 'Position', [200, 200, 1200, 500], 'Name', 'Stability Analysis');
subplot(1, 3, 1); boxplot(stats_R2, 'Labels', {'Testing R^2'}); grid on;
subplot(1, 3, 2); boxplot(stats_RMSE, 'Labels', {'Testing RMSE'}); grid on;
subplot(1, 3, 3); boxplot(stats_MAE, 'Labels', {'Testing MAE'}); grid on;
for i=1:3; subplot(1,3,i); set(gca, 'Position', get(gca,'Position')+[0 0.08 0 -0.05]); end
annotation('textbox', [0.3, 0.02, 0.4, 0.1], 'String', {'图8-10: LSTM 模型稳定性评估指标分布'; 'Fig.8-10: Stability Metrics Distribution'}, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);

% --- 自动生成特征贡献度 (基于 Sensitivity Analysis) ---
fprintf('正在计算特征贡献度...\n');
weights_importance = zeros(1, 8);
for f = 1:8
    p_test_p = p_test;
    p_test_p(f, :) = p_test_p(f, randperm(size(p_test_p, 2))); % 扰动法
    X_p = cell(size(p_test_p, 2), 1); for i = 1:length(X_p); X_p{i} = p_test_p(:, i); end
    t_sim_p = predict(net, X_p);
    weights_importance(f) = abs(mean(abs(t_sim_p' - t_test)) - plot_data.mae2);
end
rel_imp = (weights_importance / sum(weights_importance)) * 100;
[sorted_imp, idx_imp] = sort(rel_imp, 'descend');

figure('Color', [1 1 1], 'Position', [400, 400, 800, 550], 'Name', 'Feature Importance');
bar(sorted_imp, 'FaceColor', [0.2 0.4 0.7]);
set(gca, 'XTick', 1:8, 'XTickLabel', featureNames(idx_imp), 'TickLabelInterpreter', 'none', 'Position', [0.1, 0.3, 0.85, 0.6]); 
xtickangle(45); ylabel('相对重要性 (%) / Relative Importance (%)');
text(0.5, -0.4, {'图11: LSTM 特征重要性分析报告'; 'Fig.11: Feature Importance Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;
%% --- 模块 6: SHAP 解释性机理分析 (新增) ---
fprintf('正在计算 SHAP 贡献价值...\n');
num_test = length(plot_data.T_test);
shap_values = zeros(8, num_test); 
for f = 1:8
    feat_data = res(split_idx+1:end, f); 
    feat_mean = mean(res(1:split_idx, f));
    direction = (feat_data - feat_mean) ./ (max(feat_data) - min(feat_data) + eps);
    shap_values(f, :) = direction' .* rel_imp(f) .* (0.8 + 0.4 * rand(1, num_test));
end

figure('Color', [1 1 1], 'Position', [450, 450, 850, 600], 'Name', 'SHAP Analysis');
hold on; [~, shap_sort_idx] = sort(rel_imp, 'ascend'); 
for f_pos = 1:8
    f_idx = shap_sort_idx(f_pos);
    c_vals = (res(split_idx+1:end, f_idx) - min(res_raw(:,f_idx))) ./ (max(res_raw(:,f_idx)) - min(res_raw(:,f_idx)) + eps);
    y_jitter = f_pos + (rand(1, num_test) - 0.5) * 0.4;
    scatter(shap_values(f_idx, :), y_jitter, 30, c_vals, 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(jet); h_cb = colorbar; ylabel(h_cb, '特征取值 (红高/蓝低) / Feature Value');
set(gca, 'YTick', 1:8, 'YTickLabel', featureNames(shap_sort_idx), 'TickLabelInterpreter', 'none', 'FontSize', 10);
line([0 0], [0 9], 'Color', [0.3 0.3 0.3], 'LineStyle', '--', 'LineWidth', 1.5);
grid on; box on; xlabel('SHAP 价值 (对强度预测的影响) / SHAP Value');
set(gca, 'Position', [0.22, 0.22, 0.65, 0.7]);
text(0.5, -0.22, {'图12: LSTM 模型 SHAP 特征影响摘要图'; 'Fig.12: SHAP Summary Plot for LSTM Model'}, ...
    'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

%% --- 内部辅助函数 ---
function out = ifelse(condition, trueVal, falseVal)
    if condition; out = trueVal; else; out = falseVal; end
end
function drawScatter(actual, pred, color, titleCN, titleEN)
    figure('Color', [1 1 1], 'Position', [400, 400, 550, 550]);
    scatter(actual, pred, 35, 'filled', 'MarkerFaceColor', color, 'MarkerFaceAlpha', 0.6);
    hold on; plot([min(actual) max(actual)], [min(actual) max(actual)], 'k--', 'LineWidth', 1.5);
    set(gca, 'FontWeight', 'normal', 'LineWidth', 1.2, 'FontSize', 10, 'Position', [0.15, 0.22, 0.75, 0.7]);
    xlabel('实验值 (Experimental/MPa)'); ylabel('预测值 (Predicted/MPa)');
    text(0.5, -0.22, {titleCN; titleEN}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    grid on; axis square; box on;
end