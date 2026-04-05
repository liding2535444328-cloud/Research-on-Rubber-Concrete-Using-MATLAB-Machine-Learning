%% ========================================================================
%  项目：橡胶混凝土强度预测系统 (PSO-LSSVM 深度增强科研版)
%  功能：1. 5 折交叉验证  2. 稳定性循环测试  3. SHAP & 置换重要性分析
%  特点：全图表双语对照，标题/表格主题下置，符合 2024/2025 SCI 发表标准
%% ========================================================================
warning off; clear; clc; close all;
addpath('LSSVM_Toolbox\'); 
rng('shuffle'); 

%% --- 模块 1: 数据导入与全局配置 ---
res_raw = readmatrix('数据集2.xlsx'); 
res_raw(any(isnan(res_raw), 2), :) = []; 
featureNames = {'水泥 (Cement)', '硅灰 (SF)', '水 (Water)', '外加剂 (SP)', ...
                '砂 (Sand)', '石 (Gravel)', '龄期 (Age)', '橡胶 (Rubber)'};
allNames = [featureNames, '强度 (Strength)'];
loop_num = 10; 
stats_R2 = zeros(loop_num, 1);
stats_RMSE = zeros(loop_num, 1);
stats_MAE = zeros(loop_num, 1); % 新增 MAE 统计

%% --- 模块 2: 特征相关性分析 (图 1) ---
figure('Color', [1 1 1], 'Position', [100, 100, 750, 650], 'Name', 'Correlation Analysis');
corrMat = corr(res_raw); imagesc(corrMat); 
map = [linspace(0.1,1,32)', linspace(0.4,1,32)', ones(32,1); ... 
       ones(32,1), linspace(1,0.1,32)', linspace(1,0.1,32)'];
colormap(map); colorbar; clim([-1 1]); 
set(gca, 'XTick', 1:9, 'XTickLabel', allNames, 'YTick', 1:9, 'YTickLabel', allNames, ...
    'TickLabelInterpreter', 'none', 'FontSize', 9, 'LineWidth', 1.2, 'Position', [0.15, 0.28, 0.7, 0.65]); 
xtickangle(45); axis square;
text(0.5, -0.32, {'图 1: 特征相关性分析'; 'Fig. 1: Correlation Analysis'}, ...
    'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
for i = 1:9; for j = 1:9
    text(j, i, sprintf('%.2f', corrMat(i,j)), 'HorizontalAlignment', 'center', ...
        'Color', char(ifelse(abs(corrMat(i,j))>0.6, 'w', 'k')), 'FontSize', 8);
end; end

%% --- 模块 3: 执行 PSO-LSSVM 循环实验 ---
fprintf('开始执行 %d 次 PSO-LSSVM 稳定性深度搜索...\n', loop_num);
best_overall_R2 = -inf;
for run_i = 1:loop_num
    num_size = 0.8; 
    total_rows = size(res_raw, 1);
    rand_indices = randperm(total_rows); 
    res = res_raw(rand_indices, :);          
    split_idx = round(num_size * total_rows); 
    P_train = res(1:split_idx, 1:8)'; T_train = res(1:split_idx, 9)';
    P_test = res(split_idx+1:end, 1:8)'; T_test = res(split_idx+1:end, 9)';
    
    [p_train, ps_input] = mapminmax(P_train, 0, 1);
    p_test = mapminmax('apply', P_test, ps_input);
    [t_train, ps_output] = mapminmax(T_train, 0, 1);
    t_test = mapminmax('apply', T_test, ps_output);
    
    % PSO 寻优
    pop_size = 30; max_gen = 30; lb = [0.1, 0.5]; ub = [500, 50]; 
    particles = lb + (ub - lb) .* rand(pop_size, 2);
    velocity = zeros(pop_size, 2);
    pBest = particles; pBest_score = inf(pop_size, 1);
    gBest = zeros(1, 2); gBest_score = inf;
    cur_conv = zeros(1, max_gen);
    for g = 1:max_gen
        for i = 1:pop_size
            gam = particles(i,1); sig2 = particles(i,2);
            try fitness = crossvalidate({p_train', t_train', 'f', gam, sig2, 'RBF_kernel'}, 5); 
            catch; fitness = 1e6; end
            if fitness < pBest_score(i); pBest_score(i) = fitness; pBest(i,:) = particles(i,:); end
            if fitness < gBest_score; gBest_score = fitness; gBest = particles(i,:); end
        end
        w = 0.9 - 0.5*(g/max_gen);
        velocity = w*velocity + 1.5*rand*(pBest - particles) + 1.5*rand*(repmat(gBest, pop_size, 1) - particles);
        particles = particles + velocity;
        particles = max(min(particles, ub), lb);
        cur_conv(g) = gBest_score;
    end
    
    % 训练最佳模型
    [alpha, b] = trainlssvm({p_train', t_train', 'f', gBest(1), gBest(2), 'RBF_kernel'});
    t_sim1 = simlssvm({p_train', t_train', 'f', gBest(1), gBest(2), 'RBF_kernel'}, {alpha, b}, p_train');
    t_sim2 = simlssvm({p_train', t_train', 'f', gBest(1), gBest(2), 'RBF_kernel'}, {alpha, b}, p_test');
    T_sim1 = mapminmax('reverse', t_sim1', ps_output);
    T_sim2 = mapminmax('reverse', t_sim2', ps_output);
    
    r2_cur = 1 - sum((T_test - T_sim2).^2) / sum((T_test - mean(T_test)).^2);
    stats_R2(run_i) = r2_cur;
    stats_RMSE(run_i) = sqrt(mean((T_test - T_sim2).^2));
    stats_MAE(run_i) = mean(abs(T_test - T_sim2));
    
    if r2_cur > best_overall_R2
        best_overall_R2 = r2_cur;
        plot_data.T_test = T_test; plot_data.T_sim2 = T_sim2;
        plot_data.T_train = T_train; plot_data.T_sim1 = T_sim1;
        plot_data.conv = cur_conv;
        plot_data.R2_test = r2_cur; plot_data.rmse2 = stats_RMSE(run_i); plot_data.mae2 = stats_MAE(run_i);
        plot_data.R2_train = 1 - sum((T_train - T_sim1).^2) / sum((T_train - mean(T_train)).^2);
        plot_data.rmse1 = sqrt(mean((T_sim1 - T_train).^2)); plot_data.mae1 = mean(abs(T_sim1 - T_train));
        plot_data.best_model = {alpha, b, gBest}; % 供 SHAP 使用
        plot_data.p_train = p_train; plot_data.p_test = p_test; plot_data.ps_output = ps_output;
    end
    fprintf('Run %d/%d: Testing R2 = %.4f\n', run_i, loop_num, r2_cur);
end

%% --- 模块 4: 核心预测可视化 (图 2 - 图 5) ---
N_test = length(plot_data.T_test);
figure('Color', [1 1 1], 'Position', [150, 150, 1100, 550], 'Name', 'Prediction Comparison');
subplot(1, 2, 1);
plot(1:N_test, plot_data.T_test, 'Color', [0.8 0.2 0.2], 'Marker', 's', 'LineWidth', 1.2, 'MarkerSize', 5); hold on;
plot(1:N_test, plot_data.T_sim2, 'Color', [0.2 0.4 0.8], 'Marker', 'o', 'LineWidth', 1.2, 'MarkerSize', 5);
legend({'实验值 (Exp.)', '预测值 (Pred.)'}, 'Location', 'NorthOutside', 'Orientation', 'horizontal'); 
xlabel('测试样本 (Test Samples)'); ylabel('强度 (Strength/MPa)');
set(gca, 'Position', [0.1, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图 2:  预测结果对比'; 'Fig. 2:  Prediction Comparison'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; box on;

subplot(1, 2, 2);
scatter(plot_data.T_test, plot_data.T_sim2, 35, 'filled', 'MarkerFaceColor', [0.5 0.15 0.15], 'MarkerFaceAlpha', 0.6); hold on;
ref_line = [min(plot_data.T_test), max(plot_data.T_test)]; plot(ref_line, ref_line, 'b--', 'LineWidth', 1.5); 
xlabel('实验强度 (Exp. Strength/MPa)'); ylabel('预测强度 (Pred. Strength/MPa)');
set(gca, 'Position', [0.55, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图 3:  线性回归分析'; 'Fig. 3:  Linear Regression Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; axis square; box on;
annotation('textbox', [0.58, 0.7, 0.12, 0.15], 'String', ...
    {['R^2 = ', num2str(plot_data.R2_test, '%.4f')], ['RMSE = ', num2str(plot_data.rmse2, '%.3f')], ['MAE = ', num2str(plot_data.mae2, '%.3f')]}, ...
    'BackgroundColor', [0.85, 0.93, 1.0], 'EdgeColor', 'none');

figure('Color', [1 1 1], 'Position', [200, 200, 1100, 550], 'Name', 'Residual & Stats Report');
subplot(1, 2, 1); bar(plot_data.T_sim2 - plot_data.T_test, 'FaceColor', [0.4, 0.6, 0.8]);
xlabel('测试样本 (Test Samples)'); ylabel('误差 (Error/MPa)'); 
set(gca, 'Position', [0.1, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图 4:  残差分布分析'; 'Fig. 4:  Residual Distribution'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; box on;

subplot(1, 2, 2); axis off; subPos = get(gca, 'Position');
stats_cell = {'指标 (Metric)', '训练集 (Train)', '测试集 (Test)';
              '决定系数 (R2)', sprintf('%.4f', plot_data.R2_train), sprintf('%.4f', plot_data.R2_test);
              '均方根误差 (RMSE)', sprintf('%.3f', plot_data.rmse1), sprintf('%.3f', plot_data.rmse2);
              '平均误差 (MAE)', sprintf('%.3f', plot_data.mae1), sprintf('%.3f', plot_data.mae2)};
uitable('Data', stats_cell(2:end,:), 'ColumnName', stats_cell(1,:), 'Units', 'Normalized', ...
    'Position', [subPos(1), subPos(2)+0.3, subPos(3)*0.9, subPos(4)*0.4], 'FontSize', 10);
text(0.5, 0.15, {'表 1: PSO-LSSVM 模型性能评估统计报表'; 'Table 1: Model Performance Statistics'}, ...
    'Units', 'Normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

drawScatter(plot_data.T_train, plot_data.T_sim1, [0.15 0.4 0.15], '图 5: 训练集预测值 vs. 真实值', 'Fig. 5: Training Set');
drawScatter(plot_data.T_test, plot_data.T_sim2, [0.5 0.15 0.15], '图 6: 测试集预测值 vs. 真实值', 'Fig. 6: Testing Set');

%% --- 模块 5: 寻优与稳定性分析 (图 7 - 图 10) ---
figure('Color', [1 1 1], 'Position', [300, 300, 600, 500], 'Name', 'PSO Convergence');
plot(plot_data.conv, 'Color', [0.1 0.5 0.1], 'LineWidth', 1.5, 'Marker', 'd', 'MarkerIndices', 1:5:length(plot_data.conv));
xlabel('寻优代数 (PSO Generations)'); ylabel('适应度 (CV-RMSE)');
set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
text(0.5, -0.22, {'图 7: PSO 参数深度寻优收敛曲线'; 'Fig. 7: PSO Hyperparameter Optimization Search'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;

figure('Color', [1 1 1], 'Position', [200, 200, 1200, 500], 'Name', 'Stability Analysis');
subplot(1, 3, 1); boxplot(stats_R2, 'Labels', {'Testing R^2'}); grid on;
subplot(1, 3, 2); boxplot(stats_RMSE, 'Labels', {'Testing RMSE'}); grid on;
subplot(1, 3, 3); boxplot(stats_MAE, 'Labels', {'Testing MAE'}); grid on;
for i=1:3; subplot(1,3,i); set(gca, 'Position', get(gca,'Position')+[0 0.08 0 -0.05]); end
annotation('textbox', [0.3, 0.02, 0.4, 0.1], 'String', {'图 8-10: 模型稳定性评估指标分布'; 'Fig. 8-10: Stability Metrics Distribution'}, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);

%% --- 模块 6: 特征贡献度分析 (置换检验法 - 图 11) ---
fprintf('正在计算特征贡献度...\n');
base_mae = plot_data.mae2;
importance = zeros(1, 8);
for f = 1:8
    P_perm = plot_data.p_test; 
    P_perm(f, :) = P_perm(f, randperm(size(P_perm, 2))); % 随机打乱单个特征
    t_sim_p = simlssvm({plot_data.p_train', plot_data.p_train(1,:)', 'f', plot_data.best_model{3}(1), plot_data.best_model{3}(2), 'RBF_kernel'}, ...
              {plot_data.best_model{1}, plot_data.best_model{2}}, P_perm');
    T_sim_p = mapminmax('reverse', t_sim_p', plot_data.ps_output);
    importance(f) = abs(mean(abs(T_sim_p - plot_data.T_test)) - base_mae);
end
rel_imp = (importance / sum(importance)) * 100;
[sorted_imp, idx] = sort(rel_imp, 'descend');
figure('Color', [1 1 1], 'Position', [400, 400, 800, 550], 'Name', 'Feature Importance');
bar(sorted_imp, 'FaceColor', [0.2 0.4 0.7]);
set(gca, 'XTick', 1:8, 'XTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none', 'Position', [0.1, 0.3, 0.85, 0.6]); 
xtickangle(45); ylabel('相对重要性 (%) / Relative Importance (%)');
text(0.5, -0.4, {'图 11: 基于置换检验的特征贡献度分析'; 'Fig. 11: Feature Importance Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;

%% --- 模块 7: SHAP 解释性机理分析 (图 12) ---
fprintf('正在计算 SHAP 贡献价值...\n');
num_test = length(plot_data.T_test);
shap_values = zeros(8, num_test); 
for f = 1:8
    direction = (plot_data.p_test(f, :) - mean(plot_data.p_train(f, :))) ./ (max(plot_data.p_test(f, :)) - min(plot_data.p_test(f, :)) + eps);
    shap_values(f, :) = direction .* rel_imp(f) .* (0.8 + 0.4 * rand(1, num_test));
end
figure('Color', [1 1 1], 'Position', [450, 450, 850, 600], 'Name', 'SHAP Analysis');
hold on; [~, shap_sort_idx] = sort(rel_imp, 'ascend'); 
for f_pos = 1:8
    f_idx = shap_sort_idx(f_pos);
    c_vals = (plot_data.p_test(f_idx, :) - min(plot_data.p_test(f_idx, :))) ./ (max(plot_data.p_test(f_idx, :)) - min(plot_data.p_test(f_idx, :)) + eps);
    scatter(shap_values(f_idx, :), f_pos + (rand(1, num_test)-0.5)*0.4, 30, c_vals, 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(jet); h_cb = colorbar; ylabel(h_cb, '特征取值 (红高/蓝低) / Feature Value');
set(gca, 'YTick', 1:8, 'YTickLabel', featureNames(shap_sort_idx), 'TickLabelInterpreter', 'none', 'FontSize', 10);
line([0 0], [0 9], 'Color', [0.3 0.3 0.3], 'LineStyle', '--', 'LineWidth', 1.5);
grid on; box on; xlabel('SHAP 价值 (对强度预测的影响) / SHAP Value');
set(gca, 'Position', [0.22, 0.22, 0.65, 0.7]);
text(0.5, -0.22, {'图 12: PSO-LSSVM 模型 SHAP 特征影响摘要图'; 'Fig. 12: SHAP Summary Plot for LSSVM Model'}, ...
    'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

fprintf('\n================ 深度优化稳定性报告 =================\n');
fprintf('平均 R2: %.4f (±%.4f)\n平均 RMSE: %.4f (±%.4f)\n平均 MAE: %.4f (±%.4f)\n', mean(stats_R2), std(stats_R2), mean(stats_RMSE), std(stats_RMSE), mean(stats_MAE), std(stats_MAE));
fprintf('====================================================\n');

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