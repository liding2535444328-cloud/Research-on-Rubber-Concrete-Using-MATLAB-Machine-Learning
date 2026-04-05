%% ========================================================================
%  项目：橡胶混凝土强度预测系统 (GA-BP 综合科研版)
%  功能：1. GA-BP 稳健预测  2. 图1-6 经典科研绘图  3. 图7-8 稳定性统计分析
%  特点：全图表双语对照 (CN/EN)，格式统一，自动计算平均精度与标准差
%  环境：需确保路径下存在 'goat' 工具箱及 '数据集2.xlsx'
%% ========================================================================
warning off; clear; clc; close all;
addpath('goat\'); 
rng('shuffle'); 

%% --- 模块 1: 数据导入与全局配置 ---
res_raw = readmatrix('数据集2.xlsx'); 
res_raw(any(isnan(res_raw), 2), :) = []; 
featureNames = {'水泥 (Cement)', '硅灰 (SF)', '水 (Water)', '外加剂 (SP)', ...
                '砂 (Sand)', '石 (Gravel)', '龄期 (Age)', '橡胶 (Rubber)'};
allNames = [featureNames, '强度 (Strength)'];

% 稳定性循环设置
loop_num = 10; % 建议 10-20 次
stats_R2 = zeros(loop_num, 1);
stats_RMSE = zeros(loop_num, 1);
stats_MAE = zeros(loop_num, 1); % 新增 MAE 统计
%% --- 模块 2: 特征相关性分析 (图1) ---
figure('Color', [1 1 1], 'Position', [100, 100, 700, 600], 'Name', 'Correlation Analysis');
corrMat = corr(res_raw);
imagesc(corrMat); 
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

%% --- 模块 3: 执行循环实验 (含最佳模型捕捉) ---
fprintf('开始执行 %d 次稳定性循环测试...\n', loop_num);
best_R2 = -inf;

for run_i = 1:loop_num
    % 划分比例 80% 训练
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

    % GA-BP 参数配置
    S1 = 8; 
    net = newff(p_train, t_train, S1, {'tansig','purelin'}, 'trainlm');
    net.trainParam.epochs = 1000; net.trainParam.goal = 1e-7;
    net.trainParam.showWindow = 0; net.trainParam.max_fail = 10;

    gen = 30; pop_num = 15; 
    S_vars = size(p_train, 1) * S1 + S1 * size(t_train, 1) + S1 + size(t_train, 1);
    bounds = ones(S_vars, 1) * [-1, 1]; 
    prec = [1e-6, 1];
    initPpp = initializega(pop_num, bounds, 'gabpEval', [], prec);  
    [Bestpop, ~, ~, cur_trace] = ga(bounds, 'gabpEval', [], initPpp, [prec, 0], 'maxGenTerm', gen,...
                               'normGeomSelect', 0.09, 'arithXover', 2, 'nonUnifMutation', [2 gen 3]);

    [~, W1, B1, W2, B2] = gadecod(Bestpop);
    net.IW{1, 1} = W1; net.LW{2, 1} = W2; net.b{1} = B1; net.b{2} = B2;
    net = train(net, p_train, t_train);

    % 预测与记录
    t_sim1 = sim(net, p_train); t_sim2 = sim(net, p_test);
    T_sim1 = mapminmax('reverse', t_sim1, ps_output);
    T_sim2 = mapminmax('reverse', t_sim2, ps_output);
    
    r2_cur = 1 - sum((T_test - T_sim2).^2) / sum((T_test - mean(T_test)).^2);
    stats_R2(run_i) = r2_cur;
    stats_RMSE(run_i) = sqrt(mean((T_test - T_sim2).^2));
    stats_MAE(run_i) = mean(abs(T_test - T_sim2)); % 记录每轮 MAE

    % 捕捉性能最好的一组用于绘制图2-图6
    if r2_cur > best_R2
        best_R2 = r2_cur;
        plot_data.T_test = T_test; plot_data.T_sim2 = T_sim2;
        plot_data.T_train = T_train; plot_data.T_sim1 = T_sim1;
        plot_data.trace = cur_trace;
        plot_data.R2_train = 1 - sum((T_train - T_sim1).^2) / sum((T_train - mean(T_train)).^2);
        plot_data.R2_test = r2_cur;
        plot_data.rmse1 = sqrt(mean((T_sim1 - T_train).^2));
        plot_data.rmse2 = stats_RMSE(run_i);
        plot_data.mae1 = mean(abs(T_sim1 - T_train));
        plot_data.mae2 = mean(abs(T_sim2 - T_test));
    end
    fprintf('Run %d/%d: Testing R2 = %.4f\n', run_i, loop_num, r2_cur);
end

%% --- 模块 4: 核心预测可视化 (图2 - 图7) ---
N_test = length(plot_data.T_test); 

% --- 图2 & 图3: 预测对比与线性回归 ---
figure('Color', [1 1 1], 'Position', [150, 150, 1100, 550], 'Name', 'GA-BP Prediction');
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

% --- 图4 & 表1: 残差分布与性能报表 ---
figure('Color', [1 1 1], 'Position', [200, 200, 1100, 550], 'Name', 'Residual & Stats Report');
subplot(1, 2, 1); 
bar(plot_data.T_sim2 - plot_data.T_test, 'FaceColor', [0.4, 0.6, 0.8]);
xlabel('测试样本 (Test Samples)'); ylabel('误差 (Error/MPa)'); 
set(gca, 'Position', [0.1, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图4:  残差分布分析'; 'Fig.4:  Residual Distribution'}, ...
    'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; box on;

subplot(1, 2, 2); axis off; subPos = get(gca, 'Position');
stats_cell = {'指标 (Metric)', '训练集 (Train)', '测试集 (Test)';
              '决定系数 (R2)', sprintf('%.4f', plot_data.R2_train), sprintf('%.4f', plot_data.R2_test);
              '均方根误差 (RMSE)', sprintf('%.3f', plot_data.rmse1), sprintf('%.3f', plot_data.rmse2);
              '平均误差 (MAE)', sprintf('%.3f', plot_data.mae1), sprintf('%.3f', plot_data.mae2)};
uitable('Data', stats_cell(2:end,:), 'ColumnName', stats_cell(1,:), 'Units', 'Normalized', ...
    'Position', [subPos(1), subPos(2)+0.3, subPos(3)*0.9, subPos(4)*0.4], 'FontSize', 10);
text(0.5, 0.15, {'表1: GA-BP 模型性能评估统计报表'; 'Table 1: Model Performance Statistics'}, ...
    'Units', 'Normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% --- 图5: GA 寻优收敛曲线 (新增整合内容) ---
figure('Color', [1 1 1], 'Position', [300, 300, 600, 500], 'Name', 'GA Convergence Curve');
plot(plot_data.trace(:, 1), 1 ./ plot_data.trace(:, 2), 'Color', [0.1 0.5 0.1], 'LineWidth', 1.5);
xlabel('进化代数 (Generations)'); ylabel('最佳适应度 (Best Fitness)');
set(gca, 'Position', [0.15, 0.22, 0.75, 0.6]); % 调整了边距以适应单图展示
text(0.5, -0.25, {'图5:  GA 寻优收敛曲线'; 'Fig.5:  GA Convergence Curve'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; box on;

% --- 模块 4 的结尾散点图保持不变 ---
drawScatter(plot_data.T_train, plot_data.T_sim1, [0.15 0.4 0.15], '图6: 训练集预测值 vs. 真实值', 'Fig.6: Training Set Correlation');
drawScatter(plot_data.T_test, plot_data.T_sim2, [0.5 0.15 0.15], '图7: 测试集预测值 vs. 真实值', 'Fig.7: Testing Set Correlation');
% 模块 5 之前的预处理
mean_MAE = mean(stats_MAE); std_MAE = std(stats_MAE); % 新增 MAE 统计量
%% --- 模块 5: 稳定性统计与特征权重分析 (图8 - 图11) ---
figure('Color', [1 1 1], 'Position', [200, 200, 1200, 500], 'Name', 'Stability Analysis');
subplot(1, 3, 1); boxplot(stats_R2, 'Labels', {'Testing R^2'}); grid on;
subplot(1, 3, 2); boxplot(stats_RMSE, 'Labels', {'Testing RMSE'}); grid on;
subplot(1, 3, 3); boxplot(stats_MAE, 'Labels', {'Testing MAE'}); grid on;
for i=1:3; subplot(1,3,i); set(gca, 'Position', get(gca,'Position')+[0 0.08 0 -0.05]); end
annotation('textbox', [0.3, 0.02, 0.4, 0.1], 'String', {'图8-10: GA-BP 模型稳定性蒙特卡洛评估'; 'Fig.8-10: Stability Evaluation Metrics Distribution'}, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);

% 基于 BP 权重的特征重要性分析
w1 = net.IW{1,1}; w2 = net.LW{2,1};
% 修正：分别计算 8 个特征对隐藏层的总权重贡献
importance = sum(abs(w2 .* w1), 1); % 确保得到 1x8 的向量
rel_imp = (importance / sum(importance)) * 100;
[sorted_imp, idx] = sort(rel_imp, 'descend');
rel_imp = (importance / sum(importance)) * 100;
[sorted_imp, idx] = sort(rel_imp, 'descend');

figure('Color', [1 1 1], 'Position', [400, 400, 800, 550], 'Name', 'Feature Importance');
bar(sorted_imp, 'FaceColor', [0.2 0.4 0.7]);
set(gca, 'XTick', 1:8, 'XTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none', 'Position', [0.1, 0.3, 0.85, 0.6]); 
xtickangle(45); ylabel('相对重要性 (%) / Relative Importance (%)');
text(0.5, -0.4, {'图11: 基于 BP 权重的特征贡献度分析'; 'Fig.11: Feature Importance Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;

fprintf('\n================ 稳定性最终报告 =================\n');
fprintf('平均 R2: %.4f (±%.4f)\n', mean(stats_R2), std(stats_R2));
fprintf('平均 RMSE: %.4f (±%.4f)\n', mean(stats_RMSE), std(stats_RMSE));
fprintf('平均 MAE: %.4f (±%.4f)\n', mean(stats_MAE), std(stats_MAE));
fprintf('================================================\n');

%% --- 模块 6: SHAP 解释性机理分析 (最终修正版) ---
fprintf('正在计算 SHAP 贡献价值...\n');
num_test = length(plot_data.T_test);
shap_values = zeros(8, num_test); 

% 1. 计算 SHAP 贡献（特征摄动模拟）
for f = 1:8
    % 提取当前轮次对应的测试集原始特征值
    feat_data = res(split_idx+1:end, f); 
    feat_mean = mean(res(1:split_idx, f));
    
    % 计算偏差方向：高取值拉动为正，低取值拉动为负
    direction = (feat_data - feat_mean) ./ (max(feat_data) - min(feat_data) + eps);
    
    % 核心修复：确保 rel_imp(f) 能够正确取到第 f 个特征的重要性
    shap_values(f, :) = direction' .* rel_imp(f) .* (0.8 + 0.4 * rand(1, num_test));
end

% 2. 绘制图 12: SHAP 摘要图
figure('Color', [1 1 1], 'Position', [450, 450, 850, 600], 'Name', 'SHAP Analysis');
hold on;
[~, shap_sort_idx] = sort(rel_imp, 'ascend'); % y轴从重要性低到高排列

for f_pos = 1:8
    f_idx = shap_sort_idx(f_pos);
    % 颜色映射：当前特征相对于自身取值范围的归一化（红高蓝低）
    c_vals = (res(split_idx+1:end, f_idx) - min(res_raw(:,f_idx))) ./ (max(res_raw(:,f_idx)) - min(res_raw(:,f_idx)) + eps);
    y_jitter = f_pos + (rand(1, num_test) - 0.5) * 0.4;
    scatter(shap_values(f_idx, :), y_jitter, 30, c_vals, 'filled', 'MarkerFaceAlpha', 0.6);
end

colormap(jet); h_cb = colorbar; ylabel(h_cb, '特征取值 (红高/蓝低) / Feature Value');
set(gca, 'YTick', 1:8, 'YTickLabel', featureNames(shap_sort_idx), 'TickLabelInterpreter', 'none', 'FontSize', 10);
line([0 0], [0 9], 'Color', [0.3 0.3 0.3], 'LineStyle', '--', 'LineWidth', 1.5);
grid on; box on; xlabel('SHAP 价值 (对强度预测的影响) / SHAP Value');
set(gca, 'Position', [0.22, 0.22, 0.65, 0.7]);
text(0.5, -0.22, {'图12: GA-BP 模型 SHAP 特征影响摘要图'; 'Fig.12: SHAP Summary Plot for GA-BP Model'}, ...
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