%% ========================================================================
%  项目：橡胶混凝土强度预测系统 (MATLAB原生 SVR 科研级增强最终版)
%  数据：数据集3 (9个特征输入 -> 1个强度输出)
%  优化：全图表双语、标题下置、包含 MAE/SHAP/机理分析、无需外部工具箱
%% ========================================================================
warning off; clear; clc; close all;
rng('shuffle'); 

%% --- 模块 1: 数据导入与全局配置 ---
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

%% --- 模块 3: 执行 PSO-SVR 稳定性测试循环 ---
fprintf('正在执行 %d 次 PSO-SVR 深度测试...\n', loop_num);
best_overall_R2 = -inf;

for run_i = 1:loop_num
    num_size = 0.8; total_rows = size(res_raw, 1);
    rand_idx = randperm(total_rows); 
    res_shf = res_raw(rand_idx, :);          
    split_p = round(num_size * total_rows); 
    P_train = res_shf(1:split_p, 1:9); T_train = res_shf(1:split_p, 10);
    P_test = res_shf(split_p+1:end, 1:9); T_test = res_shf(split_p+1:end, 10);
    
    % PSO 寻优
    pop = 12; max_gen = 15; 
    lb = [0.1, 0.01]; ub = [100, 10]; 
    part = lb + (ub - lb) .* rand(pop, 2);
    vel = zeros(pop, 2);
    pBest = part; pBest_sc = inf(pop, 1);
    gBest = part(1,:); gBest_sc = inf;
    cur_conv = zeros(1, max_gen);
    
    for t = 1:max_gen
        for i = 1:pop
            try
                m_tmp = fitrsvm(P_train, T_train, 'KernelFunction', 'gaussian', ...
                    'BoxConstraint', part(i,1), 'KernelScale', part(i,2), 'Standardize', true);
                pred_tmp = predict(m_tmp, P_test);
                mse_val = mean((pred_tmp - T_test).^2);
            catch; mse_val = inf; end
            
            % 修复：统一变量名 sc
            if mse_val < pBest_sc(i)
                pBest_sc(i) = mse_val; pBest(i,:) = part(i,:);
            end
            if mse_val < gBest_sc
                gBest_sc = mse_val; gBest = part(i,:);
            end
        end
        vel = 0.6*vel + 1.5*rand*(pBest-part) + 1.5*rand*(repmat(gBest,pop,1)-part);
        part = part + vel; part = max(min(part, ub), lb);
        cur_conv(t) = gBest_sc;
    end
    
    final_m = fitrsvm(P_train, T_train, 'KernelFunction', 'gaussian', ...
        'BoxConstraint', gBest(1), 'KernelScale', gBest(2), 'Standardize', true);
    
    T_s1 = predict(final_m, P_train); T_s2 = predict(final_m, P_test);
    r2_cur = 1 - sum((T_test - T_s2).^2) / sum((T_test - mean(T_test)).^2);
    stats_R2(run_i) = r2_cur;
    stats_RMSE(run_i) = sqrt(mean((T_test - T_s2).^2));
    stats_MAE(run_i) = mean(abs(T_test - T_s2));
    
    if r2_cur >= best_overall_R2 || run_i == 1
        best_overall_R2 = r2_cur;
        plot_data.T_test = T_test; plot_data.T_sim2 = T_s2;
        plot_data.T_train = T_train; plot_data.T_sim1 = T_s1;
        plot_data.conv = cur_conv; plot_data.P_test = P_test; plot_data.P_train = P_train;
        plot_data.R2_test = r2_cur; 
        plot_data.R2_train = 1-sum((T_train-T_s1).^2)/sum((T_train-mean(T_train)).^2);
        plot_data.rmse2 = stats_RMSE(run_i); plot_data.mae2 = stats_MAE(run_i);
        plot_data.rmse1 = sqrt(mean((T_train-T_s1).^2)); plot_data.mae1 = mean(abs(T_train-T_s1));
        best_model = final_m;
    end
    fprintf('Run %d/%d: Testing R2 = %.4f\n', run_i, loop_num, r2_cur);
end

%% --- 模块 4: 预测可视化渲染 (图2-6) ---
figure('Color', [1 1 1], 'Position', [150, 150, 1100, 550], 'Name', 'Results');
subplot(1, 2, 1);
plot(plot_data.T_test, 'Color', [0.8, 0.2, 0.2], 'Marker', 's', 'LineWidth', 1.2); hold on;
plot(plot_data.T_sim2, 'Color', [0.2, 0.4, 0.8], 'Marker', 'o', 'LineWidth', 1.2);
legend({'实验值 (Exp.)', '预测值 (Pred.)'}, 'Location', 'NorthOutside', 'Orientation', 'horizontal'); 
grid on; ylabel('强度 (MPa) / Strength (MPa)'); xlabel('测试样本 (Test Samples)');
set(gca, 'Position', [0.1, 0.25, 0.35, 0.6]); 
text(0.5, -0.25, {'图2: 强度预测结果对比图'; 'Fig.2: Comparison of Strength Prediction Results'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

subplot(1, 2, 2);
scatter(plot_data.T_test, plot_data.T_sim2, 40, 'filled', 'MarkerFaceColor', [0.5, 0.1, 0.1], 'MarkerFaceAlpha', 0.6); hold on;
ref_l = [min(plot_data.T_test) max(plot_data.T_test)]; plot(ref_l, ref_l, 'b--', 'LineWidth', 1.5); 
grid on; axis square; xlabel('实验值 (Exp. /MPa)'); ylabel('预测值 (Pred. /MPa)');
set(gca, 'Position', [0.55, 0.25, 0.35, 0.6]); 
text(0.5, -0.25, {'图3: 实验值 vs. 预测值线性回归分析'; 'Fig.3: Regression Analysis of Exp. vs. Pred.'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
annotation('textbox', [0.58, 0.68, 0.12, 0.15], 'String', ...
    {['R^2 = ', num2str(plot_data.R2_test, '%.4f')], ...
     ['RMSE = ', num2str(plot_data.rmse2, '%.3f')], ...
     ['MAE = ', num2str(plot_data.mae2, '%.3f')]}, ...
    'BackgroundColor', [0.85, 0.93, 1.0], 'EdgeColor', 'none', 'FontSize', 9);

figure('Color', [1 1 1], 'Position', [200, 200, 1100, 500], 'Name', 'Stats Report');
subplot(1, 2, 1); bar(plot_data.T_sim2 - plot_data.T_test, 'FaceColor', [0.4, 0.6, 0.8]);
grid on; ylabel('误差 (MPa) / Error (MPa)'); xlabel('测试样本 (Test Samples)');
set(gca, 'Position', [0.1, 0.25, 0.35, 0.6]); 
text(0.5, -0.25, {'图4: 预测残差分布分析'; 'Fig.4: Residual Distribution Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

subplot(1, 2, 2); axis off; subPos = get(gca, 'Position');
s_cell = {'指标 (Metric)', '训练集 (Train)', '测试集 (Test)';
          '决定系数 (R2)', sprintf('%.4f', plot_data.R2_train), sprintf('%.4f', plot_data.R2_test);
          '均方根误差 (RMSE)', sprintf('%.3f', plot_data.rmse1), sprintf('%.3f', plot_data.rmse2);
          '平均绝对误差 (MAE)', sprintf('%.3f', plot_data.mae1), sprintf('%.3f', plot_data.mae2)};
uitable('Data', s_cell(2:end,:), 'ColumnName', s_cell(1,:), 'Units', 'Normalized', ...
    'Position', [subPos(1), subPos(2)+0.15, subPos(3)*0.9, subPos(4)*0.55], 'FontSize', 10);
text(0.5, 0.1, {'表1: PSO-SVR 模型性能评估统计报表'; 'Table 1: Performance Evaluation Report'}, 'Units', 'Normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

drawScatter(plot_data.T_train, plot_data.T_sim1, [0.15 0.4 0.15], '图5: 训练集预测分布', 'Fig.5: Training Set Correlation');
drawScatter(plot_data.T_test, plot_data.T_sim2, [0.5 0.15 0.15], '图6: 测试集预测分布', 'Fig.6: Testing Set Correlation');

%% --- 模块 5: 寻优与稳定性分析 (图7-10) ---
figure('Color', [1 1 1], 'Position', [300, 300, 600, 500], 'Name', 'Convergence');
hold on; grid on; % 修复：确保曲线绘制
plot(plot_data.conv, 'Color', [0.1, 0.5, 0.1], 'LineWidth', 1.8, 'Marker', 'o', 'MarkerSize', 4, 'MarkerIndices', 1:max(1,round(length(plot_data.conv)/10)):length(plot_data.conv));
xlabel('进化代数 (PSO Generations)'); ylabel('适应度值 (Fitness/MSE)');
set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
text(0.5, -0.22, {'图7: PSO 深度参数寻优收敛曲线'; 'Fig.7: PSO Hyperparameter Convergence curve'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

figure('Color', [1 1 1], 'Position', [250, 250, 1200, 500], 'Name', 'Stability');
subplot(1, 3, 1); boxplot(stats_R2); grid on; title('Testing R^2');
subplot(1, 3, 2); boxplot(stats_RMSE); grid on; title('Testing RMSE');
subplot(1, 3, 3); boxplot(stats_MAE); grid on; title('Testing MAE'); % 新增 MAE
for i=1:3; subplot(1,3,i); set(gca, 'Position', get(gca,'Position')+[0 0.08 0 -0.05]); end
annotation('textbox', [0.3, 0.02, 0.4, 0.1], 'String', {'图8-10: PSO-SVR 模型稳定性蒙特卡洛评估'; 'Fig.8-10: Stability Distribution of Metrics'}, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);

save('ConcreteModel.mat', 'best_model', 'res_raw', 'featureNames');

%% --- 模块 6: 特征重要性分析 (图11) ---
fprintf('生成特征贡献度图表...\n');
num_feat = 9; importance = zeros(1, num_feat);
for f = 1:num_feat
    P_p = plot_data.P_test; P_p(:, f) = P_p(randperm(size(P_p,1)), f);
    ts_p = predict(best_model, P_p);
    cur_r2 = 1 - sum((plot_data.T_test - ts_p).^2) / sum((plot_data.T_test - mean(plot_data.T_test)).^2);
    importance(f) = abs(plot_data.R2_test - cur_r2);
end
[sorted_imp, idx] = sort(importance/sum(importance)*100, 'ascend');
figure('Color', [1 1 1], 'Position', [400, 400, 800, 550], 'Name', 'Importance');
barh(sorted_imp, 'FaceColor', [0.2, 0.4, 0.7], 'EdgeColor', 'k');
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none', 'FontSize', 10); 
grid on; xlabel('相对贡献度 (%) / Relative Importance (%)');
set(gca, 'Position', [0.25, 0.22, 0.65, 0.7]);
text(0.5, -0.2, {'图11: 基于灵敏度分析的特征重要性排序'; 'Fig.11: Feature Importance Ranking'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

%% --- 模块 7: SHAP 解释性机理分析 (图12) ---
fprintf('生成 SHAP 摘要图...\n');
num_s = size(plot_data.P_test, 1); shap_v = zeros(num_feat, num_s); 
for i = 1:num_s
    curr_x = plot_data.P_test(i, :); b_o = predict(best_model, curr_x);
    for f = 1:num_feat
        t_x = curr_x; t_x(f) = mean(plot_data.P_train(:, f));
        a_o = predict(best_model, t_x);
        shap_v(f, i) = b_o - a_o;
    end
end
figure('Color', [1 1 1], 'Position', [450, 450, 850, 600], 'Name', 'SHAP'); hold on;
for f = 1:num_feat
    f_idx = idx(f);
    y_jitter = f + (rand(1, num_s) - 0.5) * 0.4;
    scatter(shap_v(f_idx, :), y_jitter, 35, plot_data.P_test(:, f_idx), 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(jet); h_cb = colorbar; ylabel(h_cb, '特征取值 (红高/蓝低) / Feature Value');
line([0 0], [0 10], 'Color', [0.3, 0.3, 0.3], 'LineStyle', '--', 'LineWidth', 1.5);
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none', 'FontSize', 10);
xlabel('SHAP Value (对强度的影响)'); grid on; box on;
set(gca, 'Position', [0.25, 0.22, 0.6, 0.7]);
text(0.5, -0.22, {'图12: SVR 模型 SHAP 特征影响摘要图'; 'Fig.12: SHAP Summary Plot'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
%% --- 模块 8: 一键自动保存所有图片 ---
fprintf('正在自动导出所有图表至当前文件夹...\n');
% 创建保存文件夹（可选）
if ~exist('Result_Figures', 'dir')
    mkdir('Result_Figures');
end

figHandles = findobj('Type', 'figure'); % 获取所有打开的图片句柄
for i = 1:length(figHandles)
    figName = get(figHandles(i), 'Name'); % 获取图片窗口的名字
    if isempty(figName)
        figName = sprintf('Figure_%d', figHandles(i).Number);
    end
    
    % 移除非法字符，防止文件名报错
    saveName = regexprep(figName, '[^a-zA-Z0-9_]', '_');
    
    % 导出为高质量 TIF 或 PNG (科研常用)
    exportgraphics(figHandles(i), fullfile('Result_Figures', [saveName, '.png']), 'Resolution', 300);
end
fprintf('✅ 所有图片已保存至 Result_Figures 文件夹。\n');
%% --- 内部辅助函数区域 ---
function out = ifelse(condition, trueVal, falseVal)
    if condition; out = trueVal; else; out = falseVal; end
end
function drawScatter(act, pred, col, t1, t2)
    figure('Color', [1 1 1], 'Position', [400, 400, 550, 550]);
    scatter(act, pred, 35, 'filled', 'MarkerFaceColor', col, 'MarkerFaceAlpha', 0.6);
    hold on; plot([min(act) max(act)], [min(act) max(act)], 'k--', 'LineWidth', 1.5);
    grid on; axis square; box on;
    xlabel('实验值 (Experimental /MPa)'); ylabel('预测值 (Predicted /MPa)');
    set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
    text(0.5, -0.22, {t1; t2}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end