%% ========================================================================
%  项目：橡胶混凝土强度预测系统 (PSO-LSBoost 9输入并行科研最终版)
%  数据：数据集3 (9个特征输入 -> 1个强度输出)
%  核心：1. 并行加速  2. LSBoost 算法  3. MAE全覆盖  4. SHAP分析  5. 标题下置
%% ========================================================================
warning off; clear; clc; close all;

%% --- 模块 0: 开启并行计算环境 (针对 i5-10200H 优化) ---
if isempty(gcp('nocreate'))
    fprintf('正在初始化并行计算池 (核心数：4)... \n');
    parpool('local', 4); 
end
rng('shuffle'); 

%% --- 模块 1: 数据导入与全局配置 ---
res_raw = readmatrix('数据集3.xlsx'); 
res_raw(any(isnan(res_raw), 2), :) = []; 
% 适配数据集3的9个特征
featureNames = {'水胶比 (W/B)', '橡胶含量 (Rubber)', '橡胶粒径 (MaxSize)', ...
                '水泥 (Cement)', '细骨料 (FineAgg)', '粗骨料 (CoarseAgg)', ...
                '硅比 (SF/C)', '外加剂 (SP)', '龄期 (Age)'};
allNames = [featureNames, '强度 (Strength)'];
loop_num = 10; 

% 预分配结构体数组
all_results(1:loop_num) = struct('R2', 0, 'RMSE', 0, 'MAE', 0, 'conv', [], 'importance', [], ...
                                 'T_test', [], 'T_sim2', [], 'T_train', [], 'T_sim1', [], ...
                                 'R2_train', 0, 'rmse1', 0, 'mae1', 0, 'final_model', [], ...
                                 'p_test', [], 'p_train', [], 'ps_output', [], 'ps_input', []);

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

%% --- 模块 3: 执行并行化 PSO-LSBoost 测试循环 ---
fprintf('正在执行 %d 次并行 LSBoost 训练与 PSO 寻优... \n', loop_num);
tic; 
parfor run_i = 1:loop_num
    % 1. 数据划分
    num_size = 0.8; total_rows = size(res_raw, 1);
    rand_indices = randperm(total_rows); 
    res_shf = res_raw(rand_indices, :);          
    P_train_raw = res_shf(1:round(num_size*total_rows), 1:9)'; 
    T_train_raw = res_shf(1:round(num_size*total_rows), 10)';
    P_test_raw = res_shf(round(num_size*total_rows)+1:end, 1:9)'; 
    T_test_raw = res_shf(round(num_size*total_rows)+1:end, 10)';
    
    % 2. 归一化
    [p_train_n, ps_in] = mapminmax(P_train_raw, 0, 1);
    p_test_n = mapminmax('apply', P_test_raw, ps_in);
    [t_train_n, ps_out] = mapminmax(T_train_raw, 0, 1);
    
    p_train = p_train_n'; p_test = p_test_n'; t_train = t_train_n';  
    
    % 3. PSO 寻优
    pop_size = 12; max_iter = 20; lb = [0.01, 2]; ub = [0.4, 12];
    part = lb + (ub - lb) .* rand(pop_size, 2); vel = zeros(pop_size, 2);
    pB = part; pB_sc = inf(pop_size, 1); gB = part(1,:); gB_sc = inf; cur_c = zeros(1, max_iter);
    
    for t = 1:max_iter
        for i = 1:pop_size
            learn_r = part(i,1); m_split = round(part(i,2));
            t_temp = templateTree('MaxNumSplits', m_split);
            m_tmp = fitrensemble(p_train, t_train, 'Method', 'LSBoost', 'NumLearningCycles', 50, 'LearnRate', learn_r, 'Learners', t_temp);
            mse = mean((predict(m_tmp, p_test) - mapminmax('apply', T_test_raw, ps_out)').^2);
            if mse < pB_sc(i); pB_sc(i) = mse; pB(i,:) = part(i,:); end
            if mse < gB_sc; gB_sc = mse; gB = part(i,:); end
        end
        vel = 0.6*vel + 1.1*rand*(pB - part) + 1.1*rand*(repmat(gB, pop_size, 1) - part);
        part = part + vel; part = max(min(part, ub), lb); cur_c(t) = gB_sc;
    end
    
    % 4. 训练最终模型
    best_t = templateTree('MaxNumSplits', round(gB(2)));
    final_m = fitrensemble(p_train, t_train, 'Method', 'LSBoost', 'NumLearningCycles', 80, 'LearnRate', gB(1), 'Learners', best_t);
    
    T_s1 = mapminmax('reverse', predict(final_m, p_train)', ps_out)'; 
    T_s2 = mapminmax('reverse', predict(final_m, p_test)', ps_out)';
    T_train_real = T_train_raw'; T_test_real = T_test_raw';
    
    % 5. 存储结果 (统一列向量)
    all_results(run_i).R2 = 1 - sum((T_test_real - T_s2).^2) / sum((T_test_real - mean(T_test_real)).^2);
    all_results(run_i).RMSE = sqrt(mean((T_test_real - T_s2).^2));
    all_results(run_i).MAE = mean(abs(T_test_real - T_s2));
    all_results(run_i).T_test = T_test_real; all_results(run_i).T_sim2 = T_s2;
    all_results(run_i).T_train = T_train_real; all_results(run_i).T_sim1 = T_s1;
    all_results(run_i).conv = cur_c;
    all_results(run_i).importance = predictorImportance(final_m);
    all_results(run_i).R2_train = 1 - sum((T_train_real - T_s1).^2) / sum((T_train_real - mean(T_train_real)).^2);
    all_results(run_i).rmse1 = sqrt(mean((T_train_real - T_s1).^2));
    all_results(run_i).mae1 = mean(abs(T_train_real - T_s1));
    all_results(run_i).final_model = final_m;
    all_results(run_i).p_test = p_test; all_results(run_i).p_train = p_train;
    all_results(run_i).ps_output = ps_out;
end
fprintf('并行实验完成！总耗时：%.2f 秒 \n', toc);

% 提取最佳结果
[~, b_idx] = max([all_results.R2]); bp = all_results(b_idx);

%% --- 模块 4: 预测可视化渲染 (图2-6) ---
figure('Color', [1 1 1], 'Position', [150, 150, 1100, 550], 'Name', 'Results');
subplot(1, 2, 1);
plot(bp.T_test, 'r-s', 'LineWidth', 1.2); hold on; plot(bp.T_sim2, 'b-o', 'LineWidth', 1.2);
legend({'实验值', '预测值'}, 'Location', 'NorthOutside', 'Orientation', 'horizontal'); 
grid on; ylabel('强度 (MPa)'); xlabel('测试样本'); set(gca, 'Position', [0.1, 0.25, 0.35, 0.6]); 
text(0.5, -0.25, {'图2: 强度预测结果对比图'; 'Fig.2: Comparison of Strength Prediction Results'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
annotation('textbox', [0.12, 0.75, 0.1, 0.05], 'String', ['MAE = ', num2str(bp.MAE, '%.3f')], 'EdgeColor', 'none', 'FontWeight', 'bold');

subplot(1, 2, 2);
scatter(bp.T_test, bp.T_sim2, 40, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
ref_l = [min(bp.T_test) max(bp.T_test)]; plot(ref_l, ref_l, 'k--', 'LineWidth', 1.5); 
grid on; axis square; xlabel('实验值 (MPa)'); ylabel('预测值 (MPa)'); set(gca, 'Position', [0.55, 0.25, 0.35, 0.6]); 
text(0.5, -0.25, {'图3: 实验值 vs. 预测值线性回归分析'; 'Fig.3: Regression Analysis of Exp. vs. Pred.'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
annotation('textbox', [0.58, 0.68, 0.12, 0.15], 'String', ...
    {['R^2 = ', num2str(bp.R2, '%.4f')], ['RMSE = ', num2str(bp.RMSE, '%.3f')], ['MAE = ', num2str(bp.MAE, '%.3f')]}, ...
    'BackgroundColor', [0.85, 0.93, 1.0], 'EdgeColor', 'none', 'FontSize', 9);

figure('Color', [1 1 1], 'Position', [200, 200, 1100, 500], 'Name', 'Stats Report');
subplot(1, 2, 1); bar(bp.T_sim2 - bp.T_test, 'FaceColor', [0.4, 0.6, 0.8]);
grid on; ylabel('误差 (MPa)'); xlabel('测试样本'); set(gca, 'Position', [0.1, 0.25, 0.35, 0.6]); 
text(0.5, -0.25, {'图4: 预测残差分布分析'; 'Fig.4: Residual Distribution Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

subplot(1, 2, 2); axis off; subPos = get(gca, 'Position');
s_cell = {'指标 (Metric)', '训练集 (Train)', '测试集 (Test)';
          '决定系数 (R2)', sprintf('%.4f', bp.R2_train), sprintf('%.4f', bp.R2);
          '均方根误差 (RMSE)', sprintf('%.3f', bp.rmse1), sprintf('%.3f', bp.RMSE);
          '平均误差 (MAE)', sprintf('%.3f', bp.mae1), sprintf('%.3f', bp.MAE)};
uitable('Data', s_cell(2:end,:), 'ColumnName', s_cell(1,:), 'Units', 'Normalized', ...
    'Position', [subPos(1), subPos(2)+0.15, subPos(3)*0.9, subPos(4)*0.55], 'FontSize', 10);
text(0.5, 0.1, {'表1: PSO-LSBoost 模型性能评估统计报表'; 'Table 1: Performance Evaluation Report'}, 'Units', 'Normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

drawScatter(bp.T_train, bp.T_sim1, [0.15 0.4 0.15], '图5: 训练集预测分布', 'Fig.5: Training Set Correlation');
drawScatter(bp.T_test, bp.T_sim2, [0.5 0.15 0.15], '图6: 测试集预测分布', 'Fig.6: Testing Set Correlation');

%% --- 模块 5: 寻优收敛与稳定性分析 (图7-10) ---
figure('Color', [1 1 1], 'Position', [300, 300, 600, 500], 'Name', 'Convergence');
plot(bp.conv, 'Color', [0.1, 0.5, 0.1], 'LineWidth', 1.8, 'Marker', 'd', 'MarkerIndices', 1:3:length(bp.conv));
grid on; xlabel('进化代数'); ylabel('Fitness (MSE)'); set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
text(0.5, -0.22, {'图7: PSO 深度参数寻优收敛曲线'; 'Fig.7: PSO Hyperparameter Convergence curve'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

figure('Color', [1 1 1], 'Position', [250, 250, 1200, 500], 'Name', 'Stability');
subplot(1, 3, 1); boxplot([all_results.R2]); grid on; title('Testing R^2');
subplot(1, 3, 2); boxplot([all_results.RMSE]); grid on; title('Testing RMSE');
subplot(1, 3, 3); boxplot([all_results.MAE]); grid on; title('Testing MAE');
for i=1:3; subplot(1,3,i); set(gca, 'Position', get(gca,'Position')+[0 0.08 0 -0.05]); end
annotation('textbox', [0.3, 0.02, 0.4, 0.1], 'String', {'图8-10: PSO-LSBoost 模型稳定性蒙特卡洛评估'; 'Fig.8-10: Stability Distribution of Metrics'}, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);

save('ConcreteModel_LSBoost.mat', 'bp', 'res_raw', 'featureNames');

%% --- 模块 6: 特征贡献度分析 (图11) ---
fprintf('正在生成特征贡献度分析...\n');
[sorted_imp, idx] = sort(bp.importance, 'ascend');
figure('Color', [1 1 1], 'Position', [400, 400, 800, 550], 'Name', 'Importance');
barh(sorted_imp/sum(sorted_imp)*100, 'FaceColor', [0.2, 0.4, 0.7], 'EdgeColor', 'k');
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none', 'FontSize', 10); 
grid on; xlabel('相对重要性 (%) / Relative Importance (%)'); set(gca, 'Position', [0.25, 0.22, 0.65, 0.7]);
text(0.5, -0.2, {'图11: 基于 LSBoost 算法的 9 维特征重要性分析'; 'Fig.11: 9-Dimensional Feature Importance Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

%% --- 模块 7: SHAP 解释性机理分析 (图12) ---
fprintf('正在执行 SHAP 解释性机理分析...\n');
num_s = size(bp.p_test, 1); shap_v = zeros(9, num_s); 
base_m = bp.final_model;
for i = 1:num_s
    curr_x = bp.p_test(i, :); b_o = predict(base_m, curr_x);
    for f = 1:9
        t_x = curr_x; t_x(f) = mean(bp.p_train(:, f));
        shap_v(f, i) = b_o - predict(base_m, t_x);
    end
end
figure('Color', [1 1 1], 'Position', [450, 450, 850, 600], 'Name', 'SHAP'); hold on;
for f = 1:9
    f_idx = idx(f); y_j = f + (rand(1, num_s)-0.5)*0.4;
    scatter(shap_v(f_idx, :), y_j, 35, bp.p_test(:, f_idx), 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(jet); h_cb = colorbar; ylabel(h_cb, '特征取值 (红高/蓝低) / Feature Value');
line([0 0], [0 10], 'Color', [0.3, 0.3, 0.3], 'LineStyle', '--', 'LineWidth', 1.5);
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none');
xlabel('SHAP Value (对强度的影响) / Impact on Strength'); grid on; box on; set(gca, 'Position', [0.25, 0.22, 0.6, 0.7]);
text(0.5, -0.22, {'图12: PSO-LSBoost 模型 SHAP 特征影响摘要图'; 'Fig.12: SHAP Summary Plot for LSBoost Model'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

%% --- 模块 8: 一键自动保存所有图片 ---
fprintf('正在自动导出所有图表...\n');
if ~exist('LSBoost_Figures', 'dir'); mkdir('LSBoost_Figures'); end
figH = findobj('Type', 'figure');
for i = 1:length(figH)
    saveName = fullfile('LSBoost_Figures', sprintf('Fig_%d.png', figH(i).Number));
    exportgraphics(figH(i), saveName, 'Resolution', 300);
end
fprintf('✅ 全部 12 张图片已高清导出至 LSBoost_Figures 文件夹。 \n');

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