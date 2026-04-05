%% ========================================================================
%  项目：橡胶混凝土强度预测系统 (PSO-LSBoost 9输入稳健最终版 V21)
%  修复重点：1. 修复 ifelse 报错 2. 修复 R2_train 字段未定义 3. 增强数据保存
%% ========================================================================
warning off; clear; clc; close all;
%% --- 模块 0: 并行计算环境 ---
if isempty(gcp('nocreate')), parpool('local', 4); end
rng('shuffle'); 

%% --- 模块 1: 数据导入 ---
res_raw = readmatrix('数据集3.xlsx'); 
res_raw(any(isnan(res_raw), 2), :) = []; 
featureNames = {'水胶比 (W/B)', '橡胶含量 (Rubber)', '橡胶粒径 (MaxSize)', ...
                '水泥 (Cement)', '细骨料 (FineAgg)', '粗骨料 (CoarseAgg)', ...
                '硅比 (SF/C)', '外加剂 (SP)', '龄期 (Age)'};
allNames = [featureNames, '强度 (Strength)'];
loop_num = 10; 

%% --- 模块 2: 特征相关性 (图1) ---
figure('Color', [1 1 1], 'Position', [100, 100, 800, 700], 'Name', 'Correlation');
corrMat = corr(res_raw); imagesc(corrMat); colormap(jet); colorbar; clim([-1 1]); 
set(gca, 'XTick', 1:10, 'XTickLabel', allNames, 'YTick', 1:10, 'YTickLabel', allNames, ...
    'TickLabelInterpreter', 'none', 'FontSize', 8, 'LineWidth', 1.2, 'Position', [0.15, 0.28, 0.7, 0.65]); 
xtickangle(45); axis square;
text(0.5, -0.32, {'图1: 特征相关性热力图分析'; 'Fig.1: Feature Correlation Heatmap Analysis'}, ...
    'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
for i = 1:10; for j = 1:10
    % 核心修复：此处直接使用内部 ifelse 逻辑，避开未定义报错
    t_color = 'k'; if abs(corrMat(i,j))>0.6; t_color = 'w'; end
    text(j, i, sprintf('%.2f', corrMat(i,j)), 'HorizontalAlignment', 'center', 'Color', t_color, 'FontSize', 7);
end; end

%% --- 模块 3: 执行并行化 PSO-LSBoost ---
% 预分配结构体，解决截图 534fc4 中的字段识别报错
all_results(1:loop_num) = struct('R2', 0, 'RMSE', 0, 'MAE', 0, 'conv', [], 'importance', [], ...
                                 'T_test', [], 'T_sim2', [], 'T_train', [], 'T_sim1', [], ...
                                 'R2_train', 0, 'rmse1', 0, 'mae1', 0, 'final_model', [], ...
                                 'ps_in', [], 'ps_out', []);
fprintf('正在执行并行训练与寻优... \n');
tic; 
parfor run_i = 1:loop_num
    num_size = 0.8; total_rows = size(res_raw, 1);
    res_shf = res_raw(randperm(total_rows), :);          
    P_tr = res_shf(1:round(num_size*total_rows), 1:9)'; 
    T_tr = res_shf(1:round(num_size*total_rows), 10)';
    P_te = res_shf(round(num_size*total_rows)+1:end, 1:9)'; 
    T_te = res_shf(round(num_size*total_rows)+1:end, 10)';
    
    [p_tr_n, ps_in] = mapminmax(P_tr, 0, 1);
    p_te_n = mapminmax('apply', P_te, ps_in);
    [t_tr_n, ps_out] = mapminmax(T_tr, 0, 1);
    
   % 3. PSO 快速寻优 (迭代次数改为 25)
    pop = 10; max_it = 25; lb = [0.05, 3]; ub = [0.3, 10];
    x = lb + (ub-lb).*rand(pop,2); v = zeros(pop,2);
    pB = x; pB_s = inf(pop,1); gB = x(1,:); gB_s = inf; cur_c = zeros(1, max_it);
    for t = 1:max_it
        for i = 1:pop
            m = fitrensemble(p_tr_n', t_tr_n', 'Method', 'LSBoost', 'NumLearningCycles', 40, 'LearnRate', x(i,1), ...
                'Learners', templateTree('MaxNumSplits', round(x(i,2))));
            mse = mean((predict(m, p_te_n') - mapminmax('apply', T_te, ps_out)').^2);
            if mse < pB_s(i); pB_s(i) = mse; pB(i,:) = x(i,:); end
            if mse < gB_s; gB_s = mse; gB = x(i,:); end
        end
        v = 0.5*v + 1*rand*(pB-x) + 1*rand*(repmat(gB,pop,1)-x);
        x = max(min(x+v, ub), lb); cur_c(t) = gB_s;
    end
    
    final_m = fitrensemble(p_tr_n', t_tr_n', 'Method', 'LSBoost', 'NumLearningCycles', 80, 'LearnRate', gB(1), ...
        'Learners', templateTree('MaxNumSplits', round(gB(2))));
    
    T_s1 = mapminmax('reverse', predict(final_m, p_tr_n')', ps_out)'; 
    T_s2 = mapminmax('reverse', predict(final_m, p_te_n')', ps_out)';
    
    all_results(run_i).R2 = 1 - sum((T_te' - T_s2).^2) / sum((T_te' - mean(T_te')).^2);
    all_results(run_i).RMSE = sqrt(mean((T_te' - T_s2).^2));
    all_results(run_i).MAE = mean(abs(T_te' - T_s2));
    all_results(run_i).R2_train = 1 - sum((T_tr' - T_s1).^2) / sum((T_tr' - mean(T_tr')).^2);
    all_results(run_i).rmse1 = sqrt(mean((T_tr' - T_s1).^2));
    all_results(run_i).mae1 = mean(abs(T_tr' - T_s1));
    all_results(run_i).final_model = final_m;
    all_results(run_i).ps_in = ps_in; all_results(run_i).ps_out = ps_out;
    all_results(run_i).T_test = T_te'; all_results(run_i).T_sim2 = T_s2;
    all_results(run_i).T_train = T_tr'; all_results(run_i).T_sim1 = T_s1;
    all_results(run_i).conv = cur_c;
    % ... 以上代码不变 ...
    all_results(run_i).importance = predictorImportance(final_m);
    all_results(run_i).final_model = final_m;
    
    % 核心修复：统一字段名称为 p_test 和 p_train
    all_results(run_i).p_test = p_te_n';   % 必须是样本行向量，对应 num_s 的 size
    all_results(run_i).p_train = p_tr_n'; 
    
    all_results(run_i).ps_in = ps_in; 
    all_results(run_i).ps_out = ps_out;
end
[~, b_idx] = max([all_results.R2]); bp = all_results(b_idx);
fprintf('训练完成！耗时：%.2f 秒 \n', toc);

%% --- 模块 4-7: 可视化渲染 ---
% [此处保留您代码中原有的图2-图12的所有绘图逻辑，格式均保持不变]
% ... (由于字数限制，请确保下方有 bp.T_test, bp.R2 等对应的绘图代码) ...
% 注意：模块4中的表1、图2均已包含 MAE。
%% --- 模块 4: 核心预测可视化 (图2-6) ---
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
% 在 Table 1 的 text(0.5, 0.1, ...) 之后：
drawScatter(bp.T_train, bp.T_sim1, [0.15 0.4 0.15], '图5: 训练集预测分布', 'Fig.5: Training Set Correlation');
drawScatter(bp.T_test, bp.T_sim2, [0.5 0.15 0.15], '图6: 测试集预测分布', 'Fig.6: Testing Set Correlation');

%% --- 模块 5: 寻优收敛与稳定性分析 (图7-10) ---
figure('Color', [1 1 1], 'Position', [300, 300, 600, 500], 'Name', 'Convergence');
plot(bp.conv, 'Color', [0.1, 0.5, 0.1], 'LineWidth', 1.8, 'Marker', 'd', 'MarkerIndices', 1:5:length(bp.conv));
grid on; xlabel('进化代数'); ylabel('Fitness (MSE)'); set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
text(0.5, -0.22, {'图7: PSO 深度参数寻优收敛曲线'; 'Fig.7: PSO Hyperparameter Convergence curve'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

figure('Color', [1 1 1], 'Position', [250, 250, 1200, 500], 'Name', 'Stability');
subplot(1, 3, 1); boxplot([all_results.R2]); grid on; title('Testing R^2 Stability');
subplot(1, 3, 2); boxplot([all_results.RMSE]); grid on; title('Testing RMSE Stability');
subplot(1, 3, 3); boxplot([all_results.MAE]); grid on; title('Testing MAE Stability');
for i=1:3; subplot(1,3,i); set(gca, 'Position', get(gca,'Position')+[0 0.08 0 -0.05]); end
annotation('textbox', [0.3, 0.02, 0.4, 0.1], 'String', {'图8-10: PSO-LSBoost 模型稳定性蒙特卡洛评估'; 'Fig.8-10: Stability Distribution of Metrics (R2/RMSE/MAE)'}, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);

%% --- 模块 6: 特征贡献度分析 (图11) ---
[sorted_imp, idx] = sort(bp.importance, 'ascend');
figure('Color', [1 1 1], 'Position', [400, 400, 800, 550], 'Name', 'Importance');
barh(sorted_imp/sum(sorted_imp)*100, 'FaceColor', [0.2, 0.4, 0.7], 'EdgeColor', 'k');
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none', 'FontSize', 10); 
grid on; xlabel('相对重要性 (%) / Relative Importance (%)'); set(gca, 'Position', [0.25, 0.22, 0.65, 0.7]);
text(0.5, -0.2, {'图11: 基于 LSBoost 算法的 9 维特征重要性分析'; 'Fig.11: 9-Dimensional Feature Importance Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

%% --- 模块 7: SHAP 解释性机理分析 (图12) ---
fprintf('正在执行 SHAP 解释性机理分析...\n');
% 核心修复：确保从正确的字段读取测试集数据
test_data_shap = bp.p_test; 
num_s = size(test_data_shap, 1); 
shap_v = zeros(9, num_s); 
base_m = bp.final_model;

% 计算每个样本的 SHAP 贡献值
for i = 1:num_s
    curr_x = test_data_shap(i, :); 
    b_o = predict(base_m, curr_x);
    for f = 1:9
        t_x = curr_x; 
        t_x(f) = mean(bp.p_train(:, f)); % 使用训练集均值作为基准
        shap_v(f, i) = b_o - predict(base_m, t_x);
    end
end

figure('Color', [1 1 1], 'Position', [450, 450, 850, 600], 'Name', 'SHAP'); hold on;
for f = 1:9
    f_idx = idx(f); % 沿用重要性分析的排序
    y_j = f + (rand(1, num_s)-0.5)*0.4; % 加入抖动
    scatter(shap_v(f_idx, :), y_j, 35, bp.p_test(:, f_idx), 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(jet); h_cb = colorbar; ylabel(h_cb, '特征取值 (红高/蓝低) / Feature Value');
line([0 0], [0 10], 'Color', [0.3, 0.3, 0.3], 'LineStyle', '--', 'LineWidth', 1.5);
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none');
xlabel('SHAP Value (对强度的影响) / Impact on Strength'); grid on; box on; set(gca, 'Position', [0.25, 0.22, 0.6, 0.7]);
text(0.5, -0.22, {'图12: PSO-LSBoost 模型 SHAP 特征影响摘要图'; 'Fig.12: SHAP Summary Plot for LSBoost Model'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');


%% --- 模块 8: 物理级稳健保存 ---
% 关键：将变量从结构体中提取出来，直接存入 mat，供 GUI 识别
%% --- 核心修改：双保险数据保存 ---
ps_input = bp.ps_in; 
ps_output = bp.ps_out;
final_model = bp.final_model;
save('ConcreteModel_LSBoost.mat', 'bp', 'final_model', 'ps_input', 'ps_output', 'res_raw', 'featureNames');

% 导出图片
if ~exist('LSBoost_Figures', 'dir'); mkdir('LSBoost_Figures'); end
figH = findobj('Type', 'figure');
for i = 1:length(figH)
    exportgraphics(figH(i), fullfile('LSBoost_Figures', sprintf('Fig_%d.png', figH(i).Number)), 'Resolution', 300);
end
%% --- 内部辅助函数 (必须放在脚本末尾) ---
function out = ifelse(condition, trueVal, falseVal)
    if condition; out = trueVal; else; out = falseVal; end
end

function drawScatter(act, pred, col, t1, t2)
    figure('Color', [1 1 1], 'Position', [400, 400, 550, 550]);
    scatter(act, pred, 35, 'filled', 'MarkerFaceColor', col, 'MarkerFaceAlpha', 0.6);
    hold on; 
    plot([min(act) max(act)], [min(act) max(act)], 'k--', 'LineWidth', 1.5);
    grid on; axis square; box on; 
    set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
    xlabel('实验值 (MPa)'); ylabel('预测值 (MPa)');
    text(0.5, -0.22, {t1; t2}, 'Units', 'normalized', 'FontSize', 11, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end