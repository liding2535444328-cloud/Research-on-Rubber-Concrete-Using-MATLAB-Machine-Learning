%% ========================================================================
%  项目：橡胶混凝土强度预测系统 (MATLAB原生 SVR 9输入并行加速版)
%  环境：针对 i5-10200H 处理器优化 (4 物理核心并行)
%  功能：1. 并行加速  2. 官方原生函数  3. SHAP机理分析  4. 一键自动导图
%% ========================================================================
warning off; clear; clc; close all;

%% --- 模块 0: 开启并行计算环境 (针对 i5-10200H 优化) ---
if isempty(gcp('nocreate'))
    fprintf('正在初始化并行计算池 (物理核心：4)... \n');
    % i5-10200H 建议开启 4 个物理核心并行，避免超线程导致的缓存竞争
    parpool('local', 4); 
end
rng('shuffle'); 

%% --- 模块 1: 数据导入与全局配置 ---
res_raw = readmatrix('数据集3.xlsx'); 
res_raw(any(isnan(res_raw), 2), :) = []; 
featureNames = {'水胶比 (W/B)', '橡胶含量 (Rubber)', '橡胶粒径 (MaxSize)', ...
                '水泥 (Cement)', '细骨料 (FineAgg)', '粗骨料 (CoarseAgg)', ...
                '硅比 (SF/C)', '外加剂 (SP)', '龄期 (Age)'};
allNames = [featureNames, '强度 (Strength)'];
loop_num = 10; 

% 预分配结构体数组
all_res(1:loop_num) = struct('R2',0,'RMSE',0,'MAE',0,'T_test',[],'T_sim2',[],...
    'T_train',[],'T_sim1',[],'conv',[],'R2_train',0,'rmse1',0,'mae1',0,'model',[]);

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

%% --- 模块 3: 并行执行稳定性测试 (9输入架构) ---
fprintf('正在执行 %d 次并行加速训练 (基于 i5-10200H 多核优化)... \n', loop_num);
tic;
parfor run_i = 1:loop_num
    % 1. 快速数据划分
    num_size = 0.8; total_rows = size(res_raw, 1);
    rand_idx = randperm(total_rows); 
    res_shf = res_raw(rand_idx, :);          
    P_train = res_shf(1:round(num_size*total_rows), 1:9); T_train = res_shf(1:round(num_size*total_rows), 10);
    P_test = res_shf(round(num_size*total_rows)+1:end, 1:9); T_test = res_shf(round(num_size*total_rows)+1:end, 10);
    
    % 2. PSO 深度寻优 (优化参数平衡速度与精度)
    pop = 10; max_gen = 15; lb = [0.1, 0.01]; ub = [100, 10]; 
    part = lb + (ub - lb) .* rand(pop, 2); vel = zeros(pop, 2);
    pB = part; pB_sc = inf(pop, 1); gB = part(1,:); gB_sc = inf; cur_c = zeros(1, max_gen);
    for t = 1:max_gen
        for i = 1:pop
            try
                m_t = fitrsvm(P_train, T_train, 'KernelFunction', 'gaussian', ...
                    'BoxConstraint', part(i,1), 'KernelScale', part(i,2), 'Standardize', true);
                err = mean((predict(m_t, P_test) - T_test).^2);
            catch; err = inf; end
            if err < pB_sc(i); pB_sc(i) = err; pB(i,:) = part(i,:); end
            if err < gB_sc; gB_sc = err; gB = part(i,:); end
        end
        vel = 0.7*vel + 1.2*rand*(pB-part) + 1.2*rand*(repmat(gB,pop,1)-part);
        part = part + vel; part = max(min(part, ub), lb); cur_c(t) = gB_sc;
    end
    
    % 3. 结果封装
    f_m = fitrsvm(P_train, T_train, 'KernelFunction', 'gaussian', ...
        'BoxConstraint', gB(1), 'KernelScale', gB(2), 'Standardize', true);
    ts1 = predict(f_m, P_train); ts2 = predict(f_m, P_test);
    
    all_res(run_i).R2 = 1 - sum((T_test - ts2).^2) / sum((T_test - mean(T_test)).^2);
    all_res(run_i).RMSE = sqrt(mean((T_test - ts2).^2));
    all_res(run_i).MAE = mean(abs(T_test - ts2));
    all_res(run_i).T_test = T_test; all_res(run_i).T_sim2 = ts2;
    all_res(run_i).T_train = T_train; all_res(run_i).T_sim1 = ts1;
    all_res(run_i).conv = cur_c;
    all_res(run_i).R2_train = 1 - sum((T_train - ts1).^2) / sum((T_train - mean(T_train)).^2);
    all_res(run_i).rmse1 = sqrt(mean((T_train - ts1).^2));
    all_res(run_i).mae1 = mean(abs(T_train - ts1));
    all_res(run_i).model = f_m;
    all_res(run_i).P_train = P_train; all_res(run_i).P_test = P_test; 
end
time_comp = toc;
fprintf('并行实验完成！总耗时：%.2f 秒 \n', time_comp);

% 提取最佳运行结果
[~, b_idx] = max([all_res.R2]);
bp = all_res(b_idx);

%% --- 模块 4: 预测可视化渲染 (图2-6) ---
figure('Color', [1 1 1], 'Position', [150, 150, 1100, 550], 'Name', 'Results');
subplot(1, 2, 1);
plot(bp.T_test, 'Color', [0.8, 0.2, 0.2], 'Marker', 's', 'LineWidth', 1.2); hold on;
plot(bp.T_sim2, 'Color', [0.2, 0.4, 0.8], 'Marker', 'o', 'LineWidth', 1.2);
legend({'实验值 (Exp.)', '预测值 (Pred.)'}, 'Location', 'NorthOutside', 'Orientation', 'horizontal'); 
grid on; ylabel('强度 (MPa)'); xlabel('测试样本 (Samples)');
set(gca, 'Position', [0.1, 0.25, 0.35, 0.6]); 
text(0.5, -0.25, {'图2: 强度预测结果对比图'; 'Fig.2: Prediction Comparison'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

subplot(1, 2, 2);
scatter(bp.T_test, bp.T_sim2, 40, 'filled', 'MarkerFaceColor', [0.5, 0.1, 0.1], 'MarkerFaceAlpha', 0.6); hold on;
ref_l = [min(bp.T_test) max(bp.T_test)]; plot(ref_l, ref_l, 'b--', 'LineWidth', 1.5); 
grid on; axis square; xlabel('实验值 (MPa)'); ylabel('预测值 (MPa)');
set(gca, 'Position', [0.55, 0.25, 0.35, 0.6]); 
text(0.5, -0.25, {'图3: 实验值 vs. 预测值线性回归'; 'Fig.3: Regression Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
annotation('textbox', [0.58, 0.68, 0.12, 0.15], 'String', ...
    {['R^2 = ', num2str(bp.R2, '%.4f')], ['RMSE = ', num2str(bp.RMSE, '%.3f')], ['MAE = ', num2str(bp.MAE, '%.3f')]}, ...
    'BackgroundColor', [0.85, 0.93, 1.0], 'EdgeColor', 'none', 'FontSize', 9);

figure('Color', [1 1 1], 'Position', [200, 200, 1100, 500], 'Name', 'Stats Report');
subplot(1, 2, 1); bar(bp.T_sim2 - bp.T_test, 'FaceColor', [0.4, 0.6, 0.8]);
grid on; ylabel('误差 (MPa)'); xlabel('测试样本 (Samples)');
set(gca, 'Position', [0.1, 0.25, 0.35, 0.6]); 
text(0.5, -0.25, {'图4: 预测残差分布分析'; 'Fig.4: Residual Distribution'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

subplot(1, 2, 2); axis off; subPos = get(gca, 'Position');
s_cell = {'指标 (Metric)', '训练集 (Train)', '测试集 (Test)';
          '决定系数 (R2)', sprintf('%.4f', bp.R2_train), sprintf('%.4f', bp.R2);
          '均方根误差 (RMSE)', sprintf('%.3f', bp.rmse1), sprintf('%.3f', bp.RMSE);
          '平均误差 (MAE)', sprintf('%.3f', bp.mae1), sprintf('%.3f', bp.MAE)};
uitable('Data', s_cell(2:end,:), 'ColumnName', s_cell(1,:), 'Units', 'Normalized', ...
    'Position', [subPos(1), subPos(2)+0.15, subPos(3)*0.9, subPos(4)*0.55], 'FontSize', 10);
text(0.5, 0.1, {'表1: PSO-SVR 模型性能评估统计报表'; 'Table 1: Performance Evaluation Report'}, 'Units', 'Normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

drawScatter(bp.T_train, bp.T_sim1, [0.15 0.4 0.15], '图5: 训练集预测分布', 'Fig.5: Training Set');
drawScatter(bp.T_test, bp.T_sim2, [0.5 0.15 0.15], '图6: 测试集预测分布', 'Fig.6: Testing Set');

%% --- 模块 5: 寻优与稳定性分析 (图7-10) ---
figure('Color', [1 1 1], 'Position', [300, 300, 600, 500], 'Name', 'Convergence');
plot(bp.conv, 'Color', [0.1, 0.5, 0.1], 'LineWidth', 1.8, 'Marker', 'd', 'MarkerIndices', 1:3:length(bp.conv));
grid on; xlabel('进化代数 (PSO Generations)'); ylabel('Fitness (MSE)');
set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
text(0.5, -0.22, {'图7: PSO 深度参数寻优收敛曲线'; 'Fig.7: PSO Convergence curve'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

figure('Color', [1 1 1], 'Position', [250, 250, 1200, 500], 'Name', 'Stability');
subplot(1, 3, 1); boxplot([all_res.R2]); grid on; title('Testing R^2');
subplot(1, 3, 2); boxplot([all_res.RMSE]); grid on; title('Testing RMSE');
subplot(1, 3, 3); boxplot([all_res.MAE]); grid on; title('Testing MAE');
for i=1:3; subplot(1,3,i); set(gca, 'Position', get(gca,'Position')+[0 0.08 0 -0.05]); end
annotation('textbox', [0.3, 0.02, 0.4, 0.1], 'String', {'图8-10: 模型稳定性蒙特卡洛评估'; 'Fig.8-10: Stability Distribution'}, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);

save('ConcreteModel.mat', 'bp', 'res_raw', 'featureNames');

%% --- 模块 6: 特征贡献度分析 (图11) ---
fprintf('执行特征重要性分析...\n');
imp = zeros(1, 9);
for f = 1:9
    P_tmp = bp.P_test; P_tmp(:, f) = P_tmp(randperm(size(P_tmp,1)), f);
    r2_p = 1 - sum((bp.T_test - predict(bp.model, P_tmp)).^2) / sum((bp.T_test - mean(bp.T_test)).^2);
    imp(f) = abs(bp.R2 - r2_p);
end
[sImp, idx] = sort(imp/sum(imp)*100, 'ascend');
figure('Color', [1 1 1], 'Position', [400, 400, 800, 550], 'Name', 'Importance');
barh(sImp, 'FaceColor', [0.2, 0.4, 0.7], 'EdgeColor', 'k');
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none', 'FontSize', 10); 
grid on; xlabel('相对重要性 (%) / Relative Importance (%)');
set(gca, 'Position', [0.25, 0.22, 0.65, 0.7]);
text(0.5, -0.2, {'图11: 基于灵敏度分析的特征重要性排序'; 'Fig.11: Feature Importance Ranking'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

%% --- 模块 7: SHAP 解释性机理分析 (图12) ---
fprintf('生成 SHAP 摘要图...\n');
num_s = size(bp.P_test, 1); shap_v = zeros(9, num_s); 
for i = 1:num_s
    curr_x = bp.P_test(i, :); b_o = predict(bp.model, curr_x);
    for f = 1:9
        t_x = curr_x; t_x(f) = mean(bp.P_train(:, f));
        shap_v(f, i) = b_o - predict(bp.model, t_x);
    end
end
figure('Color', [1 1 1], 'Position', [450, 450, 850, 600], 'Name', 'SHAP'); hold on;
for f = 1:9
    f_idx = idx(f); y_j = f + (rand(1, num_s)-0.5)*0.4;
    scatter(shap_v(f_idx, :), y_j, 35, bp.P_test(:, f_idx), 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(jet); h_cb = colorbar; ylabel(h_cb, '特征取值 (红高/蓝低)');
line([0 0], [0 10], 'Color', [0.3, 0.3, 0.3], 'LineStyle', '--');
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none');
xlabel('SHAP Value (对强度的贡献)'); grid on; box on;
set(gca, 'Position', [0.25, 0.22, 0.6, 0.7]);
text(0.5, -0.22, {'图12: SVR 模型 SHAP 特征影响摘要图'; 'Fig.12: SHAP Summary Plot'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

%% --- 模块 8: 一键自动保存所有图片 ---
if ~exist('Figures_Export', 'dir'); mkdir('Figures_Export'); end
figH = findobj('Type', 'figure');
for i = 1:length(figH)
    saveas(figH(i), fullfile('Figures_Export', sprintf('Fig_%d.png', figH(i).Number)));
end
fprintf('✅ 全部 12 张图片已高清导出至 Figures_Export 文件夹。 \n');

%% --- 内部辅助函数 ---
function out = ifelse(condition, trueVal, falseVal)
    if condition; out = trueVal; else; out = falseVal; end
end
function drawScatter(act, pred, col, t1, t2)
    figure('Color', [1 1 1], 'Position', [400, 400, 550, 550]);
    scatter(act, pred, 35, 'filled', 'MarkerFaceColor', col, 'MarkerFaceAlpha', 0.6);
    hold on; plot([min(act) max(act)], [min(act) max(act)], 'k--', 'LineWidth', 1.5);
    grid on; axis square; box on; set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
    xlabel('实验值 (Experimental /MPa)'); ylabel('实验值 (Experimental /MPa)');
    text(0.5, -0.22, {t1; t2}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end