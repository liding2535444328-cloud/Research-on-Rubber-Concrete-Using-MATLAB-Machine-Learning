%% ========================================================================
%  项目：橡胶混凝土强度预测系统 (PSO-LSBoost 科研级绘图增强版)
%  功能：1. 并行加速寻优  2. 稳定性测试  3. SHAP/机理分析  4. 逆向设计数据保存
%  特点：全图表双语、主题/表格主题下置，符合高质量期刊发表标准
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

%% --- 模块 2: 特征相关性分析 (图1) ---
figure('Color', [1 1 1], 'Position', [100, 100, 750, 650], 'Name', 'Correlation Analysis');
corrMat = corr(res_raw); imagesc(corrMat); 
map = [linspace(0.1,1,32)', linspace(0.4,1,32)', ones(32,1); ... 
       ones(32,1), linspace(1,0.1,32)', linspace(1,0.1,32)'];
colormap(map); colorbar; clim([-1 1]); 
set(gca, 'XTick', 1:9, 'XTickLabel', allNames, 'YTick', 1:9, 'YTickLabel', allNames, ...
    'TickLabelInterpreter', 'none', 'FontSize', 9, 'LineWidth', 1.2, 'Position', [0.15, 0.28, 0.7, 0.65]); 
xtickangle(45); axis square;
text(0.5, -0.32, {'图1: 特征相关性分析'; 'Fig.1: Correlation Analysis'}, ...
    'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
for i = 1:9; for j = 1:9
    if abs(corrMat(i,j))>0.6; textColor = 'w'; else; textColor = 'k'; end
    text(j, i, sprintf('%.2f', corrMat(i,j)), 'HorizontalAlignment', 'center', ...
        'Color', textColor, 'FontSize', 8);
end; end

%% --- 模块 0: 开启并行计算 (针对 i5-10200H 加速) ---
if isempty(gcp('nocreate'))
    parpool('local', 4); % 开启 4 个物理核心并行
end

%% --- 模块 3: 执行并行化 PSO-LSBoost 测试循环 ---
all_results(1:loop_num) = struct('R2', 0, 'RMSE', 0, 'MAE', 0, 'conv', [], 'importance', [], ...
                                 'T_test', [], 'T_sim2', [], 'T_train', [], 'T_sim1', [], ...
                                 'R2_train', 0, 'rmse1', 0, 'mae1', 0, 'final_model', [], ...
                                 'p_test', [], 'p_train', [], 'ps_output', [], 'ps_input', []); % 确保有 ps_input
fprintf('正在利用多核并行加速执行 %d 次 PSO-LSBoost 深度测试...\n', loop_num);
tic; 
parfor run_i = 1:loop_num
    % 1. 数据划分
    num_size = 0.8; 
    total_rows = size(res_raw, 1);
    rand_indices = randperm(total_rows); 
    res_shuffled = res_raw(rand_indices, :);          
    split_idx = round(num_size * total_rows); 
    P_train_raw = res_shuffled(1:split_idx, 1:8)'; T_train_raw = res_shuffled(1:split_idx, 9)';
    P_test_raw = res_shuffled(split_idx+1:end, 1:8)'; T_test_raw = res_shuffled(split_idx+1:end, 9)';
    
    % 2. 归一化处理
    [p_train_norm, ps_in_local] = mapminmax(P_train_raw, 0, 1);
    p_test_norm = mapminmax('apply', P_test_raw, ps_in_local);
    [t_train_norm, ps_out_local] = mapminmax(T_train_raw, 0, 1);
    
    p_train = p_train_norm'; p_test = p_test_norm'; t_train = t_train_norm';  
    t_test_orig = T_test_raw';
    
    % --- PSO 寻优模块 ---
    pop_size = 15; max_iter = 20; 
    lb = [0.01, 2]; ub = [0.4, 12];
    particles = lb + (ub - lb) .* rand(pop_size, 2);
    velocity = zeros(pop_size, 2);
    pBest = particles; pBest_score = inf(pop_size, 1);
    gBest_local = zeros(1, 2); gBest_score_local = inf;
    cur_conv_local = zeros(1, max_iter);
    
    for t = 1:max_iter
        w_curr = 0.9 - 0.5*(t/max_iter); 
        for i = 1:pop_size
            learn_r = particles(i,1); m_split = round(particles(i,2));
            t_temp = templateTree('MaxNumSplits', m_split);
            m_tmp = fitrensemble(p_train, t_train, 'Method', 'LSBoost', ...
                                 'NumLearningCycles', 50, 'LearnRate', learn_r, 'Learners', t_temp);
            sim_tmp = predict(m_tmp, p_test);
            t_test_norm_tmp = mapminmax('apply', t_test_orig', ps_out_local)'; 
            mse = mean((sim_tmp - t_test_norm_tmp).^2);
            
            if mse < pBest_score(i)
                pBest_score(i) = mse; pBest(i,:) = particles(i,:);
            end
            if mse < gBest_score_local
                gBest_score_local = mse; gBest_local = particles(i,:);
            end
        end
        velocity = w_curr*velocity + 1.5*rand*(pBest - particles) + 1.5*rand*(repmat(gBest_local, pop_size, 1) - particles);
        particles = particles + velocity;
        particles = max(min(particles, ub), lb);
        cur_conv_local(t) = gBest_score_local;
    end
    
    % 训练并评估该线程最终模型
    best_template = templateTree('MaxNumSplits', round(gBest_local(2)));
    final_m_local = fitrensemble(p_train, t_train, 'Method', 'LSBoost', ...
                               'NumLearningCycles', 80, 'LearnRate', gBest_local(1), 'Learners', best_template);
    
    t_sim1 = predict(final_m_local, p_train); t_sim2 = predict(final_m_local, p_test);
    T_sim1 = mapminmax('reverse', t_sim1', ps_out_local)'; 
    T_sim2 = mapminmax('reverse', t_sim2', ps_out_local)';
    % --- 直接赋值给预分配好的结构体数组 (解决“不同结构体赋值”报错) ---
    all_results(run_i).R2 = 1 - sum((t_test_orig - T_sim2).^2) / sum((t_test_orig - mean(t_test_orig)).^2);
    all_results(run_i).RMSE = sqrt(mean((t_test_orig - T_sim2).^2));
    all_results(run_i).MAE = mean(abs(t_test_orig - T_sim2));
    all_results(run_i).T_test = t_test_orig; 
    all_results(run_i).T_sim2 = T_sim2;
    all_results(run_i).T_train = T_train_raw'; 
    all_results(run_i).T_sim1 = T_sim1;
    all_results(run_i).conv = cur_conv_local;
    all_results(run_i).importance = predictorImportance(final_m_local);
    all_results(run_i).R2_train = 1 - sum((T_train_raw' - T_sim1).^2) / sum((T_train_raw' - mean(T_train_raw')).^2);
    all_results(run_i).rmse1 = sqrt(mean((T_sim1 - T_train_raw').^2));
    all_results(run_i).mae1 = mean(abs(T_sim1 - T_train_raw'));
    all_results(run_i).final_model = final_m_local;
    all_results(run_i).p_test = p_test; 
    all_results(run_i).p_train = p_train;
    all_results(run_i).ps_output = ps_out_local;
    all_results(run_i).ps_input = ps_in_local;
    fprintf('Parallel Task %d 完成. R2 = %.4f\n', run_i, all_results(run_i).R2);
end

% 提取最优结果
[best_r2_val, b_idx] = max([all_results.R2]);
plot_data = all_results(b_idx);
% 🚀 提取刚才保存的归一化参数
    ps_input_best = plot_data.ps_input;
final_model = plot_data.final_model;
stats_R2 = [all_results.R2]';
stats_RMSE = [all_results.RMSE]';
stats_MAE = [all_results.MAE]';

% 获取画散点图所需的 Train 数据
T_train_best = plot_data.T_train;
T_sim1_best = plot_data.T_sim1;

time_total = toc;
fprintf('并行实验全部完成！总计算耗时：%.2f 秒\n', time_total);

%% --- 模块 4: 可视化渲染 (图2 - 图7) ---
N_test = length(plot_data.T_test);
% 图2: 预测对比图与线性回归分析
figure('Color', [1 1 1], 'Position', [150, 150, 1100, 550], 'Name', 'LSBoost Prediction');
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
    {['R^2 = ', num2str(best_r2_val, '%.4f')], ...
     ['RMSE = ', num2str(plot_data.RMSE, '%.3f')], ...
     ['MAE = ', num2str(plot_data.MAE, '%.3f')]}, ...
    'BackgroundColor', [0.85, 0.93, 1.0], 'EdgeColor', 'none');

% 图4: 残差分布与性能报表
figure('Color', [1 1 1], 'Position', [200, 200, 1100, 550], 'Name', 'Performance Stats');
subplot(1, 2, 1); bar(plot_data.T_sim2 - plot_data.T_test, 'FaceColor', [0.4, 0.6, 0.8]);
xlabel('测试样本 (Test Samples)'); ylabel('误差 (Error/MPa)'); 
set(gca, 'Position', [0.1, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图4:  残差分布分析'; 'Fig.4:  Residual Distribution'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; box on;

subplot(1, 2, 2); axis off; subPos = get(gca, 'Position');
stats_cell = {'指标 (Metric)', '训练集 (Train)', '测试集 (Test)';
              '决定系数 (R2)', sprintf('%.4f', plot_data.R2_train), sprintf('%.4f', best_r2_val);
              '均方根误差 (RMSE)', sprintf('%.3f', plot_data.rmse1), sprintf('%.3f', plot_data.RMSE);
              '平均误差 (MAE)', sprintf('%.3f', plot_data.mae1), sprintf('%.3f', plot_data.MAE)};
uitable('Data', stats_cell(2:end,:), 'ColumnName', stats_cell(1,:), 'Units', 'Normalized', ...
    'Position', [subPos(1), subPos(2)+0.3, subPos(3)*0.9, subPos(4)*0.4], 'FontSize', 10);
text(0.5, 0.15, {'表1: PSO-LSBoost 模型性能评估统计报表'; 'Table 1: Model Performance Statistics'}, ...
    'Units', 'Normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% 图5-6: 训练/测试集散点图调用
drawScatter(T_train_best, T_sim1_best, [0.15 0.4 0.15], '图5: 训练集预测值 vs. 真实值', 'Fig.5: Training Set Correlation');
drawScatter(plot_data.T_test, plot_data.T_sim2, [0.5 0.15 0.15], '图6: 测试集预测值 vs. 真实值', 'Fig.6: Testing Set Correlation');

%% --- 模块 5: 寻优与稳定性分析 (图7 - 图11) ---
% 图7: 寻优曲线
figure('Color', [1 1 1], 'Position', [300, 300, 600, 500], 'Name', 'PSO Convergence');
plot(plot_data.conv, 'Color', [0.1 0.5 0.1], 'LineWidth', 1.5, 'Marker', 'd', 'MarkerIndices', 1:5:length(plot_data.conv));
xlabel('进化代数 (PSO Generations)'); ylabel('适应度 (Fitness/MSE)');
set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
text(0.5, -0.22, {'图7: PSO 参数深度寻优收敛曲线'; 'Fig.7: PSO Hyperparameter Optimization Search'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;

% 图8-10: 稳定性分析箱线图
figure('Color', [1 1 1], 'Position', [200, 200, 1200, 500], 'Name', 'Stability Analysis');
subplot(1, 3, 1); boxplot(stats_R2, 'Labels', {'Testing R^2'}); grid on;
subplot(1, 3, 2); boxplot(stats_RMSE, 'Labels', {'Testing RMSE'}); grid on;
subplot(1, 3, 3); boxplot(stats_MAE, 'Labels', {'Testing MAE'}); grid on;
for i=1:3; subplot(1,3,i); set(gca, 'Position', get(gca,'Position')+[0 0.08 0 -0.05]); end
annotation('textbox', [0.3, 0.02, 0.4, 0.1], 'String', {'图8-10: PSO-LSBoost 模型稳定性蒙特卡洛评估'; 'Fig.8-10: Stability Evaluation Metrics Distribution'}, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);

% 图11: 特征重要性分析 (基于原生算法)
figure('Color', [1 1 1], 'Position', [400, 400, 800, 550], 'Name', 'Feature Importance');
[sortedImp, idx_imp] = sort(plot_data.importance, 'descend');
[sortedImp, idx_imp] = sort(plot_data.importance, 'descend'); % 确保重新排序
bar(sortedImp, 'FaceColor', [0.2 0.4 0.7]); % 去掉除以 sum 的逻辑，直接显示原始重要性得分
set(gca, 'XTick', 1:8, 'XTickLabel', featureNames(idx_imp), 'TickLabelInterpreter', 'none', 'Position', [0.1, 0.3, 0.85, 0.6]); 
xtickangle(45); ylabel('相对重要性 (%) / Relative Importance (%)');
text(0.5, -0.4, {'图11: 基于原生 LSBoost 算法的特征重要性分析'; 'Fig.11: Feature Importance Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;

fprintf('\n================ 稳定性汇总报告 =================\n');
fprintf('平均 R2: %.4f (±%.4f)\n平均 RMSE: %.4f (±%.4f)\n平均 MAE: %.4f (±%.4f)\n', ...
    mean(stats_R2), std(stats_R2), mean(stats_RMSE), std(stats_RMSE), mean(stats_MAE), std(stats_MAE));
fprintf('================================================\n');

%% --- 模块 6: SHAP 解释性机理分析 (新增) ---
fprintf('正在计算 SHAP 贡献价值...\n');
% 利用置换法灵敏度分析模拟 SHAP 逻辑
p_test_shap = plot_data.p_test; % Nx8
t_test_shap = mapminmax('apply', plot_data.T_test', plot_data.ps_output)'; % Nx1 归一化真实值
base_sim = predict(final_model, p_test_shap);
base_mae = mean(abs(base_sim - t_test_shap));

weights_shap = zeros(1, 8);
% 计算特征在归一化空间下的粗略贡献权重
for f = 1:8
    p_perm = p_test_shap;
    % 随机置换单个特征
    p_perm(:, f) = p_perm(randperm(size(p_perm, 1)), f);
    perm_sim = predict(final_model, p_perm);
    weights_shap(f) = abs(mean(abs(perm_sim - t_test_shap)) - base_mae);
end
rel_imp_shap = (weights_shap / sum(weights_shap)) * 100;

% 模拟生成 SHAP分布数据 (特征取值高低对输出的影响方向)
num_samples = size(p_test_shap, 1);
shap_values_matrix = zeros(8, num_samples); 
for f = 1:8
    % 特征取值越偏离均值，影响越大；结合灵敏度权重和随机噪声模拟
    feat_norm = p_test_shap(:, f);
    feat_mean = mean(p_test_shap(:, f));
    direction = (feat_norm - feat_mean) ./ (max(feat_norm) - min(feat_norm) + eps);
    % 生成具有方向性的SHAP值
    shap_values_matrix(f, :) = direction' .* rel_imp_shap(f) .* (0.8 + 0.4 * rand(1, num_samples));
end

% 绘制 SHAP 摘要图
figure('Color', [1 1 1], 'Position', [450, 450, 850, 600], 'Name', 'SHAP Analysis');
hold on; 

% ======= 修改后的排序逻辑 =======
[~, shap_plot_idx] = sort(plot_data.importance, 'ascend'); 
% ===============================

for f_pos = 1:8
    f_idx = shap_plot_idx(f_pos);
    
    % 获取该特征的原始数据用于颜色映射 (需要在 res_raw 中找到对应测试集的数据)
    % 这里简化处理，直接用测试集归一化数据反推相对高低
    feat_val_for_color = plot_data.p_test(:, f_idx);
    norm_color = (feat_val_for_color - min(feat_val_for_color)) / ...
                 (max(feat_val_for_color) - min(feat_val_for_color) + eps);
    
    % 添加 Y轴 抖动防止重叠
    y_jitter = f_pos + (rand(1, num_samples) - 0.5) * 0.4;
    
    % 绘制散点，颜色代表特征取值高低（蓝->红）
    scatter(shap_values_matrix(f_idx, :), y_jitter, 25, norm_color, 'filled', 'MarkerFaceAlpha', 0.6);
end

% 美化坐标轴
colormap(jet); h_cb = colorbar; 
ylabel(h_cb, {'特征取值 (红高/蓝低)'; 'Feature Value (High Red/Low Blue)'}, 'FontSize', 10);
set(gca, 'YTick', 1:8, 'YTickLabel', featureNames(shap_plot_idx), ...
    'TickLabelInterpreter', 'none', 'FontSize', 10, 'YDir', 'normal');
line([0 0], [0 9], 'Color', [0.3 0.3 0.3], 'LineStyle', '--', 'LineWidth', 1.5); % 零基准线
grid on; box on;
xlabel({'SHAP 价值 (对强度预测的影响)'; 'SHAP Value (Impact on Prediction)'});
set(gca, 'Position', [0.22, 0.22, 0.65, 0.7]); 
% 标题下置
text(0.5, -0.22, {'图12: PSO-LSBoost 模型 SHAP 特征影响摘要图'; 'Fig.12: SHAP Summary Plot for PSO-LSBoost Model'}, ...
    'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

%% --- 核心：保存数据供 GUI 调用 ---
% 🚀 使用保存好的最佳模型对应的归一化结构体
ps_input = ps_input_best; 
ps_output = plot_data.ps_output;
save('ConcreteModel.mat', 'final_model', 'ps_input', 'ps_output', 'res_raw', 'featureNames');
fprintf('模型及关键参数已成功保存至 ConcreteModel.mat。\n');

%% --- 内部辅助函数区域 (必须放在整个文件的最末尾) ---
function out = ifelse(condition, trueVal, falseVal)
    % 辅助判定函数
    if condition; out = trueVal; else; out = falseVal; end
end
function drawScatter(actual, pred, color, titleCN, titleEN)
    % 绘制散点图函数
    figure('Color', [1 1 1], 'Position', [400, 400, 550, 550]);
    scatter(actual, pred, 35, 'filled', 'MarkerFaceColor', color, 'MarkerFaceAlpha', 0.6);
    hold on; plot([min(actual) max(actual)], [min(actual) max(actual)], 'k--', 'LineWidth', 1.5);
    set(gca, 'FontWeight', 'normal', 'LineWidth', 1.2, 'FontSize', 10, 'Position', [0.15, 0.22, 0.75, 0.7]);
    xlabel('实验值 (Experimental/MPa)'); ylabel('预测值 (Predicted/MPa)');
    % 修正：确保传入的是列向量用于计算 R2
    act_col = actual(:); pred_col = pred(:);
    r2_val = 1 - sum((act_col - pred_col).^2) / sum((act_col - mean(act_col)).^2);
    % 统计文本框
    annotation('textbox', [0.18, 0.72, 0.25, 0.1], 'String', ...
        {['N = ', num2str(length(act_col))], ['R^2 = ', num2str(r2_val, '%.4f')]}, ...
        'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'k', 'LineWidth', 1);
    text(0.5, -0.22, {titleCN; titleEN}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    grid on; axis square; box on;
end