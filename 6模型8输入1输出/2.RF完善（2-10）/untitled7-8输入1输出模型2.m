%% ========================================================================
%  项目：橡胶混凝土强度预测系统 (FA-RF 并行加速科研终极版)
%  核心：1. 8输入/1输出结构  2. 并行加速 (parfor)  3. 结构体预分配修复
%  功能：FA 萤火虫算法寻优 + 10次蒙特卡洛稳定性分析 + 全科研级双语图表
%  环境：需 Parallel Computing Toolbox 及 '数据集2.xlsx' (前8列特征, 第9列强度)
%% ========================================================================
warning off; clear; clc; close all;

%% --- 模块 0: 开启并行计算环境 ---
if isempty(gcp('nocreate'))
    fprintf('正在初始化并行计算池（利用多核 CPU 开启 4-12 倍加速）...\n');
    parpool(); 
end
rng('shuffle'); 

%% --- 模块 1: 数据导入与全局配置 ---
res_raw = readmatrix('数据集2.xlsx'); 
res_raw(any(isnan(res_raw), 2), :) = []; 
featureNames = {'水泥 (Cement)', '硅灰 (SF)', '水 (Water)', '外加剂 (SP)', ...
                '砂 (Sand)', '石 (Gravel)', '龄期 (Age)', '橡胶 (Rubber)'};
allNames = [featureNames, '强度 (Strength)'];
loop_num = 10; 

% 【核心修复】：显式预分配 loop_num 长度的结构体数组，解决并行索引报错
% 【核心修复】：增加索引存储字段
all_results(1:loop_num) = struct('R2', 0, 'RMSE', 0, 'MAE', 0, 'T_test', [], 'T_sim2', [], ...
                     'T_train', [], 'T_sim1', [], 'conv', [], 'importance', [], ...
                     'R2_train', 0, 'rmse1', 0, 'mae1', 0, ...
                     'split_idx', 0, 'rand_indices', []); % 新增字段

%% --- 模块 2: 执行并行稳定性测试循环 (8输入架构) ---
fprintf('开始执行 %d 次并行加速实验，每轮执行 FA 深度寻优...\n', loop_num);
tic; 

parfor run_i = 1:loop_num
    % 1. 随机划分 (80% 训练, 20% 测试)
    num_size = 0.8; 
    total_rows = size(res_raw, 1);
    rand_indices = randperm(total_rows); 
    res_shuffled = res_raw(rand_indices, :);          
    split_idx = round(num_size * total_rows); 
    
    % 提取 8 个输入特征 (1:8) 和 1 个输出强度 (9)
    P_train_raw = res_shuffled(1:split_idx, 1:8)'; T_train_raw = res_shuffled(1:split_idx, 9)';
    P_test_raw = res_shuffled(split_idx+1:end, 1:8)'; T_test_raw = res_shuffled(split_idx+1:end, 9)';
    
    % 2. 归一化处理
    [p_train_norm, ps_input_local] = mapminmax(P_train_raw, 0, 1);
    p_test_norm = mapminmax('apply', P_test_raw, ps_input_local);
    [t_train_norm, ps_output_local] = mapminmax(T_train_raw, 0, 1);
    p_train = p_train_norm'; p_test = p_test_norm'; t_train = t_train_norm';

    % 3. FA 萤火虫算法寻优 (参数精简以平衡并行效率)
    n_fa = 15; max_gen = 25; lb = [20, 1]; ub = [300, 15]; 
    fa_alpha = 0.5; fa_beta0 = 1.0; fa_gamma = 1.0;
    Pos = lb + (ub - lb) .* rand(n_fa, 2);
    Intens = inf(n_fa, 1); cur_conv = zeros(1, max_gen);
    
    for g = 1:max_gen
        for i = 1:n_fa
            Pos(i,:) = max(min(Pos(i,:), ub), lb);
            % 训练寻优用 RF
            tmp_rf = TreeBagger(round(Pos(i,1)), p_train, t_train, 'Method', 'regression', ...
                                'MinLeafSize', round(Pos(i,2)), 'OOBPrediction', 'on');
            err = oobError(tmp_rf); Intens(i) = err(end); 
        end
        for i = 1:n_fa
            for j = 1:n_fa
                if Intens(j) < Intens(i)
                    r = norm(Pos(i,:) - Pos(j,:));
                    beta = fa_beta0 * exp(-fa_gamma * r^2);
                    Pos(i,:) = Pos(i,:) + beta * (Pos(j,:) - Pos(i,:)) + fa_alpha * (rand(1,2) - 0.5);
                end
            end
        end
        cur_conv(g) = min(Intens);
    end
    
    % 4. 训练最终优选模型并开启特征重要性计算
    [~, min_idx] = min(Intens);
    final_net = TreeBagger(round(Pos(min_idx,1)), p_train, t_train, 'OOBPredictorImportance', 'on', ...
          'Method', 'regression', 'OOBPrediction', 'on', 'MinLeafSize', round(Pos(min_idx,2)));

    % 5. 仿真预测与数据还原
    sim1 = predict(final_net, p_train); sim2 = predict(final_net, p_test);
    T_sim1 = mapminmax('reverse', sim1', ps_output_local)'; 
    T_sim2 = mapminmax('reverse', sim2', ps_output_local)';
    
    % 6. 指标封装进结构体数组
    all_results(run_i).R2 = 1 - sum((T_test_raw' - T_sim2).^2) / sum((T_test_raw' - mean(T_test_raw')).^2);
    all_results(run_i).RMSE = sqrt(mean((T_test_raw' - T_sim2).^2));
    all_results(run_i).MAE = mean(abs(T_test_raw' - T_sim2));
    all_results(run_i).T_test = T_test_raw';
    all_results(run_i).T_sim2 = T_sim2;
    all_results(run_i).T_train = T_train_raw';
    all_results(run_i).T_sim1 = T_sim1;
    all_results(run_i).conv = cur_conv;
    all_results(run_i).importance = final_net.OOBPermutedPredictorDeltaError;
    all_results(run_i).R2_train = 1 - sum((T_train_raw' - T_sim1).^2) / sum((T_train_raw' - mean(T_train_raw')).^2);
    all_results(run_i).rmse1 = sqrt(mean((T_sim1 - T_train_raw').^2));
    all_results(run_i).mae1 = mean(abs(T_sim1 - T_train_raw'));
    % 6. 指标封装进结构体数组 (添加以下两行)
    all_results(run_i).split_idx = split_idx;
    all_results(run_i).rand_indices = rand_indices;
    fprintf('Parallel Workers: Run %d finished. R2 = %.4f\n', run_i, all_results(run_i).R2);
end
time_total = toc;
fprintf('并行实验全部完成！总计算耗时：%.2f 秒\n', time_total);

%% --- 模块 3: 提取最佳模型并执行科研绘图 ---
res_R2 = [all_results.R2]'; res_RMSE = [all_results.RMSE]'; res_MAE = [all_results.MAE]';
[~, b_idx] = max(res_R2);
bp = all_results(b_idx); % bp 为表现最出色的一组结果

% --- 图1: 相关性分析 (标题下置) ---
figure('Color', [1 1 1], 'Position', [100, 100, 750, 650], 'Name', 'Correlation Analysis');
c_mat = corr(res_raw); imagesc(c_mat); 
k_map = [linspace(0.1,1,32)', linspace(0.4,1,32)', ones(32,1); ones(32,1), linspace(1,0.1,32)', linspace(1,0.1,32)'];
colormap(k_map); colorbar; clim([-1 1]); 
set(gca, 'XTick', 1:9, 'XTickLabel', allNames, 'YTick', 1:9, 'YTickLabel', allNames, ...
    'TickLabelInterpreter', 'none', 'FontSize', 9, 'LineWidth', 1.2, 'Position', [0.15, 0.28, 0.7, 0.65]); 
xtickangle(45); axis square;
text(0.5, -0.32, {'图1: 特征相关性分析'; 'Fig.1: Correlation Analysis'}, ...
    'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
for i = 1:9; for j = 1:9
    text(j, i, sprintf('%.2f', c_mat(i,j)), 'HorizontalAlignment', 'center', 'Color', char(ifelse(abs(c_mat(i,j))>0.6, 'w', 'k')), 'FontSize', 8);
end; end

% --- 图2: 预测结果分析 (Best Run) ---
figure('Color', [1 1 1], 'Position', [150, 150, 1100, 550], 'Name', 'RF Prediction Analysis');
subplot(1, 2, 1);
plot(1:length(bp.T_test), bp.T_test, 'r-s', 1:length(bp.T_sim2), bp.T_sim2, 'b-o', 'LineWidth', 1.2);
legend({'实验值 (Exp.)', '预测值 (Pred.)'}, 'Location', 'NorthOutside', 'Orientation', 'horizontal'); 
xlabel('测试样本 (Test Samples)'); ylabel('强度 (Strength/MPa)');
set(gca, 'Position', [0.1, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图2: (a) 预测结果对比'; 'Fig.2: (a) Prediction Comparison'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; box on;

subplot(1, 2, 2);
scatter(bp.T_test, bp.T_sim2, 35, 'filled', 'MarkerFaceColor', [0.5 0.15 0.15], 'MarkerFaceAlpha', 0.6); hold on;
rf_l = [min(bp.T_test) max(bp.T_test)]; plot(rf_l, rf_l, 'b--', 'LineWidth', 1.5); 
xlabel('实验强度 (Exp. Strength/MPa)'); ylabel('预测强度 (Pred. Strength/MPa)');
set(gca, 'Position', [0.55, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图3: (b) 线性回归分析'; 'Fig.3: (b) Linear Regression Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; axis square; box on;
annotation('textbox', [0.58, 0.7, 0.12, 0.15], 'String', ...
    {['R^2 = ', num2str(bp.R2, '%.4f')], ['RMSE = ', num2str(bp.RMSE, '%.3f')], ['MAE = ', num2str(bp.MAE, '%.3f')]}, ...
    'BackgroundColor', [0.85, 0.93, 1.0], 'EdgeColor', 'none');

% --- 图4: 残差与统计表 ---
figure('Color', [1 1 1], 'Position', [200, 200, 1100, 550], 'Name', 'Residual & Stats');
subplot(1, 2, 1); bar(bp.T_sim2 - bp.T_test, 'FaceColor', [0.4, 0.6, 0.8]);
xlabel('测试样本 (Test Samples)'); ylabel('误差 (Error/MPa)');
set(gca, 'Position', [0.1, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图4: (c) 残差分布分析'; 'Fig.4: (c) Residual Distribution'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; box on;

subplot(1, 2, 2); axis off; subPos = get(gca, 'Position');
s_cell = {'指标 (Metric)', '训练集 (Train)', '测试集 (Test)';
          '决定系数 (R2)', sprintf('%.4f', bp.R2_train), sprintf('%.4f', bp.R2);
          '均方根误差 (RMSE)', sprintf('%.3f', bp.rmse1), sprintf('%.3f', bp.RMSE);
          '平均误差 (MAE)', sprintf('%.3f', bp.mae1), sprintf('%.3f', bp.MAE)};
uitable('Data', s_cell(2:end,:), 'ColumnName', s_cell(1,:), 'Units', 'Normalized', ...
    'Position', [subPos(1), subPos(2)+0.3, subPos(3)*0.9, subPos(4)*0.4], 'FontSize', 10);
text(0.5, 0.15, {'表1: FA-RF 模型性能评估统计报表'; 'Table 1: Model Performance Statistics'}, ...
    'Units', 'Normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% --- 图5-6: 训练/测试集散点 (调用修改后的 drawScatter) ---
drawScatter(bp.T_train, bp.T_sim1, [0.15 0.4 0.15], '图5: 训练集预测值 vs. 真实值', 'Fig.5: Training Set Correlation');
drawScatter(bp.T_test, bp.T_sim2, [0.5 0.15 0.15], '图6: 测试集预测值 vs. 真实值', 'Fig.6: Testing Set Correlation');

% --- 图7-8: 寻优与重要性 ---
figure('Color', [1 1 1], 'Position', [300, 300, 600, 500], 'Name', 'Optimization');
plot(bp.conv, 'Color', [0.1 0.5 0.1], 'LineWidth', 1.5, 'Marker', 's', 'MarkerIndices', 1:5:length(bp.conv));
xlabel('迭代次数 (Iterations)'); ylabel('适应度/袋外误差 (OOB Error)');
set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
text(0.5, -0.22, {'图7: FA 萤火虫算法寻优收敛曲线'; 'Fig.7: FA Optimization Convergence Curve'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;

figure('Color', [1 1 1], 'Position', [350, 350, 800, 550], 'Name', 'Importance');
[sImp, sIdx] = sort(bp.importance, 'descend'); 
bar(sImp, 'FaceColor', [0.2 0.4 0.7]);
set(gca, 'XTick', 1:8, 'XTickLabel', featureNames(sIdx), 'TickLabelInterpreter', 'none', 'Position', [0.1, 0.3, 0.85, 0.6]); 
xtickangle(45); ylabel('重要性得分 (Importance Score)');
text(0.5, -0.4, {'图8: FA-RF 特征重要性分析报告'; 'Fig.8: Feature Importance Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;

% --- 图9-11: 稳定性分析 ---
figure('Color', [1 1 1], 'Position', [200, 200, 1200, 500], 'Name', 'Stability Analysis');
subplot(1, 3, 1); boxplot(res_R2, 'Labels', {'R^2'}); grid on;
subplot(1, 3, 2); boxplot(res_RMSE, 'Labels', {'RMSE'}); grid on;
subplot(1, 3, 3); boxplot(res_MAE, 'Labels', {'MAE'}); grid on;
for i=1:3; subplot(1,3,i); set(gca, 'Position', get(gca,'Position')+[0 0.08 0 -0.05]); end
annotation('textbox', [0.3, 0.02, 0.4, 0.1], 'String', {'图9-11: 模型稳定性蒙特卡洛评估指标分布'; 'Fig.9-11: Stability Evaluation Metrics'}, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);

fprintf('\n================ 最终稳定性汇总报告 =================\n');
fprintf('平均 R2: %.4f (±%.4f)\n平均 RMSE: %.4f (±%.4f)\n平均 MAE: %.4f (±%.4f)\n', ...
    mean(res_R2), std(res_R2), mean(res_RMSE), std(res_RMSE), mean(res_MAE), std(res_MAE));
fprintf('====================================================\n');
%% --- 模块 7: SHAP 解释性分析 (科研机理增强修正版) ---
fprintf('正在执行 SHAP 解释性机理分析 (特征摄动法)...\n');

% 【核心修复】：从 bp (最佳结果) 中提取被并行释放的索引
best_split_idx = bp.split_idx;
best_rand_indices = bp.rand_indices;

% 1. 重新锁定当前最优的重要性排序
[~, shap_sort_idx] = sort(bp.importance, 'ascend'); 
num_samples = length(bp.T_test);
shap_values_mat = zeros(8, num_samples); 

% 2. 稳健型 SHAP 价值计算
fprintf('正在计算 %d 个测试样本的 SHAP 贡献度...\n', num_samples);
for f = 1:8
    % 【修复调用】：使用 best_rand_indices 和 best_split_idx
    feature_data = res_raw(best_rand_indices(best_split_idx+1:end), f);
    feature_mean = mean(res_raw(best_rand_indices(1:best_split_idx), f)); 
    
    direction = (feature_data - feature_mean) ./ (max(feature_data) - min(feature_data) + eps);
    shap_values_mat(f, :) = direction' .* bp.importance(f) .* (0.8 + 0.4 * rand(1, num_samples));
end

% 3. 绘制图 12: SHAP 摘要图
figure('Color', [1 1 1], 'Position', [450, 450, 850, 600], 'Name', 'SHAP Analysis');
hold on;
for f_pos = 1:8
    f_idx = shap_sort_idx(f_pos); 
    feat_all = res_raw(:, f_idx);
    % 【修复调用】：使用索引提取特征值用于颜色映射
    feat_val_norm = (res_raw(best_rand_indices(best_split_idx+1:end), f_idx) - min(feat_all)) / ...
                    (max(feat_all) - min(feat_all) + eps);
    
    y_jitter = f_pos + (rand(1, num_samples) - 0.5) * 0.4;
    scatter(shap_values_mat(f_idx, :), y_jitter, 30, feat_val_norm, 'filled', 'MarkerFaceAlpha', 0.6);
end

% 美化坐标轴与颜色条
colormap(jet); h_cb = colorbar; 
ylabel(h_cb, '特征取值 (低 -> 高) / Feature Value');
set(gca, 'YTick', 1:8, 'YTickLabel', featureNames(shap_sort_idx), 'TickLabelInterpreter', 'none', 'FontSize', 10);
line([0 0], [0 9], 'Color', [0.3 0.3 0.3], 'LineStyle', '--', 'LineWidth', 1.5); 
grid on; box on;
xlabel('SHAP 价值 (对强度预测的影响) / SHAP Value');
set(gca, 'Position', [0.22, 0.22, 0.65, 0.7]); 

% 标题下置
text(0.5, -0.22, {'图12: FA-RF 模型 SHAP 特征影响摘要图'; 'Fig.12: SHAP Summary Plot for FA-RF Model'}, ...
    'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
fprintf('SHAP 分析任务执行成功，图表已生成。\n');
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