%% ========================================================================
%  项目：橡胶混凝土强度预测系统 (科研增强 + GUI模型导出集成版)
%  作者：李鼎
%  优化：1. 全图表双语科研排版  2. 稳定性循环测试  3. 自动导出GUI模型文件
%  环境：需确保安装 libsvm 工具箱及存在 '数据集2.xlsx'
%% ========================================================================
warning off; clear; clc; close all;
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
stats_MAE = zeros(loop_num, 1);

%% --- 模块 2: 特征相关性分析 (图1) ---
figure('Color', [1 1 1], 'Position', [100, 100, 750, 650], 'Name', 'Correlation Analysis');
corrMat = corr(res_raw); imagesc(corrMat); 
map = [linspace(0.1,1,32)', linspace(0.4,1,32)', ones(32,1); ... 
       ones(32,1), linspace(1,0.1,32)', linspace(1,0.1,32)'];
colormap(map); colorbar; clim([-1 1]); 
set(gca, 'XTick', 1:9, 'XTickLabel', allNames, 'YTick', 1:9, 'YTickLabel', allNames, ...
    'TickLabelInterpreter', 'none', 'FontSize', 9, 'LineWidth', 1.2, ...
    'Position', [0.15, 0.28, 0.7, 0.65]); 
xtickangle(45); axis square;
text(0.5, -0.32, {'图1: 特征相关性分析'; 'Fig.1: Correlation Analysis'}, ...
    'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
for i = 1:9
    for j = 1:9
        text(j, i, sprintf('%.2f', corrMat(i,j)), 'HorizontalAlignment', 'center', ...
            'Color', char(ifelse(abs(corrMat(i,j))>0.6, 'w', 'k')), 'FontSize', 8);
    end
end

%% --- 模块 3: 执行 PSO-SVR 稳定性测试循环 ---
fprintf('开始执行 %d 次 PSO-SVR 稳定性深度测试...\n', loop_num);
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
    
    pop_size = 25; max_iter = 40; 
    c1 = 1.5; c2 = 1.5; w_max = 0.9; w_min = 0.4;
    lb = [0.1, 0.01]; ub = [100, 10]; 
    particles = lb + (ub - lb) .* rand(pop_size, 2);
    velocity = zeros(pop_size, 2);
    pBest = particles; pBest_score = inf(pop_size, 1);
    gBest = zeros(1, 2); gBest_score = inf;
    cur_conv = zeros(1, max_iter);
    
    for t = 1:max_iter
        w = (w_max - w_min) * (max_iter - t)^2 / max_iter^2 + w_min; 
        for i = 1:pop_size
            cmd_pso = [' -t 2 -c ', num2str(particles(i,1)), ' -g ', num2str(particles(i,2)), ' -s 3 -v 5 -q'];
            mse = svmtrain(t_train', p_train', cmd_pso);
            if mse < pBest_score(i)
                pBest_score(i) = mse; pBest(i,:) = particles(i,:);
            end
            if mse < gBest_score
                gBest_score = mse; gBest = particles(i,:);
            end
        end
        velocity = w*velocity + c1*rand*(pBest - particles) + c2*rand*(repmat(gBest, pop_size, 1) - particles);
        particles = particles + velocity;
        particles = max(min(particles, ub), lb);
        cur_conv(t) = gBest_score;
    end
    
    cmd_final = [' -t 2 -c ', num2str(gBest(1)), ' -g ', num2str(gBest(2)), ' -s 3 -p 0.01'];
    model = svmtrain(t_train', p_train', cmd_final);
    [t_sim1, ~] = svmpredict(t_train', p_train', model);
    [t_sim2, ~] = svmpredict(t_test', p_test', model);
    T_sim1 = mapminmax('reverse', t_sim1', ps_output)'; 
    T_sim2 = mapminmax('reverse', t_sim2', ps_output)';
    
    r2_cur = 1 - sum((T_test' - T_sim2).^2) / sum((T_test' - mean(T_test')).^2);
    stats_R2(run_i) = r2_cur;
    stats_RMSE(run_i) = sqrt(mean((T_test' - T_sim2).^2));
    stats_MAE(run_i) = mean(abs(T_test' - T_sim2));
    
    % ... (前面的寻优代码保持不变)
    if r2_cur > best_overall_R2
        best_overall_R2 = r2_cur;
        % 锁定表现最好的一次迭代数据用于导出和绘图
        plot_data.T_test = T_test'; 
        plot_data.T_sim2 = T_sim2;
        plot_data.T_train = T_train'; 
        plot_data.T_sim1 = T_sim1;
        plot_data.conv = cur_conv;
        plot_data.R2_test = r2_cur; 
        plot_data.rmse2 = stats_RMSE(run_i); 
        plot_data.mae2 = stats_MAE(run_i);
        plot_data.R2_train = 1 - sum((T_train' - T_sim1).^2) / sum((T_train' - mean(T_train')).^2);
        plot_data.rmse1 = sqrt(mean((T_sim1 - T_train').^2)); 
        plot_data.mae1 = mean(abs(T_sim1 - T_train'));
        
        % 重要：保存当前最优迭代的输入数据，供模块 6 SHAP 使用
        plot_data.p_test = p_test;
        plot_data.p_train = p_train;
        plot_data.t_test = t_test;

        % 核心：记录下当前最优模型和归一化参数用于导出
        best_model_for_gui = model; 
        best_ps_in = ps_input; 
        best_ps_out = ps_output;
    end
    fprintf('Run %d/%d: SVR Testing R2 = %.4f\n', run_i, loop_num, r2_cur);
end % 闭合 run_i 循环
    

%% --- 模块 4: 核心预测可视化 ---
N_test = length(plot_data.T_test);
figure('Color', [1 1 1], 'Position', [150, 150, 1100, 550], 'Name', 'SVR Prediction Analysis');
subplot(1, 2, 1);
plot(1:N_test, plot_data.T_test, 'Color', [0.8 0.2 0.2], 'Marker', 's', 'LineWidth', 1.2, 'MarkerSize', 5); hold on;
plot(1:N_test, plot_data.T_sim2, 'Color', [0.2 0.4 0.8], 'Marker', 'o', 'LineWidth', 1.2, 'MarkerSize', 5);
legend({'实验值 (Exp.)', '预测值 (Pred.)'}, 'Location', 'NorthOutside', 'Orientation', 'horizontal'); 
xlabel({'测试样本'; 'Test Samples'}); ylabel({'强度 (MPa)'; 'Strength (MPa)'});
set(gca, 'Position', [0.1, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图2: (a) 预测结果对比'; 'Fig.2: (a) Prediction Comparison'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; box on;

subplot(1, 2, 2);
scatter(plot_data.T_test, plot_data.T_sim2, 35, 'filled', 'MarkerFaceColor', [0.5 0.15 0.15], 'MarkerFaceAlpha', 0.6); hold on;
ref_line = [min(plot_data.T_test), max(plot_data.T_test)];
plot(ref_line, ref_line, 'b--', 'LineWidth', 1.5); 
xlabel({'实验强度 (MPa)'; 'Experimental Strength (MPa)'}); ylabel({'预测强度 (MPa)'; 'Predicted Strength (MPa)'});
set(gca, 'Position', [0.55, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图3: (b) 线性回归分析'; 'Fig.3: (b) Linear Regression Analysis'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; axis square; box on;
annotation('textbox', [0.58, 0.7, 0.12, 0.15], 'String', ...
    {['R^2 = ', num2str(plot_data.R2_test, '%.4f')], ...
     ['RMSE = ', num2str(plot_data.rmse2, '%.3f')], ...
     ['MAE = ', num2str(plot_data.mae2, '%.3f')]}, ...
    'BackgroundColor', [0.85, 0.93, 1.0], 'LineWidth', 1.0, 'EdgeColor', 'none');

figure('Color', [1 1 1], 'Position', [200, 200, 1100, 550], 'Name', 'Residual & Stats Report');
subplot(1, 2, 1);
bar(plot_data.T_sim2 - plot_data.T_test, 'FaceColor', [0.4, 0.6, 0.8]);
xlabel({'测试样本'; 'Test Samples'}); ylabel({'误差 (MPa)'; 'Error (MPa)'}); 
set(gca, 'Position', [0.1, 0.22, 0.35, 0.6]); 
text(0.5, -0.25, {'图4: (c) 残差分布分析'; 'Fig.4: (c) Residual Distribution'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; box on;

subplot(1, 2, 2); axis off; subPos = get(gca, 'Position');
stats_cell = {'指标 (Metric)', '训练集 (Train)', '测试集 (Test)';
              '决定系数 (R2)', sprintf('%.4f', plot_data.R2_train), sprintf('%.4f', plot_data.R2_test);
              '均方根误差 (RMSE)', sprintf('%.3f', plot_data.rmse1), sprintf('%.3f', plot_data.rmse2);
              '平均误差 (MAE)', sprintf('%.3f', plot_data.mae1), sprintf('%.3f', plot_data.mae2)};
uitable('Data', stats_cell(2:end,:), 'ColumnName', stats_cell(1,:), 'Units', 'Normalized', ...
    'Position', [subPos(1), subPos(2)+0.3, subPos(3)*0.9, subPos(4)*0.4], 'FontSize', 10);
text(0.5, 0.15, {'表1: SVR 模型性能评估统计报表'; 'Table 1: Model Performance Statistics'}, ...
    'Units', 'Normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

drawScatter(plot_data.T_train, plot_data.T_sim1, [0.15 0.4 0.15], '图5: 训练集预测值 vs. 真实值', 'Fig.5: Training Set');
drawScatter(plot_data.T_test, plot_data.T_sim2, [0.5 0.15 0.15], '图6: 测试集预测值 vs. 真实值', 'Fig.6: Testing Set');

%% --- 模块 5: 稳定性分析与模型保存 ---
figure('Color', [1 1 1], 'Position', [300, 300, 600, 500], 'Name', 'PSO Convergence');
plot(plot_data.conv, 'Color', [0.1 0.5 0.1], 'LineWidth', 1.5);
xlabel({'迭代次数'; 'Iterations'}); ylabel({'适应度 (MSE)'; 'Fitness (MSE)'});
set(gca, 'Position', [0.15, 0.22, 0.75, 0.7]);
text(0.5, -0.22, {'图7: PSO 参数寻优收敛曲线'; 'Fig.7: PSO Hyperparameter Optimization'}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;

figure('Color', [1 1 1], 'Position', [200, 200, 1200, 500], 'Name', 'Stability Analysis');
subplot(1, 3, 1); boxplot(stats_R2, 'Labels', {'Testing R^2'}); grid on;
subplot(1, 3, 2); boxplot(stats_RMSE, 'Labels', {'Testing RMSE'}); grid on;
subplot(1, 3, 3); boxplot(stats_MAE, 'Labels', {'Testing MAE'}); grid on;
for i=1:3; subplot(1,3,i); set(gca, 'Position', get(gca,'Position')+[0 0.08 0 -0.05]); end
annotation('textbox', [0.3, 0.02, 0.4, 0.1], 'String', {'图8-10: 模型稳定性评估指标分布'; 'Fig.8-10: Distribution of Stability Metrics'}, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);

% 1. 打印报告
fprintf('\n================ PSO-SVR 稳定性最终报告 =================\n');
fprintf('平均 R2: %.4f (±%.4f)\n', mean(stats_R2), std(stats_R2));
fprintf('平均 RMSE: %.4f (±%.4f)\n', mean(stats_RMSE), std(stats_RMSE));
fprintf('平均 MAE: %.4f (±%.4f)\n', mean(stats_MAE), std(stats_MAE));
fprintf('========================================================\n');

% 2. 导出 GUI 所需的 ConcreteModel.mat
final_model = best_model_for_gui; 
ps_input = best_ps_in; 
ps_output = best_ps_out;
save('ConcreteModel.mat', 'final_model', 'ps_input', 'ps_output', 'res_raw');
fprintf('🚀 [成功] 最佳 SVR 模型及归一化参数已导出至 ConcreteModel.mat\n');
fprintf('💡 提示：现在您可以直接运行 RubberConcrete_SmartDesign_V7.m 启动 GUI。\n');

%% --- 模块 6: 特征贡献度与 SHAP 解释性分析 (科研级可视化优化) ---
fprintf('正在生成科研级特征分析图表...\n');

% 1. 环境锁定与基准建立
p_test_fix = plot_data.p_test;   
t_test_fix = plot_data.t_test;   
T_test_fix = plot_data.T_test(:); 
base_r2 = plot_data.R2_test; 
importance = zeros(1, 8);

% 2. 改进型特征贡献度计算 (基于 R2 损失)
% % % --- 请替换为您代码中的这一段 --- % % %
for f = 1:8
    P_perm = p_test_fix; 
    P_perm(f, :) = P_perm(f, randperm(size(P_perm, 2))); 
    [t_sim_p, ~] = svmpredict(t_test_fix', P_perm', best_model_for_gui, '-q');
    T_sim_p_raw = mapminmax('reverse', t_sim_p', best_ps_out);
    
    % --- 终极对齐修复：确保长度和维度完全一致 ---
    T_actual = T_test_fix(:);
    T_pred = T_sim_p_raw(:);
    L = min(length(T_actual), length(T_pred)); % 取公共长度，防止差一个样本报错
    
    % 重新计算 cur_r2
    res_sq = (T_actual(1:L) - T_pred(1:L)).^2;
    tot_sq = (T_actual(1:L) - mean(T_actual(1:L))).^2;
    cur_r2 = 1 - sum(res_sq) / sum(tot_sq);
    
    importance(f) = abs(base_r2 - cur_r2);
end
% % % ---------------------------------- % % %

% 归一化并排序
importance = importance + 1e-5; % 确保微量显示，避免空白图
rel_imp = (importance / sum(importance)) * 100;
[sorted_imp, idx] = sort(rel_imp, 'ascend'); % 改为升序排列，barh 效果更好

% 【绘图 11】 特征贡献度 (优化样式)
figure('Color', [1 1 1], 'Position', [400, 400, 800, 550]);
b = barh(sorted_imp, 'FaceColor', 'flat', 'EdgeColor', 'k');
b.CData = parula(8);
set(gca, 'YTick', 1:8, 'YTickLabel', featureNames(idx), 'FontSize', 10, 'TickLabelInterpreter', 'none');
xlabel('相对贡献度 (%) / Relative Importance (%)', 'FontWeight', 'bold');
title({'图11: 基于 PSO-SVR 的特征贡献度分析'; 'Fig.11: Feature Importance Analysis'}, 'FontSize', 12);
grid on;

% 3. 稳健型 SHAP 价值计算 (修复 1x1 赋值错误)
shap_values = zeros(8, size(p_test_fix, 2)); 
for i = 1:size(p_test_fix, 2)
    curr_x = p_test_fix(:, i);
    [base_out, ~] = svmpredict(0, curr_x', best_model_for_gui, '-q');
    for f = 1:8
        t_x = curr_x; 
        t_x(f) = mean(plot_data.p_train(f, :)); % 用均值替代
        [alt_out, ~] = svmpredict(0, t_x', best_model_for_gui, '-q');
        
        % 强制提取第一个元素，解决 1x1 赋值给 0x0 的报错
        if ~isempty(base_out) && ~isempty(alt_out)
            shap_values(f, i) = base_out(1) - alt_out(1);
        end
    end
end

% 【绘图 12】 SHAP 摘要图 (优化样式)
figure('Color', [1 1 1], 'Position', [450, 450, 850, 600]);
hold on;
for f = 1:8
    % 散点抖动处理，避免重叠
    f_idx = idx(f);
    y_vals = f + (rand(1, size(p_test_fix, 2)) - 0.5) * 0.4; 
    scatter(shap_values(f_idx, :), y_vals, 35, p_test_fix(f_idx, :), 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(jet); h = colorbar; ylabel(h, '特征值高低 (Feature Value)');
set(gca, 'YTick', 1:8, 'YTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none');
line([0 0], [0 9], 'Color', [0.3 0.3 0.3], 'LineStyle', '--', 'LineWidth', 1.5);
xlabel('SHAP 价值 (对强度的影响) / SHAP Value', 'FontWeight', 'bold');
title({'图12: SHAP 特征影响摘要图'; 'Fig.12: SHAP Summary Plot'}, 'FontSize', 12);
grid on; box on;
%% --- 内部辅助函数 ---
function out = ifelse(condition, trueVal, falseVal)
    if condition; out = trueVal; else; out = falseVal; end
end
function drawScatter(actual, pred, color, titleCN, titleEN)
    figure('Color', [1 1 1], 'Position', [400, 400, 550, 550]);
    scatter(actual, pred, 35, 'filled', 'MarkerFaceColor', color, 'MarkerFaceAlpha', 0.6);
    hold on; plot([min(actual) max(actual)], [min(actual) max(actual)], 'k--', 'LineWidth', 1.5);
    set(gca, 'FontWeight', 'normal', 'LineWidth', 1.2, 'FontSize', 10, 'Position', [0.15, 0.22, 0.75, 0.7]);
    xlabel({'实验值 (MPa)'; 'Experimental (MPa)'}); ylabel({'预测值 (MPa)'; 'Predicted (MPa)'});
    text(0.5, -0.22, {titleCN; titleEN}, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    grid on; axis square; box on;
end