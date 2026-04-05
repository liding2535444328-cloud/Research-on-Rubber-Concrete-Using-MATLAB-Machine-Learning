%% ========================================================================
%  项目：ConcreteModel_LSBoost.mat 数据全维度可视化工具
%  功能：一键重现所有科研图表（预测对比、回归分析、收敛曲线、重要性分析等）
%% ========================================================================
clear; clc; close all;

file_name = 'ConcreteModel_LSBoost.mat';
if ~exist(file_name, 'file')
    error('未找到文件 %s', file_name);
end

% --- 1. 加载数据 ---
fprintf('正在提取 %s 中的关键数据...\n', file_name);
load(file_name); % 加载后工作区会出现 bp, res_raw, featureNames

%% --- 2. 核心图表重现 ---

% --- 图 A: 测试集预测结果对比图 ---
figure('Color', [1 1 1], 'Name', 'Prediction Comparison');
plot(bp.T_test, 'r-s', 'LineWidth', 1.2, 'MarkerFaceColor', 'r'); hold on;
plot(bp.T_sim2, 'b-o', 'LineWidth', 1.2, 'MarkerFaceColor', 'b');
grid on; ylabel('抗压强度 (MPa)'); xlabel('测试样本编号');
legend({'实验值 (Exp.)', '预测值 (Pred.)'}, 'Location', 'NorthOutside', 'Orientation', 'horizontal');
title('测试集：实验值与预测值对比分布');

% --- 图 B: 线性回归分析 (实验 vs 预测) ---
figure('Color', [1 1 1], 'Name', 'Regression Analysis');
scatter(bp.T_test, bp.T_sim2, 50, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
ref_line = [min([bp.T_test; bp.T_sim2]), max([bp.T_test; bp.T_sim2])];
plot(ref_line, ref_line, 'k--', 'LineWidth', 2);
grid on; axis square;
xlabel('实验强度 (Experimental MPa)'); ylabel('预测强度 (Predicted MPa)');
title(['线性回归分析 (R^2 = ', num2str(bp.R2, '%.4f'), ')']);
annotation('textbox', [0.15, 0.7, 0.2, 0.15], 'String', ...
    {['RMSE: ', num2str(bp.RMSE, '%.3f')], ['MAE: ', num2str(bp.MAE, '%.3f')]}, ...
    'BackgroundColor', 'w', 'EdgeColor', 'k');

% --- 图 C: PSO 参数寻优收敛曲线 ---
if isfield(bp, 'conv')
    figure('Color', [1 1 1], 'Name', 'PSO Convergence');
    plot(bp.conv, 'Color', [0.1, 0.5, 0.1], 'LineWidth', 2, 'Marker', 'd', 'MarkerIndices', 1:5:length(bp.conv));
    grid on; xlabel('进化迭代代数'); ylabel('适应度值 (MSE)');
    title('PSO 深度参数寻优收敛轨迹');
end

% --- 图 D: 9 维特征重要性分析 (特征贡献度) ---
if isfield(bp, 'importance')
    [sorted_imp, idx] = sort(bp.importance, 'ascend');
    figure('Color', [1 1 1], 'Name', 'Feature Importance');
    barh(sorted_imp/sum(sorted_imp)*100, 'FaceColor', [0.2 0.4 0.7], 'EdgeColor', 'k');
    set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(idx));
    grid on; xlabel('相对贡献度 (%)');
    title('基于 LSBoost 权重的特征贡献度排序');
end

% --- 图 E: 原始数据集分布热力图 (相关性) ---
figure('Color', [1 1 1], 'Name', 'Correlation Heatmap');
allNames = [featureNames, {'强度'}];
corrMat = corr(res_raw);
imagesc(corrMat); colormap(jet); colorbar; clim([-1 1]);
set(gca, 'XTick', 1:10, 'XTickLabel', allNames, 'YTick', 1:10, 'YTickLabel', allNames);
xtickangle(45); axis square;
title('原始数据集特征相关性热力图');

fprintf('✅ 可视化完成！已为您生成 5 张核心分析图表。\n');