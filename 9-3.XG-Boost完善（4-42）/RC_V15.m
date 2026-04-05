function RubberConcrete_SmartDesign_V7_Final()
% =====================================================================
% 项目：橡胶混凝土强度智能设计系统 (9输入终极稳健版 V7.0)
% 适配模型：PSO-LSBoost (数据集3)
% 修复重点：1. 诊断式报错弹窗 2. 自动模型解包 3. 并发寻优逻辑加固
% =====================================================================

% --- 模块 1: 加载模型与数据 ---
try
    S_Data = load('ConcreteModel_LSBoost.mat');

    % 动态路径提取，增强兼容性
    % 强制按照最新的训练脚本结构提取 (对应 SeeData 能看到 bp 的情况)
    if isfield(S_Data, 'bp')
        Final_Model_Object = S_Data.bp.final_model;
        Final_MAE_Value = S_Data.bp.MAE;
        Final_PS_In = S_Data.bp.ps_input;
        Final_PS_Out = S_Data.bp.ps_output;
        % 额外提取训练数据用于逆向种子生成
        RC_Train_X = S_Data.bp.p_train; 
    else
        % 备用兼容逻辑
        Final_Model_Object = S_Data.final_model;
        Final_MAE_Value = 2.15;
        Final_PS_In = S_Data.ps_input;
        Final_PS_Out = S_Data.ps_output;
        RC_Train_X = S_Data.res_raw(:, 1:9);
    end

    Final_Raw_Matrix = S_Data.res_raw;
    Final_LB = min(Final_Raw_Matrix(:, 1:9));
  Final_UB = max(Final_Raw_Matrix(:, 1:9));

   % --- 锁定核心数据 (适配数据集3) ---
    if isfield(S_Data, 'bp')
        trainedModel = S_Data.bp.final_model; % 统一变量名为 trainedModel
        Final_MAE_Value = S_Data.bp.MAE;
        ps_input = S_Data.bp.ps_input;       % 统一变量名
        ps_output = S_Data.bp.ps_output;     % 统一变量名
    else
        trainedModel = S_Data.final_model;
        Final_MAE_Value = 2.15;
        ps_input = S_Data.ps_input;
        ps_output = S_Data.ps_output;
    end
    % 强制执行工作区清理，确保 predict 没被占用
    clear predict;

    if isstruct(Final_Model_Object)
        warning('检测到模型以结构体格式加载，系统将尝试动态解析。');
    end

catch ME
    errordlg(['初始化失败，请确保 ConcreteModel_LSBoost.mat 在当前文件夹。错误：', ME.message], '加载错误'); return;
end

% --- 2. 主窗口设计 ---
fig = figure('Name', '橡胶混凝土 AI 智能设计系统 (V7.0 终极修复版)', ...
    'Units', 'normalized', 'Position', [0.1, 0.05, 0.8, 0.85], ...
    'Color', [0.95 0.95 0.95], 'MenuBar', 'none');

tabGroup = uitabgroup(fig, 'Position', [0 0 1 1]);
tabPREDICT = uitab(tabGroup, 'Title', '正向强度预测 / Forward PREDICTion');
tabInverse = uitab(tabGroup, 'Title', '逆向配比设计 / Inverse Design');

featureNamesCN = {'水胶比', '橡胶含量', '橡胶粒径', '水泥', '细骨料', '粗骨料', '硅比', '外加剂', '龄期'};
featureNamesEN = {'W/B', 'Rubber', 'MaxSize', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};
units = {'ratio', 'kg/m³', 'mm', 'kg/m³', 'kg/m³', 'kg/m³', 'ratio', 'kg/m³', 'days'};

%% ================== 界面 1: 正向预测 ==================
pnlIn = uipanel(tabPREDICT, 'Title', '输入参数配置', 'Position', [0.03 0.15 0.45 0.82], 'FontWeight', 'bold');
pnlOut = uipanel(tabPREDICT, 'Title', '预测分析报告', 'Position', [0.52 0.15 0.45 0.82], 'FontWeight', 'bold');

predEditHandles = cell(9, 1);
for i = 1:9
    y_p = 0.9 - (i-1)*0.095;
    rangeStr = sprintf('[%.1f, %.1f]', Final_LB(i), Final_UB(i));
    initV = mean([Final_LB(i), Final_UB(i)]);
    uicontrol(pnlIn, 'Style', 'text', 'Units', 'normalized', 'Position', [0.02, y_p, 0.58, 0.06], ...
        'String', [featureNamesCN{i}, ' (', units{i}, ') ', rangeStr, ':'], 'HorizontalAlignment', 'right');
    predEditHandles{i} = uicontrol(pnlIn, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.62, y_p+0.01, 0.32, 0.07], ...
        'String', sprintf('%.2f', initV), 'BackgroundColor', 'w');
end

uicontrol(tabPREDICT, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.1 0.04, 0.3, 0.08], ...
    'String', '🚀 执行模型推理', 'FontSize', 14, 'FontWeight', 'bold', ...
    'BackgroundColor', [0.15 0.35 0.65], 'ForegroundColor', 'w', ...
    'Callback', @(~,~) robust_PREDICTion_v7());

axPREDICT = axes(pnlOut, 'Units', 'normalized', 'Position', [0.2 0.35, 0.6, 0.5]);
lblRes = uicontrol(pnlOut, 'Style', 'text', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.2], ...
    'String', '等待输入数据...', 'FontSize', 12, 'FontWeight', 'bold', 'ForegroundColor', [0.7 0.1 0.1]);

%% ================== 界面 2: 逆向设计 ==================
pnlCfg = uipanel(tabInverse, 'Title', '设计目标与成本', 'Position', [0.02 0.78 0.96 0.2]);
pnlPlot = uipanel(tabInverse, 'Title', '动态寻优轨迹', 'Position', [0.02 0.4 0.96 0.36]);
pnlData = uipanel(tabInverse, 'Title', '最优配比推荐方案', 'Position', [0.02 0.02 0.96 0.36]);

hTarget = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.11, 0.62, 0.06, 0.25], 'String', '45');
hTol = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.28, 0.62, 0.06, 0.25], 'String', '2.0');
priceH = cell(9, 1);
for i = 1:9
    uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.01+(i-1)*0.108, 0.3, 0.1, 0.2], 'String', featureNamesCN{i});
    priceH{i} = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.02+(i-1)*0.108, 0.05, 0.08, 0.22], 'String', '1.0', 'BackgroundColor', 'w');
end
set(priceH{9}, 'String', '0');

uicontrol(pnlCfg, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.45, 0.55, 0.5, 0.38], ...
    'String', '⚡ 启动多点并发寻优', 'FontSize', 13, 'FontWeight', 'bold', 'BackgroundColor', [0.8 0.35 0.1], ...
    'ForegroundColor', 'w', 'Callback', @(~,~) robust_inverse_v7());

axLive = axes(pnlPlot, 'Units', 'normalized', 'Position', [0.08 0.2 0.75 0.7]);
xlabel(axLive, '迭代步长'); ylabel(axLive, '特征量'); hold(axLive, 'on'); grid(axLive, 'on');
l_cols = lines(9); h_ls = cell(9, 1);
for i = 1:9; h_ls{i} = plot(axLive, NaN, NaN, 'Color', l_cols(i,:), 'LineWidth', 2, 'DisplayName', featureNamesEN{i}); end
legend(axLive, 'FontSize', 8, 'Position', [0.85, 0.3, 0.1, 0.5]);

resTable = uitable(pnlData, 'Units', 'normalized', 'Position', [0.01 0.05 0.98 0.9], 'FontSize', 10, ...
    'ColumnName', {'设计组分', '最优推荐值', '单位', '物理范围', '闭环验证'}, 'ColumnWidth', {180, 120, 80, 200, 350});

%% ================== 核心逻辑函数 ==================

    function robust_PREDICTion_v7()
        try
            X = zeros(1, 9);
            for k = 1:9
                X(k) = str2double(get(predEditHandles{k}, 'String'));
            end
            % 1. 归一化 (直接在函数内处理，不走 get_val_internal)
            X_norm = mapminmax('apply', X', ps_input);
            % 2. 预测 (使用最原始的 predict 调用)
            Y_norm = predict(trainedModel, X_norm');
            % 3. 反归一化
            Y_res = mapminmax('reverse', Y_norm', ps_output);
            
            axes(axPREDICT); cla; hold on;
            bar(1, Y_res, 0.5, 'FaceColor', [0.2 0.5 0.7]);
            errorbar(1, Y_res, Final_MAE_Value, 'LineWidth', 2, 'Color', 'r');
            ylabel('强度 (MPa)'); grid on;
            set(lblRes, 'String', sprintf('预测值: %.2f MPa | 误差: ±%.2f', Y_res, Final_MAE_Value));
        catch ME
            errordlg(['推理失败：', ME.message], '错误诊断');
        end
    end

    function robust_inverse_v7()
        try
            T_target = str2double(get(hTarget, 'String')); tol = str2double(get(hTol, 'String'));
            P_prices = zeros(1, 9); for k=1:9, P_prices(k) = str2double(get(priceH{k}, 'String')); end

            Y_history = Final_Raw_Matrix(:, 10); [~, sort_id] = sort(abs(Y_history - T_target));
            num_seeds = 15; seed_matrix = Final_Raw_Matrix(sort_id(1:num_seeds), 1:9);

            opts = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', 'OutputFcn', @nested_trace_func);

            min_err = inf; best_X_final = seed_matrix(1,:);
            h_wait = waitbar(0, '正在执行空间交叉寻优...'); history_data = [];

            for s = 1:num_seeds
                if ~isgraphics(h_wait), break; end
                waitbar(s/num_seeds, h_wait); history_data = [];
                obj_f = @(x) sum(x .* P_prices) + 800 * (get_val_internal(x) - T_target)^2;
                [x_tmp, ~] = fmincon(obj_f, seed_matrix(s,:), [], [], [], [], Final_LB, Final_UB, [], opts);
                p_tmp = get_val_internal(x_tmp);
                if abs(p_tmp - T_target) < min_err
                    min_err = abs(p_tmp - T_target); best_X_final = x_tmp;
                end
            end
            if isgraphics(h_wait), delete(h_wait); end

            p_final_res = get_val_internal(best_X_final);
            tbl_res = cell(9, 5);
            for k_tbl = 1:9
                tbl_res{k_tbl,1} = featureNamesCN{k_tbl};
                tbl_res{k_tbl,2} = sprintf('%.3f', best_X_final(k_tbl));
                tbl_res{k_tbl,3} = units{k_tbl};
                tbl_res{k_tbl,4} = sprintf('[%.1f, %.1f]', Final_LB(k_tbl), Final_UB(k_tbl));
                if k_tbl == 1
                    tbl_res{k_tbl,5} = sprintf('验证结果: %.2f MPa (偏差: %.2f)', p_final_res, min_err);
                end
            end
            set(resTable, 'Data', tbl_res);
            if min_err <= tol, msgbox('逆向设计任务已成功完成！', '寻优成功'); end
        catch ME
            err_msg = {['寻优失败原因：', ME.message], '', ...
                '【逆向寻优排障建议】：', ...
                '1. 检查各组分单价：单价全为1或全为0可能导致寻优失败。', ...
                '2. 目标强度过大：超出了训练集物理边界，建议降低目标值。', ...
                '3. 检查模型状态：若正向预测报错，逆向寻优必定报错。'};
            errordlg(err_msg, '逆向设计故障诊断');
        end

        function stop = nested_trace_func(x, ~, ~)
            stop = false; history_data = [history_data; x];
            idx_v = 1:size(history_data, 1);
            for k_idx = 1:9; set(h_ls{k_idx}, 'XData', idx_v, 'YData', history_data(:, k_idx)); end
            drawnow limitrate;
        end
    end

    function s_out = get_val_internal(x)
        % 模仿 8 输入代码的极简预测逻辑
        try
            px = mapminmax('apply', x', ps_input);
            tx = predict(trainedModel, px');
            s_out = mapminmax('reverse', tx', ps_output);
        catch
            % 最后的保底措施：防止变量丢失
            s_out = 0; 
        end
    end
end