function RubberConcrete_SmartDesign_V17_Final()
% =====================================================================
% 项目：橡胶混凝土强度智能设计系统 (V17 诊断增强版)
% 适配模型：PSO-LSBoost (数据集3)
% 修改重点：新增“数据流监控”模块，可视化内部数据匹配过程
% =====================================================================

% --- 模块 1: 加载模型与数据 ---
try
    if ~exist('ConcreteModel_LSBoost.mat', 'file')
        error('未找到 ConcreteModel_LSBoost.mat，请确保它在当前文件夹。');
    end
    S_Data = load('ConcreteModel_LSBoost.mat');
    
    if isfield(S_Data, 'bp')
        trainedModel = S_Data.bp.final_model; 
        Final_MAE_Value = S_Data.bp.MAE;
        ps_input = S_Data.bp.ps_input;       
        ps_output = S_Data.bp.ps_output;
        res_raw = S_Data.res_raw; 
    else
        trainedModel = S_Data.final_model;
        Final_MAE_Value = 2.15;
        ps_input = S_Data.ps_input;
        ps_output = S_Data.ps_output;
        res_raw = S_Data.res_raw;
    end
    
    X_data_all = res_raw(:, 1:9);
    Final_LB = min(X_data_all);
    Final_UB = max(X_data_all);
    clear predict; rehash;
    
catch ME
    errordlg(['初始化失败：', ME.message], '加载错误'); return;
end

% --- 2. 主窗口设计 ---
fig = figure('Name', 'RC AI Smart Design System (V17 Diagnosis Edition)', ...
    'Units', 'normalized', 'Position', [0.1, 0.05, 0.8, 0.85], ...
    'Color', [0.95 0.95 0.95], 'MenuBar', 'none');

tabGroup = uitabgroup(fig, 'Position', [0 0 1 1]);
tabPREDICT = uitab(tabGroup, 'Title', '正向强度预测 / Forward Prediction');
tabInverse = uitab(tabGroup, 'Title', '逆向配比设计 / Inverse Design');
tabDebug = uitab(tabGroup, 'Title', '数据流监控 / Data Diagnosis'); 

featureNamesCN = {'水胶比', '橡胶含量', '橡胶粒径', '水泥', '细骨料', '粗骨料', '硅比', '外加剂', '龄期'};
featureNamesEN = {'W/B', 'Rubber', 'MaxSize', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};
units = {'ratio', 'kg/m3', 'mm', 'kg/m3', 'kg/m3', 'kg/m3', 'ratio', 'kg/m3', 'days'};

%% ================== 界面 1: 正向预测 (保持不变) ==================
pnlIn = uipanel(tabPREDICT, 'Title', '输入参数配置 / Input Configuration', 'Position', [0.03 0.15 0.45 0.82], 'FontWeight', 'bold');
pnlOut = uipanel(tabPREDICT, 'Title', '预测分析报告 / AI Analysis Report', 'Position', [0.52 0.15 0.45 0.82], 'FontWeight', 'bold');
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
    'String', '🚀 执行模型推理 (Predict)', 'FontSize', 14, 'FontWeight', 'bold', ...
    'BackgroundColor', [0.15 0.35 0.65], 'ForegroundColor', 'w', 'Callback', @(~,~) run_prediction_v17());
axPREDICT = axes(pnlOut, 'Units', 'normalized', 'Position', [0.2 0.35, 0.6, 0.5]);
lblRes = uicontrol(pnlOut, 'Style', 'text', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.2], ...
    'String', '等待输入数据...', 'FontSize', 12, 'FontWeight', 'bold', 'ForegroundColor', [0.7 0.1 0.1]);

%% ================== 界面 2: 逆向设计 (保持不变) ==================
pnlCfg = uipanel(tabInverse, 'Title', '目标与单价配置 / Config', 'Position', [0.02 0.78 0.96 0.2]);
pnlPlot = uipanel(tabInverse, 'Title', '动态寻优轨迹 / Evolution Trajectory', 'Position', [0.02 0.4 0.96 0.36]);
pnlData = uipanel(tabInverse, 'Title', '最优配比推荐方案 / Optimal Results', 'Position', [0.02 0.02 0.96 0.36]);
uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.6, 0.08, 0.25], 'String', '目标强度:');
hTarget = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.11, 0.62, 0.06, 0.25], 'String', '45');
uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.20, 0.6, 0.08, 0.25], 'String', '容许误差:');
hTol = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.28, 0.62, 0.06, 0.25], 'String', '2.0');
priceH = cell(9, 1);
for i = 1:9
    uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.01+(i-1)*0.108, 0.3, 0.1, 0.2], ...
              'String', [featureNamesCN{i}, ' (', featureNamesEN{i}, ')'], 'FontSize', 7);
    priceH{i} = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.02+(i-1)*0.108, 0.05, 0.08, 0.22], 'String', '1.0', 'BackgroundColor', 'w');
end
set(priceH{9}, 'String', '0');
uicontrol(pnlCfg, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.45, 0.55, 0.5, 0.38], ...
    'String', '⚡ 启动 25 点并发寻优 (Start Optimization)', 'FontSize', 13, 'FontWeight', 'bold', ...
    'BackgroundColor', [0.8 0.35 0.1], 'ForegroundColor', 'w', 'Callback', @(~,~) run_inverse_v17());
axLive = axes(pnlPlot, 'Units', 'normalized', 'Position', [0.08 0.2 0.75 0.7]);
xlabel(axLive, '迭代步长 / Step'); ylabel(axLive, '特征量 / Value'); hold(axLive, 'on'); grid(axLive, 'on');
l_cols = lines(9); h_ls = cell(9, 1);
for i = 1:9; h_ls{i} = plot(axLive, NaN, NaN, 'Color', l_cols(i,:), 'LineWidth', 2, 'DisplayName', featureNamesEN{i}); end
legend(axLive, 'FontSize', 8, 'Position', [0.85, 0.3, 0.1, 0.5]);
resTable = uitable(pnlData, 'Units', 'normalized', 'Position', [0.01 0.05 0.98 0.9], 'FontSize', 10, ...
    'ColumnName', {'设计组分', '最优推荐值', '单位', '物理允许范围', '闭环回测验证 (Back-test)'}, 'ColumnWidth', {150, 120, 80, 200, 350});

%% ================== 界面 3: 数据流监控面板 (新增模块) ==================
pnlDebug = uipanel(tabDebug, 'Title', '实时数据匹配状态 / Real-time Data Mapping Status', 'Position', [0.05 0.1 0.9 0.8]);
hDebugLog = uicontrol(pnlDebug, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.02 0.02 0.96 0.96], ...
    'Max', 10, 'HorizontalAlignment', 'left', 'BackgroundColor', [0 0 0], 'ForegroundColor', [0 1 0], ...
    'FontName', 'Consolas', 'FontSize', 10, 'String', '系统就绪。等待执行推理 / System ready. Waiting for inference...');

%% ================== 核心逻辑函数 ==================

    function run_prediction_v17()
        try
            X = zeros(1, 9);
            for k = 1:9; X(k) = str2double(get(predEditHandles{k}, 'String')); end
            
            % 调用带有实时诊断功能的计算引擎
            Y_res = get_pred_with_diagnosis(X);
            
            if Y_res == 0, error('计算结果异常，请切换至“数据流监控”查看原因。'); end
            
            axes(axPREDICT); cla; hold on;
            bar(1, Y_res, 0.5, 'FaceColor', [0.2 0.5 0.7]);
            errorbar(1, Y_res, Final_MAE_Value, 'LineWidth', 2, 'Color', 'r');
            ylabel('强度 (Strength/MPa)'); grid on;
            set(lblRes, 'String', sprintf('预测值: %.2f MPa | 理论误差: ±%.2f', Y_res, Final_MAE_Value));
        catch ME
            errordlg(['推理失败：', ME.message], '故障诊断');
        end
    end

    function run_inverse_v17()
        try
            T_target = str2double(get(hTarget, 'String')); tol = str2double(get(hTol, 'String'));
            prices = zeros(1, 9); for k=1:9, prices(k) = str2double(get(priceH{k}, 'String')); end
            Y_hist = res_raw(:, 10); [~, sort_id] = sort(abs(Y_hist - T_target));
            num_seeds = 25; seed_matrix = res_raw(sort_id(1:num_seeds), 1:9);
            opts = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', 'OutputFcn', @nested_trace_func);
            
            best_err = inf; best_X = seed_matrix(1,:);
            h_wait = waitbar(0, '正在执行空间寻优...', 'Name', 'AI 优化中');
            
            for s = 1:num_seeds
                if ~isgraphics(h_wait), break; end
                waitbar(s/num_seeds, h_wait); history_data = [];
                obj_f = @(x) sum(x .* prices) + 1000 * (get_pred_with_diagnosis(x) - T_target)^2;
                [x_tmp, ~] = fmincon(obj_f, seed_matrix(s,:), [], [], [], [], Final_LB, Final_UB, [], opts);
                p_tmp = get_pred_with_diagnosis(x_tmp);
                if abs(p_tmp - T_target) < best_err
                    best_err = abs(p_tmp - T_target); best_X = x_tmp;
                end
            end
            if isgraphics(h_wait), delete(h_wait); end
            
            p_final = get_pred_with_diagnosis(best_X);
            tbl_data = cell(9, 5);
            for k = 1:9
                tbl_data{k,1} = featureNamesCN{k};
                tbl_data{k,2} = sprintf('%.3f', best_X(k));
                tbl_data{k,3} = units{k};
                tbl_data{k,4} = sprintf('[%.1f, %.1f]', Final_LB(k), Final_UB(k));
                if k == 1, tbl_data{k,5} = sprintf('回测强度: %.2f MPa (偏差: %.2f)', p_final, best_err); end
            end
            set(resTable, 'Data', tbl_data);
            if best_err <= tol, msgbox('逆向设计成功！结果已保存至表格。'); end
        catch ME
            errordlg(['寻优失败：', ME.message]);
        end
        function stop = nested_trace_func(x, ~, ~)
            stop = false; history_data = [history_data; x];
            idx_v = 1:size(history_data, 1);
            for kv = 1:9; set(h_ls{kv}, 'XData', idx_v, 'YData', history_data(:, kv)); end
            drawnow limitrate;
        end
    end

    function s_out = get_pred_with_diagnosis(x_calc)
        % 终极诊断引擎：可视化每一个计算步骤
        try
            % 1. 维度诊断
            logs = {['时间: ', datestr(now, 'HH:MM:SS')], ...
                    ['输入向量: ', num2str(reshape(x_calc, 1, []))], ...
                    ['输入大小: ', num2str(size(x_calc,1)), 'x', num2str(size(x_calc,2))]};
            
            % 2. 归一化诊断
            px = mapminmax('apply', x_calc', ps_input);
            logs{end+1} = ['归一化处理: 成功 (1x9)'];
            
            % 3. 模型对象检测
            logs{end+1} = ['模型类型: ', class(trainedModel)];
            
            % 4. 执行预测 (采用最高级调用)
            try
                % 路径 A: 类静态路径
                tx = classreg.learning.ensemble.predict(trainedModel, px');
                logs{end+1} = '调用路径: classreg.learning 静态调用成功';
            catch
                % 路径 B: 强制 feval
                tx = feval('predict', trainedModel, px');
                logs{end+1} = '调用路径: feval 动态补救成功';
            end
            
            % 5. 输出诊断
            s_out = mapminmax('reverse', tx', ps_output);
            logs{end+1} = ['反归一化输出: ', num2str(s_out), ' MPa'];
            
            % 更新日志界面
            set(hDebugLog, 'String', logs, 'ForegroundColor', [0 1 0]);
            
        catch ME
            s_out = 0;
            errLogs = {['!! 错误 ID: ', ME.identifier], ...
                       ['!! 错误信息: ', ME.message], ...
                       '!! 建议: 1. 检查命令行是否有变量 predict 2. 输入 clear all 重试'};
            set(hDebugLog, 'String', errLogs, 'ForegroundColor', [1 0 0]);
        end
    end

end