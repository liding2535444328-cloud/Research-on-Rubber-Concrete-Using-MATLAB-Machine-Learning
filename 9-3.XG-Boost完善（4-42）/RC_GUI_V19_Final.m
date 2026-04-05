function RC_GUI_V26_Final()
% =====================================================================
% 项目：橡胶混凝土 AI 智能设计系统 (V26 9输入最终稳健版)
% 修复重点：1. 修复匿名函数传参报错 2. 彻底清理 predict 干扰 3. 闭环数据补全
% =====================================================================

% --- 模块 1: 加载模型与数据 ---
try
    if ~exist('ConcreteModel_LSBoost.mat', 'file')
        error('FileNotFound: ConcreteModel_LSBoost.mat');
    end
    S_Data = load('ConcreteModel_LSBoost.mat');
    
    % 核心变量对齐
    trainedModel = S_Data.final_model; 
    ps_input = S_Data.ps_input; 
    ps_output = S_Data.ps_output;
    res_raw_data = S_Data.res_raw; 
    Final_MAE = S_Data.bp.MAE;
    
    lb_global = min(res_raw_data(:, 1:9)); 
    ub_global = max(res_raw_data(:, 1:9));
    
    % 物理级环境清理：防止工作区存在名为 predict 的变量
    evalin('base', 'clear predict'); 
    clear predict; rehash;
catch ME
    errordlg(['加载失败: ', ME.message]); return;
end

% --- 2. 主窗口设计 ---
fig = figure('Name', 'RC AI Smart Design System (V26 9-Input)', ...
             'Units', 'normalized', 'Position', [0.1, 0.05, 0.8, 0.85], ...
             'Color', [0.95 0.95 0.95], 'MenuBar', 'none');

tabGroup = uitabgroup(fig, 'Position', [0 0 1 1]);
tabPredict = uitab(tabGroup, 'Title', '正向预测 / Strength Prediction');
tabInverse = uitab(tabGroup, 'Title', '逆向设计 / Inverse Design');

featureNamesCN = {'水胶比', '橡胶含量', '橡胶粒径', '水泥', '细骨料', '粗骨料', '硅比', '外加剂', '龄期'};
featureNamesEN = {'W/B', 'Rubber', 'MaxSize', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};
units = {'ratio', 'kg/m3', 'mm', 'kg/m3', 'kg/m3', 'kg/m3', 'ratio', 'kg/m3', 'days'};

%% ================== 界面 1: 正向预测 ==================
pnlIn = uipanel(tabPredict, 'Title', '输入参数配置 / Input Features', 'Position', [0.05 0.22 0.43 0.72]);
pnlOut = uipanel(tabPredict, 'Title', '预测分析报告 / AI Analysis', 'Position', [0.52 0.22 0.43 0.72]);

hEdits = cell(9, 1); 
for i = 1:9
    y_n = 0.88 - (i-1)*0.095;
    initVal = lb_global(i) + (ub_global(i) - lb_global(i)) * rand();
    uicontrol(pnlIn, 'Style', 'text', 'Units', 'normalized', 'Position', [0.02, y_n, 0.58, 0.06], ...
              'String', [featureNamesCN{i}, ' (', units{i}, ') [', num2str(lb_global(i),'%.1f'), ',', num2str(ub_global(i),'%.1f'), ']:'], 'HorizontalAlignment', 'right');
    hEdits{i} = uicontrol(pnlIn, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.62, y_n+0.01, 0.3, 0.08], ...
                                  'String', sprintf('%.2f', initVal), 'BackgroundColor', 'w');
end

uicontrol(tabPredict, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.12 0.08, 0.3, 0.08], ...
                    'String', '🚀 执行模型推理 (Predict)', 'FontSize', 14, 'FontWeight', 'bold', ...
                    'BackgroundColor', [0.15 0.35 0.65], 'ForegroundColor', 'w', 'Callback', @(~,~) run_prediction_v26());

axPredict = axes(pnlOut, 'Units', 'normalized', 'Position', [0.15 0.3 0.7 0.6]);
lblRes = uicontrol(pnlOut, 'Style', 'text', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.15], ...
                   'String', '等待输入数据...', 'FontSize', 12, 'FontWeight', 'bold', 'ForegroundColor', [0.7 0.1 0.1]);

%% ================== 界面 2: 逆向设计 ==================
pnlCfg = uipanel(tabInverse, 'Title', '目标与成本配置 / Config', 'Position', [0.02 0.78 0.96 0.2]);
pnlPlot = uipanel(tabInverse, 'Title', '动态寻优轨迹 / Evolution', 'Position', [0.02 0.4 0.96 0.36]);
pnlData = uipanel(tabInverse, 'Title', '最优配比方案 / Results', 'Position', [0.02 0.02 0.96 0.36]);

uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.6, 0.08, 0.3], 'String', '目标强度(MPa):');
hTarget = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.11, 0.62, 0.06, 0.25], 'String', '45');

priceHandles = cell(9, 1);
for i = 1:9
    uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.005+(i-1)*0.11, 0.3, 0.11, 0.2], ...
              'String', featureNamesEN{i}, 'FontSize', 8);
    priceHandles{i} = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.02+(i-1)*0.11, 0.05, 0.08, 0.2], ...
                               'String', '1.0', 'BackgroundColor', 'w');
end

uicontrol(pnlCfg, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.45, 0.55, 0.5, 0.38], ...
                   'String', '⚡ 启动寻优与验证 (Optimization)', 'FontSize', 13, 'FontWeight', 'bold', ...
                   'BackgroundColor', [0.8 0.35 0.1], 'ForegroundColor', 'w', 'Callback', @(~,~) run_inverse_v26());

axLive = axes(pnlPlot, 'Units', 'normalized', 'Position', [0.08 0.2 0.7 0.7]);
xlabel(axLive, '步长/Step'); ylabel(axLive, '数值/Value'); hold(axLive, 'on'); grid(axLive, 'on');
history_X = []; l_cols = lines(9); hLines = cell(9, 1);
for i = 1:9; hLines{i} = plot(axLive, NaN, NaN, 'Color', l_cols(i,:), 'LineWidth', 2, 'DisplayName', featureNamesEN{i}); end
legend(axLive, 'FontSize', 7, 'Position', [0.82, 0.4, 0.15, 0.5]);

resTable = uitable(pnlData, 'Units', 'normalized', 'Position', [0.01 0.05 0.98 0.9], 'FontSize', 10, ...
                   'ColumnName', {'设计组分', '推荐值', '单位', '物理范围', '回测验证 (Back-test)'}, 'ColumnWidth', {150, 120, 80, 180, 350});

%% ================== 核心逻辑函数 ==================

    function run_prediction_v26()
        try
            X = zeros(1, 9);
            % 修复 cell 引用逻辑
            curr_h_list = hEdits; 
            for k = 1:9; X(k) = str2double(get(curr_h_list{k}, 'String')); end
            
            % 调用预测
            Y_res = get_pred_only(X);
            
            axes(axPredict); cla; hold on;
            bar(1, Y_res, 0.4, 'FaceColor', [0.2 0.5 0.7]);
            errorbar(1, Y_res, Final_MAE, 'LineWidth', 2, 'Color', 'r');
            title('Strength Prediction Analysis'); grid on;
            set(lblRes, 'String', sprintf('预测强度: %.2f MPa | 理论误差: ±%.2f', Y_res, Final_MAE));
        catch ME
            errordlg(['正向预测失败: ', ME.message]);
        end
    end

    function run_inverse_v26()
        try
            target_val = str2double(get(hTarget, 'String'));
            prices_vec = zeros(1, 9);
            curr_price_list = priceHandles;
            for k = 1:9; prices_vec(k) = str2double(get(curr_price_list{k}, 'String')); end
            
            % 初始种子选择
            Y_history = res_raw_data(:, 10);
            [~, s_idx] = sort(abs(Y_history - target_val));
            seeds = res_raw_data(s_idx(1:15), 1:9);
            
            % 核心修复：重构 fmincon 目标函数传参模式
            opts = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', 'OutputFcn', @nested_trace_v26);
            best_X = seeds(1,:); min_err = inf;
            h_wait = waitbar(0, '正在执行寻优与回测...');
            
            for s = 1:size(seeds, 1)
                if ~isgraphics(h_wait), break; end
                waitbar(s/15, h_wait); history_X = []; 
                % 修正点：将变量显式包含在匿名函数中
                obj = @(x) sum(x .* prices_vec) + 1000 * (get_pred_only(x) - target_val)^2;
                [x_tmp, ~] = fmincon(obj, seeds(s,:), [], [], [], [], lb_global, ub_global, [], opts);
                
                pred_now = get_pred_only(x_tmp);
                if abs(pred_now - target_val) < min_err
                    min_err = abs(pred_now - target_val); best_X = x_tmp;
                end
            end
            if isgraphics(h_wait), delete(h_wait); end
            
            % 填充最终表格
            final_pred_val = get_pred_only(best_X);
            tbl_data = cell(9, 5);
            for k = 1:9
                tbl_data{k,1} = featureNamesCN{k}; tbl_data{k,2} = sprintf('%.3f', best_X(k));
                tbl_data{k,3} = units{k}; tbl_data{k,4} = sprintf('[%.1f, %.1f]', lb_global(k), ub_global(k));
                if k==1, tbl_data{k,5} = sprintf('回测结果: %.2f MPa (偏差: %.2f)', final_pred_val, min_err); end
            end
            set(resTable, 'Data', tbl_data);
        catch ME
            errordlg(['逆向寻优中断: ', ME.message]);
        end
    end

    function stop = nested_trace_v26(x, ~, ~)
        stop = false; history_X = [history_X; x]; 
        st_steps = 1:size(history_X, 1);
        if isgraphics(axLive)
            for m = 1:9; set(hLines{m}, 'XData', st_steps, 'YData', history_X(:, m)); end
            drawnow limitrate;
        end
    end

    function s_out = get_pred_only(x_dosage)
        % 终极稳健预测引擎
        try
            % 强制转化为 1x9 double 
            xin = double(reshape(x_dosage, 1, 9));
            % 归一化
            px = mapminmax('apply', xin', ps_input);
            % 预测调用 (物理级隔离)
            tx = predict(trainedModel, px');
            % 反归一化
            s_out = mapminmax('reverse', tx', ps_output);
        catch
            s_out = 0; % 计算链断裂时返回0
        end
    end
end