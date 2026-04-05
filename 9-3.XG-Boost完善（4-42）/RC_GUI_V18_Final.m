function RC_GUI_V25_Final()
% =====================================================================
% 项目：橡胶混凝土 AI 智能设计系统 (V25 终极兼容版)
% 修复重点：1. 修正函数名空格错误 2. 采用 feval 绕过路径解析报错
% =====================================================================

% --- 模块 1: 加载模型与数据 ---
try
    if ~exist('ConcreteModel_LSBoost.mat', 'file')
        error('FileNotFound: ConcreteModel_LSBoost.mat');
    end
    S_Data = load('ConcreteModel_LSBoost.mat');
    
    % 变量映射
    trainedModel = S_Data.final_model; 
    ps_input = S_Data.ps_input; 
    ps_output = S_Data.ps_output;
    res_raw_data = S_Data.res_raw; 
    Final_MAE = S_Data.bp.MAE;
    
    lb_global = min(res_raw_data(:, 1:9)); 
    ub_global = max(res_raw_data(:, 1:9));
    
    clear predict; rehash;
catch ME
    errordlg(['加载失败: ', ME.message]); return;
end

% --- 2. 主窗口设计 ---
fig = figure('Name', 'RC AI Smart Design System (V25 Final)', ...
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
                    'BackgroundColor', [0.15 0.35 0.65], 'ForegroundColor', 'w', 'Callback', @(~,~) run_prediction_v25());

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
                   'BackgroundColor', [0.8 0.35 0.1], 'ForegroundColor', 'w', 'Callback', @(~,~) run_inverse_v25());

axLive = axes(pnlPlot, 'Units', 'normalized', 'Position', [0.08 0.2 0.7 0.7]);
xlabel(axLive, '步长/Step'); ylabel(axLive, '特征值/Value'); hold(axLive, 'on'); grid(axLive, 'on');
history_X = []; l_cols = lines(9); hLines = cell(9, 1);
for i = 1:9; hLines{i} = plot(axLive, NaN, NaN, 'Color', l_cols(i,:), 'LineWidth', 2, 'DisplayName', featureNamesEN{i}); end
legend(axLive, 'FontSize', 7, 'Position', [0.82, 0.4, 0.15, 0.5]);

resTable = uitable(pnlData, 'Units', 'normalized', 'Position', [0.01 0.05 0.98 0.9], 'FontSize', 10, ...
                   'ColumnName', {'组分 (Component)', '推荐值', '单位', '范围', '回测验证 (Back-test)'}, ...
                   'ColumnWidth', {150, 120, 80, 180, 350});

%% ================== 核心逻辑 ==================

    function run_prediction_v25()
        try
            % 强制环境清理
            evalin('base', 'clear predict');
            
            X = zeros(1, 9);
            curr_h = hEdits; 
            for k = 1:9; X(k) = str2double(get(curr_h{k}, 'String')); end
            
            % 调用预测
            Y_res = get_pred_only(X);
            
            if Y_res <= 0, error('计算引擎返回无效值，请检查模型文件。'); end
            
            axes(axPredict); cla; hold on;
            bar(1, Y_res, 0.4, 'FaceColor', [0.2 0.5 0.7]);
            errorbar(1, Y_res, Final_MAE, 'LineWidth', 2, 'Color', 'r');
            title('Strength Prediction Result'); grid on;
            set(lblRes, 'String', sprintf('预测值: %.2f MPa | 理论误差: ±%.2f MPa', Y_res, Final_MAE));
        catch ME
            errordlg(['推理失败: ', ME.message]);
        end
    end

    function run_inverse_v25()
        try
            target = str2double(get(hTarget, 'String'));
            prices = zeros(1, 9);
            curr_ph = priceHandles;
            for k = 1:9; prices(k) = str2double(get(curr_ph{k}, 'String')); end
            
            Y_hist = res_raw_data(:, 10);
            [~, s_idx] = sort(abs(Y_hist - target));
            seeds = res_raw_data(s_idx(1:15), 1:9);
            
            opts = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', 'OutputFcn', @o_plot_trace);
            best_overall_err = inf; best_X = seeds(1,:);
            h_wait = waitbar(0, 'AI寻优中...');
            
            for s = 1:size(seeds, 1)
                if ~isgraphics(h_wait), break; end
                waitbar(s/15, h_wait); history_X = []; 
                obj = @(x) sum(x .* prices) + 1000 * (get_pred_only(x) - target)^2;
                [x_tmp, ~] = fmincon(obj, seeds(s,:), [], [], [], [], lb_global, ub_global, [], opts);
                
                p_tmp = get_pred_only(x_tmp);
                if abs(p_tmp - target) < best_overall_err
                    best_overall_err = abs(p_tmp - target); best_X = x_tmp;
                end
            end
            if isgraphics(h_wait), delete(h_wait); end
            
            % 填表
            final_p = get_pred_only(best_X);
            tbl = cell(9, 5);
            for k = 1:9
                tbl{k,1} = featureNamesCN{k}; tbl{k,2} = sprintf('%.3f', best_X(k));
                tbl{k,3} = units{k}; tbl{k,4} = sprintf('[%.1f, %.1f]', lb_global(k), ub_global(k));
                if k==1, tbl{k,5} = sprintf('验证强度: %.2f MPa (偏差: %.2f)', final_p, best_overall_err); end
            end
            set(resTable, 'Data', tbl);
        catch ME, errordlg(ME.message); end
    end

    function stop = o_plot_trace(x)
        stop = false; history_X = [history_X; x]; 
        steps = 1:size(history_X, 1);
        if isgraphics(axLive)
            for j = 1:9; set(hLines{j}, 'XData', steps, 'YData', history_X(:, j)); end
            drawnow limitrate;
        end
    end

    function val = get_pred_only(x_dosage)
        % 终极兼容计算助手
        try
            % 修正空格错误并强制维度
            xi = double(reshape(x_dosage, 1, 9));
            px = mapminmax('apply', xi', ps_input);
            % 使用 feval 彻底避开类路径解析报错
            tx = feval('predict', trainedModel, px');
            val = mapminmax('reverse', tx', ps_output);
        catch
            val = 0;
        end
    end
end