function RC_GUI_V27_Final()
% =====================================================================
% 项目：橡胶混凝土 AI 智能设计系统 (V27 最终全功能版)
% 修复重点：1. 修复匿名函数传参报错 2. 闭环3次回带验证 3. 价格/范围检测
% =====================================================================

% --- 模块 1: 加载模型与数据 ---
try
    if ~exist('ConcreteModel_LSBoost.mat', 'file')
        error('未找到 ConcreteModel_LSBoost.mat 文件。');
    end
    S_Data = load('ConcreteModel_LSBoost.mat');
    
    trainedModel = S_Data.final_model; 
    ps_input = S_Data.ps_input; ps_output = S_Data.ps_output;
    res_raw_data = S_Data.res_raw; 
    Final_MAE = S_Data.bp.MAE;
    
    lb_global = min(res_raw_data(:, 1:9)); 
    ub_global = max(res_raw_data(:, 1:9));
    % 强度范围锁定
    lb_y = min(res_raw_data(:, 10)); ub_y = max(res_raw_data(:, 10));
    
    evalin('base', 'clear predict'); clear predict; rehash;
catch ME
    errordlg(['加载失败: ', ME.message]); return;
end

% --- 2. 主窗口设计 (中英文双语) ---
fig = figure('Name', 'RC AI Smart Design System (V27 9-Input)', ...
             'Units', 'normalized', 'Position', [0.1, 0.05, 0.8, 0.85], ...
             'Color', [0.95 0.95 0.95], 'MenuBar', 'none');

tabGroup = uitabgroup(fig, 'Position', [0 0 1 1]);
tabPredict = uitab(tabGroup, 'Title', '正向预测 / Strength Prediction');
tabInverse = uitab(tabGroup, 'Title', '逆向设计 / Inverse Design');

featureNamesCN = {'水胶比', '橡胶含量', '橡胶粒径', '水泥', '细骨料', '粗骨料', '硅比', '外加剂', '龄期'};
featureNamesEN = {'W/B', 'Rubber', 'MaxSize', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};
units = {'ratio', 'kg/m3', 'mm', 'kg/m3', 'kg/m3', 'kg/m3', 'ratio', 'kg/m3', 'days'};
priceUnits = {'元/ratio', '元/kg', '元/mm', '元/kg', '元/kg', '元/kg', '元/ratio', '元/kg', '元/天'};

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
                    'BackgroundColor', [0.15 0.35 0.65], 'ForegroundColor', 'w', 'Callback', @(~,~) run_prediction_v27());
axPredict = axes(pnlOut, 'Units', 'normalized', 'Position', [0.15 0.3 0.7 0.6]);
lblRes = uicontrol(pnlOut, 'Style', 'text', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.15], ...
                   'String', '等待输入数据...', 'FontSize', 12, 'FontWeight', 'bold', 'ForegroundColor', [0.7 0.1 0.1]);

%% ================== 界面 2: 逆向设计 ==================
pnlCfg = uipanel(tabInverse, 'Title', '目标与成本配置 / Config', 'Position', [0.02 0.78 0.96 0.2]);
pnlPlot = uipanel(tabInverse, 'Title', '动态寻优轨迹 / Evolution', 'Position', [0.02 0.4 0.96 0.36]);
pnlData = uipanel(tabInverse, 'Title', '优化配比方案 / Results', 'Position', [0.02 0.02 0.96 0.36]);

uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.6, 0.1, 0.3], 'String', ['目标强度(', num2str(lb_y,'%.1f'), '-', num2str(ub_y,'%.1f'), '):']);
hTarget = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.12, 0.62, 0.06, 0.25], 'String', '45');
uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.20, 0.6, 0.1, 0.3], 'String', '容许误差(MPa):');
hTol = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.3, 0.62, 0.06, 0.25], 'String', '2.0');

priceHandles = cell(9, 1);
for i = 1:9
    uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.005+(i-1)*0.11, 0.3, 0.11, 0.2], ...
              'String', [featureNamesCN{i}, '(', priceUnits{i}, ')'], 'FontSize', 7, 'HorizontalAlignment', 'center');
    priceHandles{i} = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.02+(i-1)*0.11, 0.05, 0.08, 0.2], ...
                               'String', '1.0', 'BackgroundColor', 'w');
end
set(priceHandles{9}, 'String', '0'); % 龄期价格默认为0

uicontrol(pnlCfg, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.45, 0.55, 0.5, 0.38], ...
                   'String', '⚡ 启动寻优与闭环验证 (Optimization)', 'FontSize', 13, 'FontWeight', 'bold', ...
                   'BackgroundColor', [0.8 0.35 0.1], 'ForegroundColor', 'w', 'Callback', @(~,~) run_inverse_v27());

axLive = axes(pnlPlot, 'Units', 'normalized', 'Position', [0.08 0.2 0.7 0.7]);
xlabel(axLive, '迭代步长 / Step'); ylabel(axLive, '特征量 / Value'); hold(axLive, 'on'); grid(axLive, 'on');
history_X = []; l_cols = lines(9); hLines = cell(9, 1);
for i = 1:9; hLines{i} = plot(axLive, NaN, NaN, 'Color', l_cols(i,:), 'LineWidth', 2, 'DisplayName', featureNamesEN{i}); end
legend(axLive, 'FontSize', 7, 'Position', [0.82, 0.4, 0.15, 0.5]);

resTable = uitable(pnlData, 'Units', 'normalized', 'Position', [0.01 0.05 0.98 0.9], 'FontSize', 10, ...
                   'ColumnName', {'设计组分 (Component)', '推荐值', '单位', '物理范围', '3次闭环验证与误差 (Back-test)'}, 'ColumnWidth', {150, 100, 60, 180, 480});

%% ================== 核心逻辑函数 ==================

    function run_prediction_v27()
        try
            X = zeros(1, 9);
            curr_h_list = hEdits; 
            for k = 1:9
                val = str2double(get(curr_h_list{k}, 'String'));
                if k == 9; val = round(val); set(curr_h_list{k}, 'String', num2str(val)); end % 龄期取整
                X(k) = val;
            end
            
            Y_res = get_pred_only(X);
            axes(axPredict); cla; hold on;
            bar(1, Y_res, 0.4, 'FaceColor', [0.2 0.5 0.7]);
            errorbar(1, Y_res, Final_MAE, 'LineWidth', 2, 'Color', 'r');
            title('Strength Prediction Result'); grid on;
            set(lblRes, 'String', sprintf('预测强度: %.2f MPa | 理论误差: ±%.2f', Y_res, Final_MAE));
        catch ME
            errordlg(['预测失败: ', ME.message]);
        end
    end

    function run_inverse_v27()
        try
            target_val = str2double(get(hTarget, 'String'));
            tol_val = str2double(get(hTol, 'String'));
            
            % 强度输入范围检测
            if target_val < lb_y || target_val > ub_y
                error(['输入强度超出数据集范围 [', num2str(lb_y), ', ', num2str(ub_y), ']，无法预测。']);
            end
            
            prices_vec = zeros(1, 9);
            for k = 1:9; prices_vec(k) = str2double(get(priceHandles{k}, 'String')); end
            
            % 初始种子
            [~, s_idx] = sort(abs(res_raw_data(:, 10) - target_val));
            seeds = res_raw_data(s_idx(1:15), 1:9);
            
            % 核心优化配置
            opts = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', 'OutputFcn', @nested_trace_v27);
            best_X = seeds(1,:); min_err = inf;
            h_wait = waitbar(0, '正在执行多点寻优与闭环回带验证...');
            
            for s = 1:size(seeds, 1)
                if ~isgraphics(h_wait), break; end
                waitbar(s/15, h_wait); history_X = []; 
                % 修复传参报错：显式传递 target_val 和价格向量
                obj = @(x) sum(x .* prices_vec) + 1000 * (get_pred_only(x) - target_val)^2;
                [x_tmp, ~] = fmincon(obj, seeds(s,:), [], [], [], [], lb_global, ub_global, [], opts);
                
                pred_now = get_pred_only(x_tmp);
                if abs(pred_now - target_val) < min_err
                    min_err = abs(pred_now - target_val); best_X = x_tmp;
                end
            end
            if isgraphics(h_wait), delete(h_wait); end
            
            % --- 3次闭环回带验证 ---
            v_preds = zeros(1,3); v_errs = zeros(1,3);
            for v = 1:3
                noise = (rand(1,9)-0.5).*0.01.*(ub_global-lb_global); % 1% 随机噪声模拟实验误差
                v_preds(v) = get_pred_only(best_X + noise);
                v_errs(v) = abs(v_preds(v) - target_val);
            end
            avg_v = mean(v_preds); avg_e = mean(v_errs);
            total_cost = sum(best_X .* prices_vec);
            
            % 填表
            tbl_data = cell(9, 5);
            for k = 1:9
                tbl_data{k,1} = featureNamesCN{k}; 
                val_out = best_X(k); if k==9; val_out = round(val_out); end
                tbl_data{k,2} = sprintf('%.3f', val_out);
                tbl_data{k,3} = units{k}; 
                tbl_data{k,4} = sprintf('[%.1f, %.1f]', lb_global(k), ub_global(k));
                if k == 1
                    tbl_data{k,5} = sprintf('1:%.1f | 2:%.1f | 3:%.1f (Avg:%.2f, Err:%.2f)', v_preds(1), v_preds(2), v_preds(3), avg_v, avg_e);
                elseif k == 2
                    tbl_data{k,5} = sprintf('估算方案总成本: %.2f 元', total_cost);
                end
            end
            set(resTable, 'Data', tbl_data);
            if avg_e <= tol_val, msgbox('寻优成功！闭环验证误差在容许范围内。'); end
            
        catch ME
            errordlg(['寻优异常: ', ME.message]);
        end
    end

    function stop = nested_trace_v27(x, ~, ~)
        stop = false; history_X = [history_X; x]; 
        st_steps = 1:size(history_X, 1);
        if isgraphics(axLive)
            for m = 1:9; set(hLines{m}, 'XData', st_steps, 'YData', history_X(:, m)); end
            drawnow limitrate;
        end
    end

    function s_out = get_pred_only(x_in)
        try
            xin_f = double(reshape(x_in, 1, 9));
            px = mapminmax('apply', xin_f', ps_input);
            tx = predict(trainedModel, px');
            s_out = mapminmax('reverse', tx', ps_output);
        catch
            s_out = 0; 
        end
    end
end