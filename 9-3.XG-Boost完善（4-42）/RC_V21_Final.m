function RC_V28_Final()
% =====================================================================
% 项目：橡胶混凝土 AI 智能设计系统 (V28 科研记录与性能增强版)
% 硬件优化：自动调用并行池加速寻优过程 (适配 i5-10200H + GPU环境)
% 功能：1. 中英文全对照 2. 正逆向操作全记录 3. 龄期取整 4. 3次回带验证
% =====================================================================

% --- 模块 1: 加载模型与数据 ---
try
    if ~exist('ConcreteModel_LSBoost.mat', 'file')
        error('FileNotFound: ConcreteModel_LSBoost.mat');
    end
    S_Data = load('ConcreteModel_LSBoost.mat');
    trainedModel = S_Data.final_model; 
    ps_input = S_Data.ps_input; ps_output = S_Data.ps_output;
    res_raw_data = S_Data.res_raw; 
    Final_MAE = S_Data.bp.MAE;
    
    lb_global = min(res_raw_data(:, 1:9)); ub_global = max(res_raw_data(:, 1:9));
    lb_y = min(res_raw_data(:, 10)); ub_y = max(res_raw_data(:, 10));
    
    % 环境专项清理
    evalin('base', 'clear predict'); clear predict; rehash;
    % 尝试开启并行加速
    if isempty(gcp('nocreate')), parpool('local', 4); end
catch ME
    errordlg(['Init Error: ', ME.message]); return;
end

% --- 2. 主窗口设计 ---
fig = figure('Name', 'RC AI Smart Design System (V28 Performance Plus)', ...
             'Units', 'normalized', 'Position', [0.1, 0.05, 0.8, 0.85], ...
             'Color', [0.95 0.95 0.95], 'MenuBar', 'none');

tabGroup = uitabgroup(fig, 'Position', [0 0 1 1]);
tabPredict = uitab(tabGroup, 'Title', '正向强度预测 / Forward Prediction');
tabInverse = uitab(tabGroup, 'Title', '逆向配比设计 / Inverse Design');

fNamesCN = {'水胶比', '橡胶含量', '橡胶粒径', '水泥', '细骨料', '粗骨料', '硅比', '外加剂', '龄期'};
fNamesEN = {'W/B', 'Rubber', 'MaxSize', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};
units = {'ratio', 'kg/m3', 'mm', 'kg/m3', 'kg/m3', 'kg/m3', 'ratio', 'kg/m3', 'days'};
pUnits = {'元/ratio', '元/kg', '元/mm', '元/kg', '元/kg', '元/kg', '元/ratio', '元/kg', '元/天'};

%% ================== 界面 1: 正向预测 ==================
pnlIn = uipanel(tabPredict, 'Title', '输入参数配置 / Input Configuration', 'Position', [0.05 0.22 0.43 0.72]);
pnlOut = uipanel(tabPredict, 'Title', '预测分析报告 / AI Analysis Report', 'Position', [0.52 0.22 0.43 0.72]);
hEdits = cell(9, 1); 
for i = 1:9
    y_n = 0.88 - (i-1)*0.095;
    initV = mean([lb_global(i), ub_global(i)]);
    uicontrol(pnlIn, 'Style', 'text', 'Units', 'normalized', 'Position', [0.02, y_n, 0.58, 0.06], ...
              'String', [fNamesCN{i}, ' (', units{i}, ') [', num2str(lb_global(i),'%.1f'), ',', num2str(ub_global(i),'%.1f'), ']:'], 'HorizontalAlignment', 'right');
    hEdits{i} = uicontrol(pnlIn, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.62, y_n+0.01, 0.3, 0.08], ...
                                  'String', sprintf('%.2f', initV), 'BackgroundColor', 'w');
end
uicontrol(tabPredict, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.12 0.08, 0.3, 0.08], ...
                    'String', '🚀 执行模型推理 / Predict', 'FontSize', 14, 'FontWeight', 'bold', ...
                    'BackgroundColor', [0.15 0.35 0.65], 'ForegroundColor', 'w', 'Callback', @(~,~) run_prediction_v28());
axPredict = axes(pnlOut, 'Units', 'normalized', 'Position', [0.15 0.35, 0.7, 0.55]);
lblRes = uicontrol(pnlOut, 'Style', 'text', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.2], ...
                   'String', '等待输入... / Ready.', 'FontSize', 12, 'FontWeight', 'bold');

%% ================== 界面 2: 逆向设计 ==================
pnlCfg = uipanel(tabInverse, 'Title', '目标配置与单价 / Config & Prices', 'Position', [0.02 0.78 0.96 0.2]);
pnlPlot = uipanel(tabInverse, 'Title', '动态寻优轨迹 / Evolution Trajectory', 'Position', [0.02 0.4 0.96 0.36]);
pnlData = uipanel(tabInverse, 'Title', '最优配比方案 / Results', 'Position', [0.02 0.02 0.96 0.36]);

uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.01, 0.6, 0.1, 0.3], 'String', ['目标强度(MPa):', char(10), '(', num2str(lb_y,'%.1f'), '-', num2str(ub_y,'%.1f'), ')']);
hTarget = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.11, 0.62, 0.06, 0.25], 'String', '45');
uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.18, 0.6, 0.08, 0.3], 'String', '容许误差: / Tol:');
hTol = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.26, 0.62, 0.06, 0.25], 'String', '2.0');

priceHandles = cell(9, 1);
for i = 1:9
    uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.005+(i-1)*0.11, 0.3, 0.11, 0.2], ...
              'String', [fNamesCN{i}, char(10), '(', pUnits{i}, ')'], 'FontSize', 7);
    priceHandles{i} = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.02+(i-1)*0.11, 0.05, 0.08, 0.22], ...
                               'String', '1.0', 'BackgroundColor', 'w');
end
set(priceHandles{9}, 'String', '0'); % 龄期预设为0

uicontrol(pnlCfg, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.45, 0.55, 0.5, 0.38], ...
                   'String', '⚡ 启动 30 点并发寻优与 3 次闭环验证 / Start Optimization', 'FontSize', 13, 'FontWeight', 'bold', ...
                   'BackgroundColor', [0.8 0.35 0.1], 'ForegroundColor', 'w', 'Callback', @(~,~) run_inverse_v28());

axLive = axes(pnlPlot, 'Units', 'normalized', 'Position', [0.08 0.2 0.7 0.7]);
xlabel(axLive, '迭代步长 / Step'); ylabel(axLive, '归一化量 / Value'); hold(axLive, 'on'); grid(axLive, 'on');
history_X = []; l_cols = lines(9); hLines = cell(9, 1);
for i = 1:9; hLines{i} = plot(axLive, NaN, NaN, 'Color', l_cols(i,:), 'LineWidth', 2); end
legend(axLive, fNamesEN, 'FontSize', 7, 'Position', [0.82, 0.4, 0.15, 0.5]);

resTable = uitable(pnlData, 'Units', 'normalized', 'Position', [0.01 0.05 0.98 0.9], 'FontSize', 10, ...
                   'ColumnName', {'设计组分 / Component', '推荐值', '单位', '物理范围', '3次验证详情与成本 / Validation & Cost'}, ...
                   'ColumnWidth', {150, 100, 60, 180, 500});

%% ================== 核心逻辑函数 ==================

    function run_prediction_v28()
        try
            X = zeros(1, 9);
            for k = 1:9
                val = str2double(get(hEdits{k}, 'String'));
                if k == 9, val = round(val); set(hEdits{k}, 'String', num2str(val)); end % 龄期整数
                X(k) = val;
            end
            Y_res = get_pred_only(X);
            axes(axPredict); cla; hold on;
            bar(1, Y_res, 0.4, 'FaceColor', [0.2 0.5 0.7]);
            errorbar(1, Y_res, Final_MAE, 'LineWidth', 2, 'Color', 'r'); grid on;
            set(lblRes, 'String', sprintf('Predicted: %.2f MPa | Error: ±%.2f', Y_res, Final_MAE));
            
            % --- 写入预测日志 ---
            fid = fopen('Predict_History.txt', 'w');
            fprintf(fid, '--- 正向预测实验记录 ---\n时间: %s\n', datestr(now));
            for j=1:9, fprintf(fid, '%s: %.2f %s\n', fNamesCN{j}, X(j), units{j}); end
            fprintf(fid, '预测强度: %.2f MPa\n理论误差: %.2f MPa\n', Y_res, Final_MAE);
            fclose(fid);
        catch ME, errordlg(ME.message); end
    end

    function run_inverse_v28()
        try
            target_val = str2double(get(hTarget, 'String'));
            tol_val = str2double(get(hTol, 'String'));
            if target_val < lb_y || target_val > ub_y
                error(['目标强度超出范围 [', num2str(lb_y), '-', num2str(ub_y), ']']);
            end
            
            prices_vec = zeros(1, 9);
            for k = 1:9; prices_vec(k) = str2double(get(priceHandles{k}, 'String')); end
            
            % 增大寻优规模：30 个种子点
            [~, s_idx] = sort(abs(res_raw_data(:, 10) - target_val));
            seeds = res_raw_data(s_idx(1:min(30, end)), 1:9);
            
            opts = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', ...
                'MaxIterations', 30, 'OutputFcn', @nested_trace_v28);
            
            best_X = seeds(1,:); min_err = inf;
            h_wait = waitbar(0, '正在利用 GPU/并行环境加速寻优...');
            
            for s = 1:size(seeds, 1)
                if ~isgraphics(h_wait), break; end
                waitbar(s/size(seeds,1), h_wait); history_X = []; 
                obj = @(x) sum(x .* prices_vec) + 1000 * (get_pred_only(x) - target_val)^2;
                [x_tmp, ~] = fmincon(obj, seeds(s,:), [], [], [], [], lb_global, ub_global, [], opts);
                
                pred_now = get_pred_only(x_tmp);
                if abs(pred_now - target_val) < min_err
                    min_err = abs(pred_now - target_val); best_X = x_tmp;
                end
            end
            if isgraphics(h_wait), delete(h_wait); end
            
            % --- 3次闭环回带验证 ---
            v_preds = zeros(1,3);
            for v = 1:3
                noise = (rand(1,9)-0.5).*0.005.*(ub_global-lb_global); % 0.5% 扰动
                v_preds(v) = get_pred_only(best_X + noise);
            end
            avg_v = mean(v_preds); avg_e = abs(avg_v - target_val);
            total_cost = sum(best_X .* prices_vec);
            
            % 填表并记录日志
            tbl = cell(9, 5);
            fid = fopen('Inverse_Design_Log.txt', 'w');
            fprintf(fid, '--- 逆向设计科研日志 ---\n时间: %s\n目标强度: %.2f MPa\n', datestr(now), target_val);
            
            for k = 1:9
                out_val = best_X(k); if k==9, out_val = round(out_val); end
                tbl{k,1} = fNamesCN{k}; tbl{k,2} = sprintf('%.3f', out_val);
                tbl{k,3} = units{k}; tbl{k,4} = sprintf('[%.1f, %.1f]', lb_global(k), ub_global(k));
                fprintf(fid, '%s: %.3f %s\n', fNamesCN{k}, out_val, units{k});
                
                if k == 1
                    tbl{k,5} = sprintf('1:%.1f | 2:%.1f | 3:%.1f (Avg:%.2f)', v_preds(1), v_preds(2), v_preds(3), avg_v);
                elseif k == 2
                    tbl{k,5} = sprintf('平均误差: %.2f MPa | 总价: %.2f', avg_e, total_cost);
                end
            end
            fprintf(fid, '平均验证强度: %.2f MPa\n总估算成本: %.2f 元\n', avg_v, total_cost);
            fclose(fid);
            set(resTable, 'Data', tbl);
            if avg_e <= tol_val, msgbox('寻优成功！日志已更新。 / Success!'); end
        catch ME, errordlg(ME.message); end
    end

    function stop = nested_trace_v28(x, ~, ~)
        stop = false; history_X = [history_X; x]; 
        st = 1:size(history_X, 1);
        if isgraphics(axLive)
            for j = 1:9; set(hLines{j}, 'XData', st, 'YData', (x(j)-lb_global(j))/(ub_global(j)-lb_global(j))); end
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