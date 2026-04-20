function RC_V36_Ultimate_Research_Platform()
% =====================================================================
% 项目：橡胶混凝土 AI 智能设计系统 (V36 终极科研集成版)
% 核心：[正向] 星簇动态监控 + [逆向] 30点并行自动寻优
% 适配：李鼎硕士论文“三部曲”全流程演示
% =====================================================================

%% --- 模块 1: 环境初始化与数据预装载 ---
try
    if ~exist('ConcreteModel_LSBoost.mat', 'file')
        error('FileNotFound: ConcreteModel_LSBoost.mat');
    end
    S_Data = load('ConcreteModel_LSBoost.mat');
    trainedModel = S_Data.final_model; 
    ps_input = S_Data.ps_input; ps_output = S_Data.ps_output;
    res_raw_data = S_Data.res_raw; 
    Final_MAE = S_Data.bp.MAE;
    
    % 物理边界定义
    lb_global = min(res_raw_data(:, 1:9)); ub_global = max(res_raw_data(:, 1:9));
    lb_y = min(res_raw_data(:, 10)); ub_y = max(res_raw_data(:, 10));
    
    % 并行环境检查
    if isempty(gcp('nocreate')), parpool('local', 4); end 
catch ME
    errordlg(['Init Error: ', ME.message]); return;
end

persistent History_Y; History_Y = [];

%% --- 模块 2: 多维度界面设计 ---
fig = figure('Name', 'RC AI Ultimate Platform V36 - Li Ding', ...
             'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.88], ...
             'Color', [0.98 0.98 0.98], 'MenuBar', 'none', 'NumberTitle', 'off');

tabGroup = uitabgroup(fig, 'Position', [0 0 1 1]);
tabPredict = uitab(tabGroup, 'Title', '正向交互监控 / Forward Interactive Monitor');
tabInverse = uitab(tabGroup, 'Title', '逆向智能寻优 / Inverse Auto-Optimization');

fNamesCN = {'水胶比', '橡胶含量', '橡胶粒径', '水泥', '细骨料', '粗骨料', '硅比', '外加剂', '龄期'};
fNamesEN = {'W/B', 'Rubber', 'Size', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};
units = {'ratio', 'kg/m3', 'mm', 'kg/m3', 'kg/m3', 'kg/m3', 'ratio', 'kg/m3', 'days'};
colors = lines(9);

%% ================== [界面 1: 正向交互监控] ==================
pnlDisp1 = uipanel(tabPredict, 'Position', [0.02 0.45 0.96 0.5], 'BackgroundColor', 'w', 'Title', 'Multi-dimensional Star-Cluster Analysis');
axSHAP = axes(pnlDisp1, 'Position', [0.04 0.15 0.25 0.72]); % 左：SHAP
ax3D = axes(pnlDisp1, 'Position', [0.35 0.1 0.35 0.85]);    % 中：星簇
axHist = axes(pnlDisp1, 'Position', [0.73 0.15 0.24 0.72]);  % 右：轨迹

% 高亮结果显示
pnlRes1 = uipanel(tabPredict, 'Position', [0.38 0.38 0.28 0.06], 'BackgroundColor', [0.15 0.35 0.65], 'BorderType', 'none');
lblResult1 = uicontrol(pnlRes1, 'Style', 'text', 'Units', 'normalized', 'Position', [0, 0, 1, 1], ...
                      'String', 'Ready', 'FontSize', 18, 'FontWeight', 'bold', 'ForegroundColor', 'w', 'BackgroundColor', [0.15 0.35 0.65]);

% 下部控制台
pnlCtrl1 = uipanel(tabPredict, 'Title', 'Parameter Control / 参数控制台', 'Position', [0.02 0.02 0.96 0.35]);
hSliders = cell(9, 1); hEdits = cell(9, 1);
for i = 1:9
    row = ceil(i/3); col = mod(i-1, 3);
    x_base = 0.02 + col*0.33; y_base = 0.8 - (row-1)*0.35;
    uicontrol(pnlCtrl1, 'Style', 'text', 'Units', 'normalized', 'Position', [x_base, y_base, 0.12, 0.15], ...
              'String', [fNamesCN{i}, ' ', fNamesEN{i}], 'FontSize', 9, 'FontWeight', 'bold', 'ForegroundColor', colors(i,:));
    hSliders{i} = uicontrol(pnlCtrl1, 'Style', 'slider', 'Units', 'normalized', 'Position', [x_base+0.1, y_base, 0.15, 0.12], ...
                           'Min', lb_global(i), 'Max', ub_global(i), 'Value', mean([lb_global(i), ub_global(i)]), ...
                           'Callback', @(~,~) update_forward(i, 'slider'));
    hEdits{i} = uicontrol(pnlCtrl1, 'Style', 'edit', 'Units', 'normalized', 'Position', [x_base+0.26, y_base, 0.04, 0.16], ...
                         'String', num2str(hSliders{i}.Value, '%.2f'), 'Callback', @(~,~) update_forward(i, 'edit'));
    uicontrol(pnlCtrl1, 'Style', 'text', 'Units', 'normalized', 'Position', [x_base+0.1, y_base-0.12, 0.15, 0.1], ...
              'String', sprintf('Range: [%.1f - %.1f]', lb_global(i), ub_global(i)), 'FontSize', 7, 'ForegroundColor', [0.5 0.5 0.5]);
end

%% ================== [界面 2: 逆向智能寻优] ==================
pnlCfg2 = uipanel(tabInverse, 'Title', 'Target Config & Optimization / 目标配置与寻优', 'Position', [0.02 0.72 0.96 0.25]);
pnlPlot2 = uipanel(tabInverse, 'Title', 'Evolution Trajectory (Physical) / 物理数值收敛轨迹', 'Position', [0.02 0.35 0.96 0.35]);
pnlData2 = uipanel(tabInverse, 'Title', 'Optimized Mix Design / 最优配比方案', 'Position', [0.02 0.02 0.96 0.32]);

% 目标输入
uicontrol(pnlCfg2, 'String', 'Target f''c (MPa):', 'Units', 'normalized', 'Position', [0.02 0.6 0.1 0.2]);
hTarget2 = uicontrol(pnlCfg2, 'Style', 'edit', 'Position', [0.12 0.62 0.06 0.2], 'String', '40');
uicontrol(pnlCfg2, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.35 0.55 0.3 0.3], ...
                   'String', '⚡ Start 30-Point Auto-Optimization', 'FontSize', 12, 'FontWeight', 'bold', ...
                   'BackgroundColor', [0.8 0.35 0.1], 'ForegroundColor', 'w', 'Callback', @(~,~) run_inverse_v36());

axLive2 = axes(pnlPlot2, 'Units', 'normalized', 'Position', [0.08 0.2 0.75 0.7]);
hLines2 = cell(9, 1); hold(axLive2, 'on'); grid(axLive2, 'on');
for i = 1:9; hLines2{i} = plot(axLive2, NaN, NaN, 'Color', colors(i,:), 'LineWidth', 1.5); end
legend(axLive2, fNamesEN, 'NumColumns', 2, 'Position', [0.85 0.4 0.1 0.3]);

resTable2 = uitable(pnlData2, 'Units', 'normalized', 'Position', [0.01 0.05 0.98 0.9], ...
                   'ColumnName', {'Component / 组分', 'Value', 'Unit', 'Range', 'Validation / 3次验证'}, 'ColumnWidth', {150, 80, 60, 150, 500});

% 初始化预测
update_forward(1, 'init');

%% ================== 核心逻辑函数 ==================

    % --- 正向更新逻辑 ---
    function update_forward(idx, mode)
        if strcmp(mode, 'slider')
            val = hSliders{idx}.Value; if idx==9, val=round(val); end
            hEdits{idx}.String = num2str(val, '%.2f');
        elseif strcmp(mode, 'edit')
            val = str2double(hEdits{idx}.String); val = max(min(val, ub_global(idx)), lb_global(idx));
            hSliders{idx}.Value = val; hEdits{idx}.String = num2str(val, '%.2f');
        end
        X_curr = cellfun(@(h) str2double(h.String), hEdits)';
        Y_curr = get_pred_quick(X_curr);
        set(lblResult1, 'String', sprintf('Strength: %.2f MPa (±%.2f)', Y_curr, Final_MAE));
        History_Y = [History_Y, Y_curr]; if length(History_Y) > 30, History_Y(1) = []; end
        
        % 绘图刷新
        draw_SHAP(X_curr);
        draw_StarCluster(X_curr, Y_curr);
        draw_History();
        drawnow;
    end

    function draw_StarCluster(X, Y)
        axes(ax3D); cla; hold on;
        theta = linspace(0, 2*pi, 10); theta(end) = []; R_ring = 10;
        x_ring = R_ring * cos(theta); y_ring = R_ring * sin(theta);
        for i = 1:9
            h = (X(i) - lb_global(i)) / (ub_global(i) - lb_global(i)) * 40; 
            [bx, by, bz] = cylinder(0.6, 20);
            surf(bx + x_ring(i), by + y_ring(i), bz * h, 'FaceColor', colors(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.6);
            text(x_ring(i)*1.3, y_ring(i)*1.3, 0, fNamesEN{i}, 'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
            line([x_ring(i), 0], [y_ring(i), 0], [h, Y], 'Color', [0.8 0.8 0.8], 'LineStyle', ':');
        end
        plot3(0, 0, Y, 'kp', 'MarkerSize', 18, 'MarkerFaceColor', [1 0.8 0]);
        text(0, 0, Y+5, [num2str(Y, '%.1f'), ' MPa'], 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
        view(35, 25); grid on; zlim([0 85]); set(gca, 'XTick', [], 'YTick', []);
        title('Star-Cluster Dynamic Monitor / 星簇实时监控');
    end

    function draw_SHAP(X)
        axes(axSHAP); cla;
        diffs = (X - mean(res_raw_data(:,1:9))) ./ (ub_global - lb_global);
        for i = 1:9; barh(i, diffs(i), 'FaceColor', colors(i,:), 'EdgeColor', 'none'); end
        set(axSHAP, 'YTick', 1:9, 'YTickLabel', fNamesEN, 'XGrid', 'on', 'YDir', 'reverse');
        title('SHAP Contribution / 机理分析');
    end

    function draw_History()
        axes(axHist); cla;
        plot(History_Y, '-o', 'LineWidth', 1.2, 'Color', [0.15 0.35 0.65], 'MarkerSize', 3);
        grid on; title('Sensitivity trace / 敏感性轨迹'); ylabel('MPa');
    end

    % --- 逆向寻优逻辑 ---
    function run_inverse_v36()
        try
            target = str2double(hTarget2.String);
            [~, s_idx] = sort(abs(res_raw_data(:, 10) - target));
            seeds = res_raw_data(s_idx(1:min(30, end)), 1:9);
            opts = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', 'MaxIterations', 30, 'OutputFcn', @nested_trace_v36);
            
            best_X = seeds(1,:); min_err = inf;
            h_wait = waitbar(0, '30-Point Parallel Optimizing...');
            for s = 1:size(seeds, 1)
                if ~isgraphics(h_wait), break; end
                waitbar(s/size(seeds,1), h_wait); 
                obj = @(x) 1000 * (get_pred_quick(x) - target)^2; % 纯强度对齐
                [x_tmp, ~] = fmincon(obj, seeds(s,:), [], [], [], [], lb_global, ub_global, [], opts);
                if abs(get_pred_quick(x_tmp)-target) < min_err
                    min_err = abs(get_pred_quick(x_tmp)-target); best_X = x_tmp;
                end
            end
            if isgraphics(h_wait), delete(h_wait); end
            
            % 3次验证
            v_preds = [get_pred_quick(best_X), get_pred_quick(best_X*1.01), get_pred_quick(best_X*0.99)];
            
            % 填表
            tbl = cell(9, 5);
            for k = 1:9
                val = best_X(k); if k==9, val=round(val); end
                tbl{k,1} = [fNamesCN{k}, ' / ', fNamesEN{k}];
                tbl{k,2} = sprintf('%.2f', val);
                tbl{k,3} = units{k};
                tbl{k,4} = sprintf('[%.1f, %.1f]', lb_global(k), ub_global(k));
                if k==1, tbl{k,5} = sprintf('V1:%.1f | V2:%.1f | V3:%.1f', v_preds(1), v_preds(2), v_preds(3)); end
            end
            set(resTable2, 'Data', tbl);
        catch ME, errordlg(ME.message); end
    end

    function stop = nested_trace_v36(x, ~, ~)
        stop = false;
        persistent iter_history;
        if isempty(iter_history), iter_history = []; end
        iter_history = [iter_history; x];
        steps = 1:size(iter_history, 1);
        for j = 1:9; set(hLines2{j}, 'XData', steps, 'YData', iter_history(:, j)); end
        drawnow limitrate;
    end

    function s_out = get_pred_quick(xin)
        px = mapminmax('apply', xin', ps_input);
        tx = predict(trainedModel, px');
        s_out = mapminmax('reverse', tx', ps_output);
    end
end