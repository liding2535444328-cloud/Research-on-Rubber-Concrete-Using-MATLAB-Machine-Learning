function RC_V35_StarCluster_Monitor()
% =====================================================================
% 项目：橡胶混凝土 AI 智能设计系统 (V35 星簇架构动态监控版)
% 核心：9个输入围成环状柱状图 + 中心预测点实时升降
% 优势：极速响应 (无卡顿) + 直观因果对应
% =====================================================================

%% --- 模块 1: 环境加载 ---
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
catch ME
    errordlg(['Init Failed: ', ME.message]); return;
end

persistent History_Y; History_Y = [];

%% --- 模块 2: 界面美化设计 ---
fig = figure('Name', 'RC AI Research Platform V35 - StarCluster', ...
             'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.88], ...
             'Color', [0.98 0.98 0.98], 'MenuBar', 'none');

% 展示区
pnlDisplay = uipanel(fig, 'Position', [0.02 0.45 0.96 0.48], 'BackgroundColor', 'w', 'Title', 'Multi-dimensional Star-Cluster Analysis');
axSHAP = axes(pnlDisplay, 'Position', [0.04 0.15 0.25 0.72]);
ax3D = axes(pnlDisplay, 'Position', [0.35 0.12 0.35 0.8]); % 中心大图
axHist = axes(pnlDisplay, 'Position', [0.73 0.15 0.24 0.72]);

% 结果高亮
pnlRes = uipanel(fig, 'Position', [0.38 0.38 0.28 0.06], 'BackgroundColor', [0.15 0.35 0.65], 'BorderType', 'none');
lblResult = uicontrol(pnlRes, 'Style', 'text', 'Units', 'normalized', 'Position', [0, 0, 1, 1], ...
                     'String', 'Ready', 'FontSize', 18, 'FontWeight', 'bold', 'ForegroundColor', 'w', 'BackgroundColor', [0.15 0.35 0.65]);

% 控制区
pnlControl = uipanel(fig, 'Title', 'Parameter Control', 'Position', [0.02 0.02 0.96 0.35], 'FontWeight', 'bold');
fNamesCN = {'水胶比', '橡胶含量', '橡胶粒径', '水泥', '细骨料', '粗骨料', '硅比', '外加剂', '龄期'};
fNamesEN = {'W/B', 'Rubber', 'Size', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};
colors = lines(9); hSliders = cell(9, 1); hEdits = cell(9, 1);

for i = 1:9
    row = ceil(i/3); col = mod(i-1, 3);
    x_base = 0.02 + col*0.33; y_base = 0.8 - (row-1)*0.35;
    uicontrol(pnlControl, 'Style', 'text', 'Units', 'normalized', 'Position', [x_base, y_base, 0.12, 0.15], 'String', [fNamesCN{i}, ' ', fNamesEN{i}], 'FontSize', 10, 'FontWeight', 'bold', 'ForegroundColor', colors(i,:));
    hSliders{i} = uicontrol(pnlControl, 'Style', 'slider', 'Units', 'normalized', 'Position', [x_base+0.1, y_base, 0.15, 0.12], ...
                           'Min', lb_global(i), 'Max', ub_global(i), 'Value', mean([lb_global(i), ub_global(i)]), ...
                           'Callback', @(~,~) update_all(i, 'slider'));
    hEdits{i} = uicontrol(pnlControl, 'Style', 'edit', 'Units', 'normalized', 'Position', [x_base+0.26, y_base, 0.04, 0.16], ...
                         'String', num2str(hSliders{i}.Value, '%.2f'), 'Callback', @(~,~) update_all(i, 'edit'));
end

update_all(1, 'init');

%% --- 模块 3: 极速星簇引擎 ---

    function update_all(idx, mode)
        if strcmp(mode, 'slider')
            val = get(hSliders{idx}, 'Value'); if idx==9, val=round(val); end
            set(hEdits{idx}, 'String', num2str(val, '%.2f'));
        elseif strcmp(mode, 'edit')
            val = str2double(get(hEdits{idx}, 'String')); val = max(min(val, ub_global(idx)), lb_global(idx));
            set(hSliders{idx}, 'Value', val); set(hEdits{idx}.String, num2str(val, '%.2f'));
        end
        
        X_curr = cellfun(@(h) str2double(get(h, 'String')), hEdits)';
        Y_curr = get_pred_quick(X_curr);
        set(lblResult, 'String', sprintf('Strength: %.2f MPa', Y_curr));
        
        History_Y = [History_Y, Y_curr]; if length(History_Y) > 30, History_Y(1) = []; end
        
        % 刷新绘图
        draw_SHAP_Insight(X_curr);
        draw_StarCluster(X_curr, Y_curr); % 核心新功能
        draw_Evolution_Path();
        drawnow; 
    end

    function draw_StarCluster(X, Y)
        axes(ax3D); cla; hold on;
        
        % 1. 计算 9 个特征的环状坐标 (Radius = 10)
        theta = linspace(0, 2*pi, 10); theta(end) = [];
        R_ring = 10;
        x_ring = R_ring * cos(theta);
        y_ring = R_ring * sin(theta);
        
        % 2. 绘制 9 个输入的“围墙”柱状图
        for i = 1:9
            % 归一化高度以便观察趋势
            h = (X(i) - lb_global(i)) / (ub_global(i) - lb_global(i)) * 50; 
            % 绘制 3D 柱体
            [bx, by, bz] = cylinder(0.8, 20);
            surf(bx + x_ring(i), by + y_ring(i), bz * h, 'FaceColor', colors(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.7);
            % 标注名称
            text(x_ring(i)*1.3, y_ring(i)*1.3, 0, fNamesEN{i}, 'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
        end
        
        % 3. 绘制中心预测点 (强度点)
        % 强度直接对应 Z 轴高度
        z_center = Y; 
        plot3(0, 0, z_center, 'kp', 'MarkerSize', 20, 'MarkerFaceColor', [1 0.8 0], 'LineWidth', 2);
        
        % 4. 绘制“能量连接线” (由柱顶连向中心点，增强视觉联系)
        for i = 1:9
            h_top = (X(i) - lb_global(i)) / (ub_global(i) - lb_global(i)) * 50;
            line([x_ring(i), 0], [y_ring(i), 0], [h_top, z_center], 'Color', [0.8 0.8 0.8], 'LineStyle', '--');
        end
        
        % 5. 实时数值标签 (悬浮在中心点上方)
        text(0, 0, z_center + 5, [num2str(Y, '%.2f'), ' MPa'], 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.15 0.35 0.65], 'HorizontalAlignment', 'center');
        
        % 界面装饰
        grid on; view(30, 25);
        xlim([-15 15]); ylim([-15 15]); zlim([0 80]);
        set(gca, 'XTick', [], 'YTick', []);
        zlabel('Strength / 归一化输入');
        title('Star-Cluster Real-time Framework');
    end

    function draw_SHAP_Insight(X)
        axes(axSHAP); cla;
        diffs = (X - mean(res_raw_data(:,1:9))) ./ (ub_global - lb_global);
        for i = 1:9; barh(i, diffs(i), 'FaceColor', colors(i,:), 'EdgeColor', 'none'); end
        set(axSHAP, 'YTick', 1:9, 'YTickLabel', fNamesEN, 'XGrid', 'on', 'YDir', 'reverse');
    end

    function draw_Evolution_Path()
        axes(axHist); cla;
        plot(History_Y, '-o', 'LineWidth', 1.2, 'Color', [0.15 0.35 0.65], 'MarkerSize', 3);
        grid on; title('Sensitivity Trace');
    end

    function s_out = get_pred_quick(xin)
        px = mapminmax('apply', xin', ps_input);
        tx = predict(trainedModel, px');
        s_out = mapminmax('reverse', tx', ps_output);
    end
end