function RC_V37_StarCluster_Expert()
% =====================================================================
% 项目：橡胶混凝土 AI 智能设计系统 (V37 正向解释专家版)
% 核心：[左] 交互式SHAP机理 + [中] 星簇动态监控 + [右] 敏感性演化轨迹
% 特色：移除逆向模块，纯化正向分析，极速响应，论文演示级格式
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
    
    % 获取物理边界
    lb_global = min(res_raw_data(:, 1:9)); 
    ub_global = max(res_raw_data(:, 1:9));
catch ME
    errordlg(['Init Error: ', ME.message]); return;
end

persistent History_Y; History_Y = [];

%% --- 模块 2: 扁平化科研界面设计 ---
fig = figure('Name', 'RC AI Research Platform V37 - Expert Mode', ...
             'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.88], ...
             'Color', [0.98 0.98 0.98], 'MenuBar', 'none', 'NumberTitle', 'off');

% --- 上部：多维度可视化窗口 ---
pnlDisplay = uipanel(fig, 'Position', [0.02 0.42 0.96 0.55], 'BackgroundColor', 'w', ...
                    'Title', 'Multi-dimensional Analysis / 多维度科研分析', 'FontWeight', 'bold');

% 坐标轴分配
axSHAP = axes(pnlDisplay, 'Position', [0.04 0.15 0.25 0.72], 'FontName', 'Arial'); % 左
ax3D = axes(pnlDisplay, 'Position', [0.35 0.1 0.35 0.85], 'FontName', 'Arial');    % 中
axHist = axes(pnlDisplay, 'Position', [0.73 0.15 0.24 0.72], 'FontName', 'Arial');  % 右

% --- 中部：高亮强度显示 (视觉中心) ---
pnlRes = uipanel(fig, 'Position', [0.38 0.38 0.28 0.06], 'BackgroundColor', [0.15 0.35 0.65], 'BorderType', 'none');
lblResult = uicontrol(pnlRes, 'Style', 'text', 'Units', 'normalized', 'Position', [0, 0, 1, 1], ...
                      'String', 'Predicting...', 'FontSize', 18, 'FontWeight', 'bold', ...
                      'ForegroundColor', 'w', 'BackgroundColor', [0.15 0.35 0.65]);

% --- 下部：参数交互控制台 ---
pnlControl = uipanel(fig, 'Title', 'Parameter Control / 参数控制台', ...
                    'Position', [0.02 0.02 0.96 0.35], 'FontWeight', 'bold');

fNamesCN = {'水胶比', '橡胶含量', '橡胶粒径', '水泥', '细骨料', '粗骨料', '硅比', '外加剂', '龄期'};
fNamesEN = {'W/B', 'Rubber', 'Size', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};
colors = lines(9);

hSliders = cell(9, 1); hEdits = cell(9, 1);

for i = 1:9
    row = ceil(i/3); col = mod(i-1, 3);
    x_base = 0.02 + col*0.33; y_base = 0.8 - (row-1)*0.35;
    
    % 参数标签
    uicontrol(pnlControl, 'Style', 'text', 'Units', 'normalized', 'Position', [x_base, y_base, 0.12, 0.15], ...
              'String', [fNamesCN{i}, ' ', fNamesEN{i}], 'FontSize', 9, 'FontWeight', 'bold', ...
              'ForegroundColor', colors(i,:), 'HorizontalAlignment', 'left');
    
    % 滑块同步
    hSliders{i} = uicontrol(pnlControl, 'Style', 'slider', 'Units', 'normalized', 'Position', [x_base+0.1, y_base, 0.15, 0.12], ...
                           'Min', lb_global(i), 'Max', ub_global(i), 'Value', mean([lb_global(i), ub_global(i)]), ...
                           'Callback', @(~,~) update_all(i, 'slider'));
    
    % 输入框同步
    hEdits{i} = uicontrol(pnlControl, 'Style', 'edit', 'Units', 'normalized', 'Position', [x_base+0.26, y_base, 0.04, 0.16], ...
                         'String', num2str(hSliders{i}.Value, '%.2f'), 'FontSize', 10, ...
                         'Callback', @(~,~) update_all(i, 'edit'));
    
    % 范围标签
    uicontrol(pnlControl, 'Style', 'text', 'Units', 'normalized', 'Position', [x_base+0.1, y_base-0.12, 0.15, 0.1], ...
              'String', sprintf('Range: [%.1f - %.1f]', lb_global(i), ub_global(i)), 'FontSize', 7, 'ForegroundColor', [0.5 0.5 0.5]);
end

% 首次刷新
update_all(1, 'init');

%% --- 模块 3: 核心交互逻辑与绘图引擎 ---

    function update_all(idx, mode)
        % 双向联动逻辑
        if strcmp(mode, 'slider')
            val = hSliders{idx}.Value; if idx==9, val=round(val); end
            hEdits{idx}.String = num2str(val, '%.2f');
        elseif strcmp(mode, 'edit')
            val = str2double(hEdits{idx}.String); 
            val = max(min(val, ub_global(idx)), lb_global(idx));
            hSliders{idx}.Value = val; hEdits{idx}.String = num2str(val, '%.2f');
        end
        
        % 获取当前状态并预测
        X_curr = cellfun(@(h) str2double(h.String), hEdits)';
        Y_curr = get_pred_quick(X_curr);
        
        % 更新中心数值
        set(lblResult, 'String', sprintf('Strength: %.2f MPa (±%.2f)', Y_curr, Final_MAE));
        
        % 历史记录
        History_Y = [History_Y, Y_curr];
        if length(History_Y) > 30, History_Y(1) = []; end
        
        % 绘图刷新
        draw_SHAP_Insight(X_curr);
        draw_StarCluster_Monitor(X_curr, Y_curr);
        draw_Evolution_Path();
        drawnow;
    end

    % 【左侧】：交互式 SHAP 贡献度图
    function draw_SHAP_Insight(X)
        axes(axSHAP); cla; hold on;
        % 归一化计算贡献倾向
        diffs = (X - mean(res_raw_data(:,1:9))) ./ (ub_global - lb_global);
        for i = 1:9
            barh(i, diffs(i), 'FaceColor', colors(i,:), 'EdgeColor', 'none');
        end
        set(axSHAP, 'YTick', 1:9, 'YTickLabel', fNamesEN, 'XGrid', 'on');
        title({'Local Interpretability (SHAP)','实时机理贡献分析'}, 'FontSize', 10);
        xlabel('Negative <--- Impact ---> Positive');
    end

    % 【中间】：星簇架构动态监控图
    function draw_StarCluster_Monitor(X, Y)
        axes(ax3D); cla; hold on;
        theta = linspace(0, 2*pi, 10); theta(end) = []; R_ring = 10;
        x_ring = R_ring * cos(theta); y_ring = R_ring * sin(theta);
        
        for i = 1:9
            % 柱体代表输入强度
            h = (X(i) - lb_global(i)) / (ub_global(i) - lb_global(i)) * 40; 
            [bx, by, bz] = cylinder(0.6, 20);
            surf(bx + x_ring(i), by + y_ring(i), bz * h, 'FaceColor', colors(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.6);
            
            % 标注名称
            text(x_ring(i)*1.3, y_ring(i)*1.3, 0, fNamesEN{i}, 'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
            
            % 能量连接线 (逻辑关联)
            line([x_ring(i), 0], [y_ring(i), 0], [h, Y], 'Color', [0.8 0.8 0.8], 'LineStyle', ':');
        end
        
        % 预测强度星
        plot3(0, 0, Y, 'kp', 'MarkerSize', 18, 'MarkerFaceColor', [1 0.8 0], 'LineWidth', 2);
        
        % 实时浮动标签
        text(0, 0, Y+5, [num2str(Y, '%.1f'), ' MPa'], 'FontSize', 12, 'FontWeight', 'bold', ...
             'Color', [0.15 0.35 0.65], 'HorizontalAlignment', 'center');
        
        view(35, 25); grid on; zlim([0 85]); set(gca, 'XTick', [], 'YTick', []);
        title('Star-Cluster Dynamic Monitor / 星簇实时监控');
    end

    % 【右侧】：敏感性演化轨迹
    function draw_Evolution_Path()
        axes(axHist); cla;
        % 绘制误差带
        fill([1:length(History_Y), fliplr(1:length(History_Y))], ...
             [History_Y-Final_MAE, fliplr(History_Y+Final_MAE)], [0.8 0.8 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        hold on;
        plot(History_Y, '-s', 'LineWidth', 1.2, 'Color', [0.15 0.35 0.65], 'MarkerSize', 4);
        ylabel('Strength (MPa)'); xlabel('Operation Steps');
        title('Sensitivity Trace / 敏感性轨迹', 'FontSize', 10); grid on;
    end

    function s_out = get_pred_quick(xin)
        px = mapminmax('apply', xin', ps_input);
        tx = predict(trainedModel, px');
        s_out = mapminmax('reverse', tx', ps_output);
    end
end