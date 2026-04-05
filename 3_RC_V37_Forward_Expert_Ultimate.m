function RC_V37_Expert_Forward_Only()
% =====================================================================
% 项目：橡胶混凝土 AI 智能设计系统 (V37 纯净正向专家版)
% 核心：[左] SHAP机理 + [中] 星簇监控 + [右] 敏感性轨迹
% 修改说明：移除所有逆向寻优模块，优化单页面显示性能
% =====================================================================

%% --- 模块 1: 环境初始化与数据加载 ---
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
    errordlg(['Init Error: ', ME.message]); return;
end
persistent History_Y; History_Y = [];

%% --- 模块 2: 扁平化界面设计 (已移除选项卡) ---
fig = figure('Name', 'RC AI Research Platform V37 - Forward Expert Mode', ...
             'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.88], ...
             'Color', [0.98 0.98 0.98], 'MenuBar', 'none', 'NumberTitle', 'off');

fNamesCN = {'水胶比', '橡胶含量', '橡胶粒径', '水泥', '细骨料', '粗骨料', '硅比', '外加剂', '龄期'};
fNamesEN = {'W/B', 'Rubber', 'Size', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};
colors = lines(9);

% --- 上部：可视化矩阵 (直接挂载在 fig) ---
pnlDisp1 = uipanel(fig, 'Position', [0.02 0.45 0.96 0.5], 'BackgroundColor', 'w', ...
                    'Title', 'Multi-dimensional Star-Cluster Analysis / 多维度科研分析', 'FontWeight', 'bold');
axSHAP = axes(pnlDisp1, 'Position', [0.04 0.15 0.25 0.72]); 
ax3D = axes(pnlDisp1, 'Position', [0.35 0.1 0.35 0.85]);    
axHist = axes(pnlDisp1, 'Position', [0.73 0.15 0.24 0.72]);  

% --- 中部：高亮结果面板 ---
pnlRes1 = uipanel(fig, 'Position', [0.38 0.38 0.28 0.06], 'BackgroundColor', [0.15 0.35 0.65], 'BorderType', 'none');
lblResult1 = uicontrol(pnlRes1, 'Style', 'text', 'Units', 'normalized', 'Position', [0, 0, 1, 1], ...
                      'String', 'Ready', 'FontSize', 18, 'FontWeight', 'bold', 'ForegroundColor', 'w', 'BackgroundColor', [0.15 0.35 0.65]);

% --- 下部：参数控制台 ---
pnlCtrl1 = uipanel(fig, 'Title', 'Parameter Control / 参数控制台', 'Position', [0.02 0.02 0.96 0.35], 'FontWeight', 'bold');
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

update_forward(1, 'init');

%% --- 模块 3: 核心逻辑函数 ---
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
        
        draw_SHAP_V33(X_curr, Y_curr); 
        draw_StarCluster_V35(X_curr, Y_curr); 
        draw_History_V35(); 
        drawnow;
    end

    function draw_SHAP_V33(X, Y)
        axes(axSHAP); cla; hold on;
        diffs = (X - mean(res_raw_data(:,1:9))) ./ (ub_global - lb_global);
        for i = 1:9
            barh(i, diffs(i), 'FaceColor', colors(i,:), 'EdgeColor', 'none');
        end
        set(axSHAP, 'YTick', 1:9, 'YTickLabel', fNamesEN, 'XGrid', 'on', 'YDir', 'reverse');
        title({'Local Interpretability (SHAP)','实时机理贡献分析'}, 'FontSize', 10);
    end

    function draw_StarCluster_V35(X, Y)
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
        text(0, 0, Y+5, [num2str(Y, '%.1f'), ' MPa'], 'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.15 0.35 0.65], 'HorizontalAlignment', 'center');
        view(35, 25); grid on; zlim([0 85]); set(gca, 'XTick', [], 'YTick', []);
        title('Star-Cluster Dynamic Monitor / 星簇实时监控');
    end

    function draw_History_V35()
        axes(axHist); cla;
        plot(History_Y, '-o', 'LineWidth', 1.2, 'Color', [0.15 0.35 0.65], 'MarkerSize', 3);
        grid on; title('Sensitivity trace / 敏感性轨迹'); ylabel('MPa');
    end

    function s_out = get_pred_quick(xin)
        px = mapminmax('apply', xin', ps_input);
        tx = predict(trainedModel, px');
        s_out = mapminmax('reverse', tx', ps_output);
    end
end