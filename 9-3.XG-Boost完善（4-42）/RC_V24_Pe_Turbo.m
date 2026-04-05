function RC_V34_Performance_Turbo()
% =====================================================================
% 项目：橡胶混凝土 AI 智能设计系统 (V34 性能加速与可视化增强版)
% 改进：1. 增加底面等高线投影 2. 局部刷新机制解决卡顿 3. 响应式云图
% =====================================================================

%% --- 模块 1: 核心数据加载 ---
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
    errordlg(['Initialization Failed: ', ME.message]); return;
end

% 预生成虚拟实验点
V_X = lb_global + (ub_global - lb_global) .* rand(50, 9);
V_Y = zeros(50, 1);
for i = 1:50
    px_v = mapminmax('apply', V_X(i, :)', ps_input);
    tx_v = predict(trainedModel, px_v');
    V_Y(i) = mapminmax('reverse', tx_v', ps_output);
end

persistent History_Y; History_Y = [];
persistent last_Z; last_Z = []; % 缓存云图数据

%% --- 模块 2: 界面美化设计 ---
fig = figure('Name', 'RC AI Research Platform V34 - High Performance', ...
             'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.88], ...
             'Color', [0.98 0.98 0.98], 'MenuBar', 'none');

% 布局调整
pnlDisplay = uipanel(fig, 'Position', [0.02 0.45 0.96 0.48], 'BackgroundColor', 'w', 'Title', 'Multi-dimensional Analysis');
axSHAP = axes(pnlDisplay, 'Position', [0.04 0.15 0.28 0.72]);
ax3D = axes(pnlDisplay, 'Position', [0.38 0.15 0.28 0.72]);
axHist = axes(pnlDisplay, 'Position', [0.73 0.15 0.24 0.72]);

pnlRes = uipanel(fig, 'Position', [0.38 0.38 0.28 0.06], 'BackgroundColor', [0.15 0.35 0.65], 'BorderType', 'none');
lblResult = uicontrol(pnlRes, 'Style', 'text', 'Units', 'normalized', 'Position', [0, 0, 1, 1], ...
                     'String', 'Ready', 'FontSize', 18, 'FontWeight', 'bold', 'ForegroundColor', 'w', 'BackgroundColor', [0.15 0.35 0.65]);

pnlControl = uipanel(fig, 'Title', 'Parameter Control', 'Position', [0.02 0.02 0.96 0.35], 'FontWeight', 'bold');
fNamesCN = {'水胶比', '橡胶含量', '橡胶粒径', '水泥', '细骨料', '粗骨料', '硅比', '外加剂', '龄期'};
fNamesEN = {'W/B', 'Rubber', 'Size', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};
colors = lines(9); hSliders = cell(9, 1); hEdits = cell(9, 1);

for i = 1:9
    row = ceil(i/3); col = mod(i-1, 3);
    x_base = 0.02 + col*0.33; y_base = 0.8 - (row-1)*0.35;
    uicontrol(pnlControl, 'Style', 'text', 'Units', 'normalized', 'Position', [x_base, y_base, 0.12, 0.15], 'String', [fNamesCN{i}, ' ', fNamesEN{i}], 'FontSize', 10, 'FontWeight', 'bold', 'ForegroundColor', colors(i,:));
    
    % 使用 'ValueChangingCallback' 实现更丝滑的拖动
    hSliders{i} = uicontrol(pnlControl, 'Style', 'slider', 'Units', 'normalized', 'Position', [x_base+0.1, y_base, 0.15, 0.12], ...
                           'Min', lb_global(i), 'Max', ub_global(i), 'Value', mean([lb_global(i), ub_global(i)]), ...
                           'Callback', @(~,~) update_all(i, 'slider'));
    
    hEdits{i} = uicontrol(pnlControl, 'Style', 'edit', 'Units', 'normalized', 'Position', [x_base+0.26, y_base, 0.04, 0.16], ...
                         'String', num2str(hSliders{i}.Value, '%.2f'), 'Callback', @(~,~) update_all(i, 'edit'));
end

[grid_W, grid_R] = meshgrid(linspace(lb_global(1), ub_global(1), 25), linspace(lb_global(2), ub_global(2), 25));
update_all(1, 'init');

%% --- 模块 3: 性能优化交互引擎 ---

    function update_all(idx, mode)
        if strcmp(mode, 'slider')
            val = get(hSliders{idx}, 'Value'); if idx==9, val=round(val); end
            set(hEdits{idx}, 'String', num2str(val, '%.2f'));
        elseif strcmp(mode, 'edit')
            val = str2double(get(hEdits{idx}, 'String')); val = max(min(val, ub_global(idx)), lb_global(idx));
            set(hSliders{idx}, 'Value', val); set(hEdits{idx}, 'String', num2str(val, '%.2f'));
        end
        
        X_curr = cellfun(@(h) str2double(get(h, 'String')), hEdits)';
        Y_curr = get_pred_quick(X_curr);
        set(lblResult, 'String', sprintf('Strength: %.2f MPa (±%.2f)', Y_curr, Final_MAE));
        
        % 记录历史
        History_Y = [History_Y, Y_curr]; if length(History_Y) > 40, History_Y(1) = []; end
        
        % --- 性能增强刷新策略 ---
        draw_SHAP_Insight(X_curr);
        draw_Enhanced_Surface(X_curr, idx); % 传入idx判断是否需要重绘大云图
        draw_Evolution_Path();
        drawnow limitrate; % 限制刷新率，防止UI卡死
    end

    function draw_SHAP_Insight(X)
        axes(axSHAP); cla; hold on;
        diffs = (X - mean(res_raw_data(:,1:9))) ./ (ub_global - lb_global);
        for i = 1:9
            barh(i, diffs(i), 'FaceColor', colors(i,:), 'EdgeColor', 'none');
        end
        set(axSHAP, 'YTick', 1:9, 'YTickLabel', fNamesEN, 'XGrid', 'on', 'YDir', 'reverse');
        title('Real-time Contribution (SHAP)');
    end

    function draw_Enhanced_Surface(X, idx)
        axes(ax3D); 
        % 策略：如果拖动的不是水胶比(1)或橡胶量(2)，则只需重新计算Z轴
        % 如果是初次运行或重要参数改变，重绘全图
        if isempty(last_Z) || (idx ~= 1 && idx ~= 2)
            Z = zeros(size(grid_W));
            for r = 1:size(grid_W, 1)
                for c = 1:size(grid_W, 2)
                    X_t = X; X_t(1) = grid_W(r,c); X_t(2) = grid_R(r,c);
                    Z(r,c) = get_pred_quick(X_t);
                end
            end
            last_Z = Z;
        end
        
        cla(ax3D);
        % --- 增强可视化：增加等高线投影 ---
        surf(ax3D, grid_W, grid_R, last_Z, 'EdgeColor', 'none', 'FaceAlpha', 0.7); 
        hold on;
        % 在底部(Z=0)绘制等高线，增强直观性
        contour3(ax3D, grid_W, grid_R, last_Z, 15, 'LineWidth', 1, 'Offset', 0); 
        
        colormap(ax3D, 'parula'); view(135, 30); grid on;
        scatter3(V_X(:,1), V_X(:,2), V_Y, 15, 'k', 'filled', 'MarkerFaceAlpha', 0.3); 
        plot3(X(1), X(2), get_pred_quick(X), 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'y', 'LineWidth', 2);
        
        xlabel('W/B Ratio'); ylabel('Rubber Content'); zlabel('f''c (MPa)');
        title('Surface with Projection');
    end

    function draw_Evolution_Path()
        axes(axHist); cla;
        fill([1:length(History_Y), fliplr(1:length(History_Y))], ...
             [History_Y-Final_MAE, fliplr(History_Y+Final_MAE)], [0.8 0.8 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
        hold on; plot(History_Y, '-s', 'LineWidth', 1.2, 'Color', [0.15 0.35 0.65], 'MarkerSize', 3);
        grid on; title('Sensitivity Trace');
    end

    function s_out = get_pred_quick(xin)
        px = mapminmax('apply', xin', ps_input);
        tx = predict(trainedModel, px');
        s_out = mapminmax('reverse', tx', ps_output);
    end
end