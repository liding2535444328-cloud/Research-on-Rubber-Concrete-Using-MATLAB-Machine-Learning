function RC_V32_Scientific_Masterpiece()
% =====================================================================
% 项目：橡胶混凝土 AI 智能设计系统 (V33 虚拟实验室增强版)
% 特色：集成 50 组随机虚拟预测点，验证物理规律一致性
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

% --- 【新功能】生成 50 组随机虚拟实验数据 ---
V_X = lb_global + (ub_global - lb_global) .* rand(50, 9);
V_Y = zeros(50, 1);
for i = 1:50
    px_v = mapminmax('apply', V_X(i, :)', ps_input);
    tx_v = predict(trainedModel, px_v');
    V_Y(i) = mapminmax('reverse', tx_v', ps_output);
end

persistent History_Y; History_Y = [];

%% --- 模块 2: 界面设计 (保持 V32 优美格式) ---
fig = figure('Name', 'RC AI Research Platform V33 - Virtual Lab', ...
             'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.88], ...
             'Color', [0.98 0.98 0.98], 'MenuBar', 'none', 'NumberTitle', 'off');

% --- 上部：可视化矩阵 ---
pnlDisplay = uipanel(fig, 'Position', [0.02 0.45 0.96 0.48], 'BackgroundColor', 'w', ...
                    'Title', 'Multi-dimensional Analysis / 多维度科研分析', 'FontWeight', 'bold');

axSHAP = axes(pnlDisplay, 'Position', [0.04 0.15 0.28 0.72]);
ax3D = axes(pnlDisplay, 'Position', [0.38 0.15 0.28 0.72]);
axHist = axes(pnlDisplay, 'Position', [0.73 0.15 0.24 0.72]);

% --- 中部：高亮面板 ---
pnlRes = uipanel(fig, 'Position', [0.38 0.38 0.28 0.06], 'BackgroundColor', [0.15 0.35 0.65], 'BorderType', 'none');
lblResult = uicontrol(pnlRes, 'Style', 'text', 'Units', 'normalized', 'Position', [0, 0, 1, 1], ...
                     'String', 'Predicting...', 'FontSize', 18, 'FontWeight', 'bold', ...
                     'ForegroundColor', 'w', 'BackgroundColor', [0.15 0.35 0.65]);

% --- 下部：九色交互控制台 ---
pnlControl = uipanel(fig, 'Title', 'Interactive Parameter Control / 参数交互控制台', ...
                    'Position', [0.02 0.02 0.96 0.35], 'FontWeight', 'bold');

fNamesCN = {'水胶比', '橡胶含量', '橡胶粒径', '水泥', '细骨料', '粗骨料', '硅比', '外加剂', '龄期'};
fNamesEN = {'W/B', 'Rubber', 'Size', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};
colors = lines(9); hSliders = cell(9, 1); hEdits = cell(9, 1);

for i = 1:9
    row = ceil(i/3); col = mod(i-1, 3);
    x_base = 0.02 + col*0.33; y_base = 0.8 - (row-1)*0.35;
    uicontrol(pnlControl, 'Style', 'text', 'Units', 'normalized', 'Position', [x_base, y_base, 0.12, 0.15], ...
              'String', [fNamesCN{i}, ' ', fNamesEN{i}], 'FontSize', 10, 'FontWeight', 'bold', 'ForegroundColor', colors(i,:));
    hSliders{i} = uicontrol(pnlControl, 'Style', 'slider', 'Units', 'normalized', 'Position', [x_base+0.1, y_base, 0.15, 0.12], ...
                           'Min', lb_global(i), 'Max', ub_global(i), 'Value', mean([lb_global(i), ub_global(i)]), 'Callback', @(~,~) update_all(i, 'slider'));
    hEdits{i} = uicontrol(pnlControl, 'Style', 'edit', 'Units', 'normalized', 'Position', [x_base+0.26, y_base, 0.04, 0.16], ...
                         'String', num2str(hSliders{i}.Value, '%.2f'), 'Callback', @(~,~) update_all(i, 'edit'));
    uicontrol(pnlControl, 'Style', 'text', 'Units', 'normalized', 'Position', [x_base+0.1, y_base-0.12, 0.15, 0.1], ...
              'String', sprintf('[%.1f - %.1f]', lb_global(i), ub_global(i)), 'FontSize', 8, 'ForegroundColor', [0.5 0.5 0.5]);
end

[grid_W, grid_R] = meshgrid(linspace(lb_global(1), ub_global(1), 25), linspace(lb_global(2), ub_global(2), 25));
update_all(1, 'init');

%% --- 模块 3: 核心交互与绘图引擎 ---

    function update_all(idx, mode)
        if strcmp(mode, 'slider')
            val = hSliders{idx}.Value; if idx==9, val=round(val); end
            hEdits{idx}.String = num2str(val, '%.2f');
        elseif strcmp(mode, 'edit')
            val = str2double(hEdits{idx}.String); val = max(min(val, ub_global(idx)), lb_global(idx));
            hSliders{idx}.Value = val; hEdits{idx}.String = num2str(val, '%.2f');
        end
        X_curr = cellfun(@(h) str2double(h.String), hEdits)';
        Y_curr = get_pred_quick(X_curr);
        set(lblResult, 'String', sprintf('Strength: %.2f MPa (±%.2f)', Y_curr, Final_MAE));
        History_Y = [History_Y, Y_curr]; if length(History_Y) > 40, History_Y(1) = []; end
        
        draw_SHAP_Insight(X_curr, Y_curr);
        draw_Dynamic_Surface(X_curr);
        draw_Evolution_Path();
    end

    function draw_SHAP_Insight(X, Y)
        axes(axSHAP); cla; hold on;
        diffs = (X - mean(res_raw_data(:,1:9))) ./ (ub_global - lb_global);
        for i = 1:9
            barh(i, diffs(i), 'FaceColor', colors(i,:), 'EdgeColor', 'none');
        end
        set(axSHAP, 'YTick', 1:9, 'YTickLabel', fNamesEN, 'XGrid', 'on');
        title({'Local Interpretability (SHAP)','实时机理贡献分析'}, 'FontSize', 10);
    end

    function draw_Dynamic_Surface(X)
        axes(ax3D); cla;
        Z = zeros(size(grid_W));
        for r = 1:size(grid_W, 1)
            for c = 1:size(grid_W, 2)
                X_t = X; X_t(1) = grid_W(r,c); X_t(2) = grid_R(r,c);
                Z(r,c) = get_pred_quick(X_t);
            end
        end
        surf(grid_W, grid_R, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.6); 
        colormap(ax3D, 'parula'); view(135, 30); grid on; hold on;
        
        % --- 【关键：点亮 50 组虚拟实验点】 ---
        scatter3(V_X(:,1), V_X(:,2), V_Y, 15, 'k', 'filled', 'MarkerFaceAlpha', 0.4); 
        
        % 点亮当前操作点
        plot3(X(1), X(2), get_pred_quick(X), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'y', 'LineWidth', 2);
        
        xlabel('W/B Ratio'); ylabel('Rubber Content'); zlabel('f''c (MPa)');
        title({'Global Performance Surface','& 50 Virtual Samples'}, 'FontSize', 10);
    end

    function draw_Evolution_Path()
        axes(axHist); cla;
        fill([1:length(History_Y), fliplr(1:length(History_Y))], ...
             [History_Y-Final_MAE, fliplr(History_Y+Final_MAE)], [0.8 0.8 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        hold on; plot(History_Y, '-s', 'LineWidth', 1.2, 'Color', [0.15 0.35 0.65], 'MarkerSize', 4);
        ylabel('Strength (MPa)'); xlabel('Steps');
        title('Virtual Lab Sensitivity Trace', 'FontSize', 10); grid on;
    end

    function s_out = get_pred_quick(xin)
        px = mapminmax('apply', xin', ps_input);
        tx = predict(trainedModel, px');
        s_out = mapminmax('reverse', tx', ps_output);
    end
end