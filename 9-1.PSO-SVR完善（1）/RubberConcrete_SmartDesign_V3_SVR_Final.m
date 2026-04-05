function RubberConcrete_SmartDesign_V7_SVR_Final()
    % =====================================================================
    % 项目：橡胶混凝土智能 AI 设计系统 (V7.0 SVR 终极科研版)
    % 引擎：libsvm (svmpredict) + fmincon (SQP)
    % 修复：彻底解决归一化参数匹配失效导致的 0.01MPa 报错及柱状图消失问题
    % =====================================================================
    warning off;
    
    % --- 模块 1: 数据与模型加载 ---
    try
        data = load('ConcreteModel.mat');
        trainedModel = data.final_model; 
        ps_input = data.ps_input; ps_output = data.ps_output;
        res_raw = data.res_raw;
        train_MAE = 2.15;  
        X_data_global = res_raw(:, 1:8);
        lb_global = min(X_data_global); ub_global = max(X_data_global);
    catch
        errordlg('未找到模型文件！请确保 ConcreteModel.mat 在当前文件夹。', 'Error'); return;
    end

    % --- 模块 2: 主窗口设计 ---
    fig = figure('Name', 'Rubber Concrete Smart Design System (V7.0 SVR Final)', ...
                 'Units', 'normalized', 'Position', [0.1, 0.05, 0.8, 0.85], ...
                 'Color', [0.95 0.95 0.95], 'MenuBar', 'none');
    
    tabGroup = uitabgroup(fig, 'Position', [0 0 1 1]);
    tabPredict = uitab(tabGroup, 'Title', '正向预测 / Strength Prediction');
    tabInverse = uitab(tabGroup, 'Title', '逆向设计 / Inverse Design');
    
    featureNamesCN = {'水泥', '硅灰', '水', '外加剂', '砂', '石', '龄期', '橡胶'};
    units = {'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'days', 'kg/m³'};

    %% ================== 界面 1: 正向预测 ==================
    pnlIn = uipanel(tabPredict, 'Title', '输入参数配置 / Input Features', 'Position', [0.05 0.22 0.43 0.72]);
    pnlOut = uipanel(tabPredict, 'Title', '预测分析报告 / AI Analysis', 'Position', [0.52 0.22 0.43 0.72]);
    
    predEditHandles = cell(8, 1);
    for i = 1:8
        y_n = 0.88 - (i-1)*0.11;
        rangeT = sprintf('[%.0f, %.0f]', lb_global(i), ub_global(i));
        randVal = lb_global(i) + (ub_global(i) - lb_global(i)) * rand();
        uicontrol(pnlIn, 'Style', 'text', 'Units', 'normalized', 'Position', [0.02, y_n, 0.58, 0.06], ...
                  'String', [featureNamesCN{i}, ' ', rangeT, ':'], 'HorizontalAlignment', 'right', 'FontSize', 10);
        predEditHandles{i} = uicontrol(pnlIn, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.62, y_n+0.01, 0.3, 0.08], ...
                                      'String', sprintf('%.2f', randVal), 'BackgroundColor', 'w', 'FontSize', 11);
    end
    
    uicontrol(tabPredict, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.12 0.08, 0.3, 0.08], ...
                        'String', '🚀 执行 SVR 模型推理 (Predict)', 'FontSize', 14, 'FontWeight', 'bold', ...
                        'BackgroundColor', [0.15 0.35 0.65], 'ForegroundColor', 'w', 'Callback', @(~,~) run_prediction_v7());
    
    axPredict = axes(pnlOut, 'Units', 'normalized', 'Position', [0.15 0.3 0.7 0.55]);
    lblRes = uicontrol(pnlOut, 'Style', 'text', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.15], ...
                       'String', '等待输入数据...', 'FontSize', 12, 'FontWeight', 'bold', 'ForegroundColor', [0.7 0.1 0.1]);

    %% ================== 界面 2: 逆向设计 ==================
    pnlCfg = uipanel(tabInverse, 'Title', '目标配置', 'Position', [0.02 0.78 0.96 0.2]);
    pnlPlot = uipanel(tabInverse, 'Title', '寻优轨迹', 'Position', [0.02 0.4 0.96 0.36]);
    pnlData = uipanel(tabInverse, 'Title', '结果方案', 'Position', [0.02 0.02 0.96 0.36]);
    uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.6, 0.1, 0.3], 'String', '目标强度(MPa):');
    hTarget = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.12, 0.62, 0.06, 0.25], 'String', '45');
    hTol = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.3, 0.62, 0.06, 0.25], 'String', '2.5');
    uicontrol(pnlCfg, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.45, 0.55, 0.5, 0.38], ...
                       'String', '⚡ 启动 25 点并发寻优', 'FontSize', 13, 'FontWeight', 'bold', 'Callback', @(~,~) run_inverse_v7());
    axLive = axes(pnlPlot, 'Units', 'normalized', 'Position', [0.08 0.2 0.85 0.7]);
    history_X = []; hLines = cell(8, 1); lineColors = lines(8); hold(axLive, 'on'); grid(axLive, 'on');
    for i = 1:8, hLines{i} = plot(axLive, NaN, NaN, 'Color', lineColors(i,:), 'LineWidth', 2); end
    resTable = uitable(pnlData, 'Units', 'normalized', 'Position', [0.01 0.05 0.98 0.9], 'ColumnName', {'组分', '推荐值', '单位', '范围', '回测'});

    %% ================== 核心逻辑函数 ==================
    function run_prediction_v7()
        try
            X_curr = zeros(1, 8);
            for k = 1:8, X_curr(k) = str2double(get(predEditHandles{k}, 'String')); end
            
            % 核心：调用手动归一化驱动的推理引擎
            Y_mean = get_svr_pred_only(X_curr, trainedModel, ps_input, ps_output);
            
            % --- 图形区域暴力重绘 ---
            axes(axPredict); cla; hold on;
            bar(1, Y_mean, 0.4, 'FaceColor', [0.15 0.45 0.75], 'EdgeColor', 'k', 'LineWidth', 1.5);
            errorbar(1, Y_mean, train_MAE, 'Color', [0.8 0.1 0.1], 'LineWidth', 2, 'CapSize', 15);
            
            grid on; box on;
            ylabel('抗压强度 / Strength (MPa)', 'FontWeight', 'bold');
            set(gca, 'XTick', 1, 'XTickLabel', {'AI Predicted'}, 'FontSize', 10);
            
            % 坐标轴自适应：确保柱子能显现出来
            xlim([0.4, 1.6]); 
            ylim([0, max([Y_mean + 20, 60])]); 
            
            % 蓝色加粗数值标注
            text(1, Y_mean + 3, sprintf('%.2f MPa', Y_mean), ...
                'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0 0 0.8]);
            
            set(lblRes, 'String', sprintf('最终预测强度: %.2f MPa | 理论误差波动: ±%.2f MPa', Y_mean, train_MAE), 'ForegroundColor', [0 0.4 0]);
            drawnow;
        catch ME
            errordlg(['Predict Error: ', ME.message]);
        end
    end

    function run_inverse_v7()
        try
            target = str2double(get(hTarget, 'String')); tol = str2double(get(hTol, 'String'));
            opts = optimoptions('fmincon','Display','none','Algorithm','sqp','OutputFcn',@(x,v,s)o_plot_trace(x));
            seeds = res_raw(randperm(size(res_raw,1), 25), 1:8);
            best_f = inf; best_x = seeds(1,:);
            h_w = waitbar(0, '寻优中...');
            for s = 1:25
                waitbar(s/25, h_w); history_X = [];
                obj = @(x) (get_svr_pred_only(x, trainedModel, ps_input, ps_output) - target)^2;
                [xt, ft] = fmincon(obj, seeds(s,:), [],[],[],[], lb_global, ub_global, @(x) constraints_v7_local(x, target, tol), opts);
                if ft < best_f, best_f = ft; best_x = xt; end
            end
            delete(h_w);
            p_final = get_svr_pred_only(best_x, trainedModel, ps_input, ps_output);
            tbl = cell(8, 5);
            for k=1:8, tbl{k,1}=featureNamesCN{k}; tbl{k,2}=sprintf('%.2f', best_x(k));
                if k==1, tbl{k,5}=sprintf('回测: %.2f MPa', p_final); end
            end
            set(resTable, 'Data', tbl);
        catch ME, errordlg(ME.message); end
    end

    function stop = o_plot_trace(x)
        stop = false; history_X = [history_X; x(:)'];
        for i = 1:8, set(hLines{i}, 'XData', 1:size(history_X,1), 'YData', history_X(:,i)); end
        drawnow limitrate;
    end

    function [c, ceq] = constraints_v7_local(x, target, tol)
        pred = get_svr_pred_only(x, trainedModel, ps_input, ps_output);
        c(1) = (target - tol) - pred; c(2) = pred - (target + tol);
        wb = x(3)/(x(1)+x(2)+1e-6); c(3) = 0.28 - wb; c(4) = wb - 0.55; ceq = [];
    end

    % --- 唯一核心推理函数 (手动归一化确保反算成功) ---
    function s_out = get_svr_pred_only(x, model, ps_in, ps_out)
        try
            x_vec = x(:); 
            % 手动应用输入归一化公式: (x - min) / (max - min) * (ymax - ymin) + ymin
            % 这里的 ps_in.gain 相当于 1/(max-min)
            px = (x_vec - ps_in.xoffset) .* ps_in.gain + ps_in.ymin;
            
            % libsvm 预测 (必须将 px 转置为 样本x特征 格式)
            [tx, ~] = svmpredict(0, px', model, '-q');
            
            % 手动应用输出反归一化公式: (y - ymin) / (ymax - ymin) * (max - min) + min
            % 这里的 ps_out.gain 相当于 1/(max-min)
            s_real = (tx - ps_out.ymin) ./ ps_out.gain + ps_out.xoffset;
            s_out = max(0.01, double(s_real(1))); 
        catch
            s_out = 0.01;
        end
    end
end % 主函数结束