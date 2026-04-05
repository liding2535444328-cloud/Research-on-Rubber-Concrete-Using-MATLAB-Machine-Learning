function RubberConcrete_SmartDesign_V7_Final()
    % 加载模型与原始数据
    try
        data = load('ConcreteModel.mat');
        trainedModel = data.final_model;
        ps_input = data.ps_input; ps_output = data.ps_output;
        res_raw = data.res_raw;
        train_R2 = 0.9654; train_MAE = 2.15;  
        X_data_global = res_raw(:, 1:8);
        lb_global = min(X_data_global); ub_global = max(X_data_global);
    catch
        errordlg('未找到模型文件！请确保 ConcreteModel.mat 在当前文件夹。', 'Error'); return;
    end

    % 1. 主窗口设计 (响应式单位)
    fig = figure('Name', 'Rubber Concrete Smart Design System (V7.0 Final)', ...
                 'Units', 'normalized', 'Position', [0.1, 0.05, 0.8, 0.85], ...
                 'Color', [0.95 0.95 0.95], 'MenuBar', 'none');

    tabGroup = uitabgroup(fig, 'Position', [0 0 1 1]);
    tabPredict = uitab(tabGroup, 'Title', '正向预测 / Strength Prediction');
    tabInverse = uitab(tabGroup, 'Title', '逆向设计 / Inverse Design');

    featureNamesCN = {'水泥', '硅灰', '水', '外加剂', '砂', '石', '龄期', '橡胶'};
    featureNamesEN = {'Cement', 'SF', 'Water', 'SP', 'Sand', 'Gravel', 'Age', 'Rubber'};
    units = {'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'days', 'kg/m³'};

    %% ================== 界面 1: 正向预测 ==================
    pnlIn = uipanel(tabPredict, 'Title', '输入参数配置 / Input Features', 'Position', [0.05 0.22 0.43 0.72]);
    pnlOut = uipanel(tabPredict, 'Title', '预测分析报告 / AI Analysis', 'Position', [0.52 0.22 0.43 0.72]);

    predEditHandles = cell(8, 1);
    for i = 1:8
        y_n = 0.88 - (i-1)*0.11;
        rangeT = sprintf('[%.1f, %.1f]', lb_global(i), ub_global(i));
        % 初始值在范围内随机生成
        randVal = lb_global(i) + (ub_global(i) - lb_global(i)) * rand();
        
        uicontrol(pnlIn, 'Style', 'text', 'Units', 'normalized', 'Position', [0.02, y_n, 0.58, 0.06], ...
                  'String', [featureNamesCN{i}, ' (', units{i}, ') ', rangeT, ':'], 'HorizontalAlignment', 'right', 'FontSize', 10);
        predEditHandles{i} = uicontrol(pnlIn, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.62, y_n+0.01, 0.3, 0.08], ...
                                      'String', sprintf('%.2f', randVal), 'BackgroundColor', 'w', 'FontSize', 11);
    end

    uicontrol(tabPredict, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.12 0.08, 0.3, 0.08], ...
                        'String', '🚀 执行模型推理 (Predict)', 'FontSize', 14, 'FontWeight', 'bold', ...
                        'BackgroundColor', [0.15 0.35 0.65], 'ForegroundColor', 'w', 'Callback', @(~,~) run_prediction_v7());

    axPredict = axes(pnlOut, 'Units', 'normalized', 'Position', [0.15 0.3 0.7 0.6]);
    lblRes = uicontrol(pnlOut, 'Style', 'text', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.15], ...
                       'String', '等待输入数据...', 'FontSize', 12, 'FontWeight', 'bold', 'ForegroundColor', [0.7 0.1 0.1]);

    %% ================== 界面 2: 逆向设计 ==================
    pnlCfg = uipanel(tabInverse, 'Title', '目标与单价配置 / Config', 'Position', [0.02 0.78 0.96 0.2]);
    pnlPlot = uipanel(tabInverse, 'Title', '动态寻优轨迹 / Evolution', 'Position', [0.02 0.4 0.96 0.36]);
    pnlData = uipanel(tabInverse, 'Title', '优化配比方案 / Results', 'Position', [0.02 0.02 0.96 0.36]);

    uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.6, 0.1, 0.3], 'String', '目标强度(MPa):');
    hTarget = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.12, 0.62, 0.06, 0.25], 'String', '45');
    uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.2, 0.6, 0.1, 0.3], 'String', '容许误差(MPa):');
    hTol = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.3, 0.62, 0.06, 0.25], 'String', '2.5');

    priceHandles = cell(8, 1);
    priceUnits = {'元/kg', '元/kg', '元/kg', '元/kg', '元/kg', '元/kg', '元/day', '元/kg'};
    for i = 1:8
        uicontrol(pnlCfg, 'Style', 'text', 'Units', 'normalized', 'Position', [0.01+(i-1)*0.12, 0.3, 0.11, 0.2], ...
                  'String', [featureNamesCN{i}, '(', priceUnits{i}, ')'], 'FontSize', 8);
        priceHandles{i} = uicontrol(pnlCfg, 'Style', 'edit', 'Units', 'normalized', 'Position', [0.02+(i-1)*0.12, 0.05, 0.08, 0.2], ...
                                   'String', '0.5', 'BackgroundColor', 'w');
    end
    set(priceHandles{7}, 'String', '0');

    btnInv = uicontrol(pnlCfg, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.45, 0.55, 0.5, 0.38], ...
                       'String', '⚡ 启动 25 点并发寻优与深度回测', 'FontSize', 13, 'FontWeight', 'bold', ...
                       'BackgroundColor', [0.8 0.35 0.1], 'ForegroundColor', 'w', 'Callback', @(~,~) run_inverse_v7());

    axLive = axes(pnlPlot, 'Units', 'normalized', 'Position', [0.08 0.2 0.7 0.7]);
    xlabel(axLive, '迭代步长 / Iteration Step', 'FontWeight', 'bold'); 
    ylabel(axLive, '组分配比数值 / Dosage Value', 'FontWeight', 'bold');
    hold(axLive, 'on'); grid(axLive, 'on');
    history_X = []; lineColors = [0.8 0.1 0.1; 0.1 0.7 0.1; 0.1 0.1 0.8; 0.7 0.7 0.1; 0.7 0.1 0.7; 0.1 0.7 0.7; 0.4 0.4 0.4; 1.0 0.5 0.0];
    hLines = cell(8, 1);
    for i = 1:8
        hLines{i} = plot(axLive, NaN, NaN, 'Color', lineColors(i,:), 'LineWidth', 2, ...
                         'DisplayName', [featureNamesCN{i}, ' (', featureNamesEN{i}, ')']);
    end
    legend(axLive, 'FontSize', 8, 'Position', [0.82, 0.4, 0.15, 0.5]);

    resTable = uitable(pnlData, 'Units', 'normalized', 'Position', [0.01 0.05 0.98 0.9], 'FontSize', 10, ...
                       'ColumnName', {'设计组分 (Component)', '最优推荐值', '单位', '物理允许范围', '闭环回测验证 (Back-test)'}, ...
                       'ColumnWidth', {200, 150, 80, 220, 380});

    %% ================== 核心逻辑 ==================

    function run_prediction_v7()
        try
            X = zeros(1, 8);
            for k = 1:8
                val = str2double(get(predEditHandles{k}, 'String'));
                if val < lb_global(k) || val > ub_global(k)
                    errordlg(['数值超限：', featureNamesCN{k}, ' 应在 [', num2str(lb_global(k)), ',', num2str(ub_global(k)), ']'], 'Range Error'); return;
                end
                X(k) = val;
            end
            X_norm = mapminmax('apply', X', ps_input);
            Y_mean = mapminmax('reverse', predict(trainedModel, X_norm')', ps_output);
            axes(axPredict); cla; hold on;
            bar(1, Y_mean, 0.4, 'FaceColor', [0.2 0.5 0.7]);
            errorbar(1, Y_mean, train_MAE, 'LineWidth', 2, 'Color', 'k');
            title('图1: 预测强度及误差区间 / Fig.1: Predicted Strength Analysis'); grid on;
            set(lblRes, 'String', sprintf('预测平均值: %.2f MPa | 理论精度范围: [%.2f - %.2f] MPa', Y_mean, Y_mean-train_MAE, Y_mean+train_MAE));
        catch ME, errordlg(ME.message); end
    end

    function run_inverse_v7()
        try
            target = str2double(get(hTarget, 'String')); tol = str2double(get(hTol, 'String'));
            prices = zeros(1, 8); for k=1:8, prices(k) = str2double(get(priceHandles{k}, 'String')); end

            X_hist = res_raw(:, 1:8); Y_hist = res_raw(:, 9);
            [~, sorted_idx] = sort(abs(Y_hist - target));
            num_seeds = min(25, length(sorted_idx)); 
            seeds = X_hist(sorted_idx(1:num_seeds), :);

            opts = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', ...
                'OptimalityTolerance', 1e-11, 'StepTolerance', 1e-11, 'MaxFunctionEvaluations', 8000, ...
                'OutputFcn', @(x,vals,state) o_plot_trace(x));
            
            penalty_factor = 1000; % 惩罚系数
            best_overall_err = inf; best_X = seeds(1,:);
            h_w = waitbar(0, '正在执行 25 轮并发寻优...', 'Name', '深度 AI 优化');
            
            for s = 1:num_seeds
                if ~isgraphics(h_w), break; end
                waitbar(s/num_seeds, h_w); history_X = []; 
                % 目标函数：成本 + 强度偏差惩罚
                obj_func = @(x) sum(x .* prices) + penalty_factor * (get_pred_only(x, trainedModel, ps_input, ps_output) - target)^2;
                
                [x_tmp, ~] = fmincon(obj_func, seeds(s,:), [], [], [], [], ...
                                     lb_global, ub_global, @(x) constraints_v7(x, target, tol), opts);
                
                pred_tmp = get_pred_only(x_tmp, trainedModel, ps_input, ps_output);
                curr_err = abs(pred_tmp - target);
                if curr_err < best_overall_err
                    best_overall_err = curr_err; best_X = x_tmp;
                end
            end
            if isgraphics(h_w), delete(h_w); end

            pred_final = get_pred_only(best_X, trainedModel, ps_input, ps_output);
            tbl = cell(8, 5);
            for k = 1:8
                tbl{k,1} = [featureNamesCN{k}, ' (', featureNamesEN{k}, ')'];
                tbl{k,2} = sprintf('%.3f', best_X(k)); tbl{k,3} = units{k};
                tbl{k,4} = sprintf('[%.1f, %.1f]', lb_global(k), ub_global(k));
                if k==1, tbl{k,5} = sprintf('验证结果: %.2f MPa (偏差: %.2f)', pred_final, best_overall_err); end
            end
            set(resTable, 'Data', tbl);

            if best_overall_err <= tol
                msgbox(['寻优成功！验证偏差为 ', num2str(best_overall_err, '%.2f'), ' MPa'], '精度达标');
            else
                warndlg(['偏差 (', num2str(best_overall_err, '%.2f'), ' MPa) 超过设定容差。建议微调单价或放宽容差。'], '精度未达标');
            end
        catch ME, errordlg(['优化中断: ', ME.message]); end
    end

    function stop = o_plot_trace(x)
        stop = false; history_X = [history_X; x]; 
        steps = 1:size(history_X, 1);
        if isgraphics(axLive)
            for i = 1:8, set(hLines{i}, 'XData', steps, 'YData', history_X(:, i)); end
            drawnow limitrate;
        end
    end

    function [c, ceq] = constraints_v7(x, target, tol)
        pred_S = get_pred_only(x, trainedModel, ps_input, ps_output);
        c(1) = (target - tol) - pred_S; c(2) = pred_S - (target + tol);
        wb = x(3) / (x(1) + x(2)); c(3) = 0.28 - wb; c(4) = wb - 0.55; ceq = [];
    end

    function s_out = get_pred_only(x, model, ps_in, ps_out)
        px = mapminmax('apply', x', ps_in);
        tx = predict(model, px');
        s_out = mapminmax('reverse', tx', ps_out);
    end
end