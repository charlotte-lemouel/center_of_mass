function visualise(Acc_subsampled, Pos_subsampled, Pos_estimate, Vel_estimate, Frequency, Title)

    % title
    if nargin < 6
        Title = '';
    end

    % Number of dimensions and samples
    [NbOfDimensions, NbOfSamples] = size(Acc_subsampled);
    time = (0:NbOfSamples-1) / Frequency;

    % Create figure and subplots
    figure;
    for dim = 1:NbOfDimensions
        % Position plot
        subplot(3, NbOfDimensions, dim);
        plot(time, Pos_subsampled(dim,:), 'k', 'DisplayName', 'measurement','LineWidth',1);
        hold on;
        plot(time, Pos_estimate(dim,:), 'b', 'DisplayName', 'estimate','LineWidth',1);
        if dim == 1
            legend(Location="best");
        end
        if dim == 1
            ylabel('Position (m)');
        elseif dim == 2    
            title(Title);
        end
        
        % Velocity plot
        subplot(3, NbOfDimensions, NbOfDimensions + dim);
        plot(time, Vel_estimate(dim,:), 'b','LineWidth',1);
        if dim == 1
            ylabel('Velocity (m/s)');
        end

        % Acceleration plot
        subplot(3, NbOfDimensions, 2*NbOfDimensions + dim);
        plot(time, Acc_subsampled(dim,:), 'k','LineWidth',1);
        xlabel('Time (s)');
        if dim == 1
            ylabel('Acceleration (m/s^2)');
        end
    end
end
