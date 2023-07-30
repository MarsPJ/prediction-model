function [fitresult, gof] = createFit(year, population)
%CREATEFIT(YEAR,POPULATION)
%  Create a fit.
%
%  Data for 'population_pre' fit:
%      X Input : year
%      Y Output: population
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 30-Jul-2023 13:46:40 自动生成


%% Fit: 'population_pre'.
[xData, yData] = prepareCurveData( year, population );

% Set up fittype and options.
ft = fittype( 'xm/(1+(xm/3.9-1)*exp(-r*(t-1790)))', 'independent', 't', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0.02 500];

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

% Plot fit with data.
figure( 'Name', 'population_pre' );
h = plot( fitresult, xData, yData );
legend( h, 'population vs. year', 'population_pre', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 'year', 'Interpreter', 'none' );
ylabel( 'population', 'Interpreter', 'none' );
grid on


