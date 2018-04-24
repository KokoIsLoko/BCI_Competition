%% Final project -- subject 3

clear; clc;
load('predicted_dg_v2.mat')

% Get data
session = IEEGSession('I521_Sub3_Training_ecog', 'KokoIsLoko', 'Kok_ieeglogin.bin');
session.openDataSet('I521_Sub3_Training_dg');
session.openDataSet('I521_Sub3_Leaderboard_ecog');

fs = session.data(1).sampleRate;
trainingECoG = session.data(1).getvalues(1:300000,1:64);
nyq = fs/2;

avgFn = @(x) mean(x);

featureMatrix = [];
for i = 1:64
    % Average voltage magnitude feature
    avgVolt = MovingWinFeats(trainingECoG(:,i),fs,.100,.05,avgFn);
    
    % Spectrogram with 100 ms time frame, 50 ms overlap, 5 Hz/bin
    s = spectrogram(trainingECoG(:,i),100,50,199);
    
    % Average frequency magnitude features
    avg5_15Hz = mean(abs(s(2:3,:)));
    avg20_25Hz = abs(s(5,:));
    avg75_115Hz = mean(abs(s(15:22,:)));
    avg125_160Hz = mean(abs(s(25:36,:)));
    avg160_175Hz = mean(abs(s(37:39,:)));
    
    % Feature matrix should be 5999x372
    featureMatrix = [featureMatrix avgVolt' avg5_15Hz' avg20_25Hz' ...
        avg75_115Hz' avg125_160Hz' avg160_175Hz'];
    
end

%% Finger trace data

fingerTrace = session.data(2).getvalues(1:300000,1:5);

for i = 1:5
    y(:,i) = decimate(fingerTrace(:,i),50);
end

Y = y(4:end,:);

% Making X matrix for predictions
v = 6*64;
N = 3;
M = 5999-(N-1);
X = zeros(M,v*N+1);
for i = 1:M
    X(i,:) = [1 reshape(featureMatrix(i:(N-1)+i,:),1,v*N)];
end

%% LASSO Calculation
X_test = X(1:1500,:);
Y_test = Y(1:1500,:);

X_train = X(1501:end,:);
Y_train = Y(1501:end,:);

modelF1 = lasso(X_train,Y_train(:,1)');
modelF2 = lasso(X_train,Y_train(:,2)');
modelF3 = lasso(X_train,Y_train(:,3)');
modelF4 = lasso(X_train,Y_train(:,4)');
modelF5 = lasso(X_train,Y_train(:,5)');

%% Correlation vs. None-Zero Finger 1
for m = 1:length(modelF1(1,:))
    %Calculating a prediction of X values based on model weights and test
    %set
    predPositionF1 = X_test*modelF1(:,m);
    
    %Calculating how many none-zero elements there are in the weight vector
    %of model
    noneZeroNrF1(m) = length(find(modelF1(:,m) ~= 0));
    
    %Calculating correlation coefficient between predicted data and actual
    %test set values
    corrCoefF1(m) = corr(predPositionF1, Y_test(:,1));
end

%% Correlation vs. None-Zero Finger 2
for m = 1:length(modelF2(1,:))
    %Calculating a prediction of X values based on model weights and test
    %set
    predPositionF2 = X_test*modelF2(:,m);
    
    %Calculating how many none-zero elements there are in the weight vector
    %of model
    noneZeroNrF2(m) = length(find(modelF2(:,m) ~= 0));
    
    %Calculating correlation coefficient between predicted data and actual
    %test set values
    corrCoefF2(m) = corr(predPositionF2, Y_test(:,2));
end

%% Correlation vs. None-Zero Finger 3
for m = 1:length(modelF3(1,:))
    %Calculating a prediction of X values based on model weights and test
    %set
    predPositionF3 = X_test*modelF3(:,m);
    
    %Calculating how many none-zero elements there are in the weight vector
    %of model
    noneZeroNrF3(m) = length(find(modelF3(:,m) ~= 0));
    
    %Calculating correlation coefficient between predicted data and actual
    %test set values
    corrCoefF3(m) = corr(predPositionF3, Y_test(:,3));
end

%% Correlation vs. None-Zero Finger 4
for m = 1:length(modelF4(1,:))
    %Calculating a prediction of X values based on model weights and test
    %set
    predPositionF4 = X_test*modelF4(:,m);
    
    %Calculating how many none-zero elements there are in the weight vector
    %of model
    noneZeroNrF4(m) = length(find(modelF4(:,m) ~= 0));
    
    %Calculating correlation coefficient between predicted data and actual
    %test set values
    corrCoefF4(m) = corr(predPositionF4, Y_test(:,4));
end

%% Correlation vs. None-Zero Finger 5
for m = 1:length(modelF5(1,:))
    %Calculating a prediction of X values based on model weights and test
    %set
    predPositionF5 = X_test*modelF5(:,m);
    
    %Calculating how many none-zero elements there are in the weight vector
    %of model
    noneZeroNrF5(m) = length(find(modelF5(:,m) ~= 0));
    
    %Calculating correlation coefficient between predicted data and actual
    %test set values
    corrCoefF5(m) = corr(predPositionF5, Y_test(:,5));
end
%% Find best model for each finger 
bestF1 = find(corrCoefF1 == max(corrCoefF1));
bestF2 = find(corrCoefF2 == max(corrCoefF2));
bestF3 = find(corrCoefF3 == max(corrCoefF3));
bestF4 = find(corrCoefF4 == max(corrCoefF4));
bestF5 = find(corrCoefF5 == max(corrCoefF5));

% Calculating the prediction for each finger from test set using best
% models
testF1 = X_test*modelF1(:,bestF1);
testF2 = X_test*modelF2(:,bestF2);
testF3 = X_test*modelF3(:,bestF3);
testF4 = X_test*modelF4(:,bestF4);
testF5 = X_test*modelF5(:,bestF5);

totalTest = [testF1, testF2, testF3, testF4, testF5];

% Calculating average correlation coefficient from all 5 fingers
sumCorr = 0;
for i = 1:5
    sumCorr = sumCorr + corr(totalTest(:,i), Y_test(:,i));
end
avgCorr = sumCorr/5;

%% Making predictions

testingTrace = session.data(3).getvalues(1:150000,1:64);

testingFeatureMatrix = [];
for i = 1:64
    % Average voltage magnitude feature
    avgVolt = MovingWinFeats(testingTrace(:,i),fs,.100,.05,avgFn);
    
    % Spectrogram with 100 ms time frame, 50 ms overlap, 5 Hz/bin
    s = spectrogram(testingTrace(:,i),100,50,199);
    
    % Average frequency magnitude features
    avg5_15Hz = mean(abs(s(2:3,:)));
    avg20_25Hz = abs(s(5,:));
    avg75_115Hz = mean(abs(s(15:22,:)));
    avg125_160Hz = mean(abs(s(25:36,:)));
    avg160_175Hz = mean(abs(s(37:39,:)));
    
    % Feature matrix should be 5999x372
    testingFeatureMatrix = [testingFeatureMatrix avgVolt' avg5_15Hz' avg20_25Hz' ...
        avg75_115Hz' avg125_160Hz' avg160_175Hz'];
    
end

v = 6*64;
N = 3;
M = 2949-(N-1);
Xtest = zeros(M,v*N+1);
for i = 1:M
    Xtest(i,:) = [1 reshape(testingFeatureMatrix(i:(N-1)+i,:),1,v*N)];
end

%%
% Make prediction for first patient for every finger based on the best hand
% finger model 
YhatF1 = Xtest*modelF1(:,bestF1);
YhatF2 = Xtest*modelF2(:,bestF2);
YhatF3 = Xtest*modelF3(:,bestF3);
YhatF4 = Xtest*modelF4(:,bestF4);
YhatF5 = Xtest*modelF5(:,bestF5);

Yhat = [YhatF1, YhatF2, YhatF3, YhatF4, YhatF5];

% Cubic spline interpolation

xq = 50:2947*50;
for i = 1:5
    pred(:,i) = spline([1:2947].*50,Yhat(:,i),xq);
    predicted(:,i) = [zeros(100,1);pred(:,i);zeros(99,1)];
end

predicted_dg{3} = predicted;
