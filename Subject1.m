%% Final project -- subject 1

% Get data
session = IEEGSession('I521_Sub1_Training_ecog', 'dweiss', 'Dwe_ieeglogin');
session.openDataSet('I521_Sub1_Training_dg');
session.openDataSet('I521_Sub1_Leaderboard_ecog');

fs = session.data(1).sampleRate;
trainingECoG = session.data(1).getvalues(1:300000,1:62);
nyq = fs/2;

avgFn = @(x) mean(x);

featureMatrix = [];
for i = 1:62
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
v = 372;
N = 3;
M = 5999-(N-1);
X = zeros(M,v*N+1);
for i = 1:M
    X(i,:) = [1 reshape(featureMatrix(i:(N-1)+i,:),1,v*N)];
end

B = (X'*X)\(X'*Y);

%% Making predictions

testingTrace = session.data(3).getvalues(1:150000,1:62);

testingFeatureMatrix = [];
for i = 1:62
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

v = 372;
N = 3;
M = 2949-(N-1);
Xtest = zeros(M,v*N+1);
for i = 1:M
    Xtest(i,:) = [1 reshape(testingFeatureMatrix(i:(N-1)+i,:),1,v*N)];
end

% Make prediction for first patient

Yhat = Xtest*B;

% Cubic spline interpolation

xq = 50:2947*50;
for i = 1:5
    pred(:,i) = spline([1:2947].*50,Yhat(:,i),xq);
    predicted(:,i) = [zeros(100,1);pred(:,i);zeros(99,1)];
end

predicted_dg{1} = predicted;
