function [feat,loc] =  MovingWinFeats(x,fs,winLen,winDisp,featFn)

NumWins = @(xLen,fs,winLen,winDisp) ceil((xLen/fs-winLen)/(winDisp)+1);
NWins = NumWins(length(x),fs,winLen,winDisp);

for i = 1:NWins
    y = x(round(1+(i-1)*winDisp*fs):round(winLen*fs+(i-1)*winDisp*fs));
    feat(i) = featFn(y);
    loc(i) = round(winLen*fs+(i-1)*winDisp*fs);
end

