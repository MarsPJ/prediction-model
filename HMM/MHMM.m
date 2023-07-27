TRANS = [0.7 0.2 0.1;
         0.3 0.3 0.4;
         0.3 0.5 0.2];%转移矩阵A
EMIS = [0.9, 0.1;
        0.5, 0.5;
        0.2, 0.8];%混淆矩阵B
seq=[2 1]; %观测状态序列
likelystates = hmmviterbi(seq, TRANS, EMIS)