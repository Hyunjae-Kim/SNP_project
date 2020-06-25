UTMOST 결과중 R square값, 유의미하다고 보기 힘든 0.1 이하 gene들 빼고
전반적인 경향 보기위해 0.1이상 500개 중 5개씩 건너뛰어서 100개
tissue : Liver
[194] : 168.188.90.194 서버 / [183] 192.168.1.183 서버  
(두개 서버에서 load하는 gene_num 순서가 달라서 pred_analysis할 때 matching 하는코드 on/off 확인) 
    
[194]Trial1 ( main1.py, model1.py, module.py, train_bash1.sh ) / ~ model 3
FCN 1개 hidden layer, Dropout 0.9, ReLU, Lasso 모델
hidden nodes num : 256 고정 / lasso coeff : 0.005, 0.0005, 0.00005 
validation set O / 194에서 학습(gene num matching 필요)

[194]Trial2 ( main2.py, model2.py, module.py, train_bash2.sh ) / ~ model 5
CNN모델, 스트라이드 없이 input위치별 다른 kernel로 convolution -> FC 모델, LeakyReLU, Lasso
kernel length : 8, 16, 32, 64, 128 / lasso coeff : 0.00005 고정
validation set O / 194에서 학습(gene num matching 필요)

[194]Trial3 ( main3.py, model3.py, module.py, train_bash3.sh ) / ~ model 1 (2 중간 cut)
one-direction LSTM 모델, 마지막 output -> FC 모델, Lasso
input_size : 100, 50 / hidden size : 200, 100 / layer num : 2 고정 / lasso coeff : 0.005 고정
validation set O / 194에서 학습(gene num matching 필요)

[194]Trial4 ( main4.py, model4.py, module2.py, train_bash4.sh ) / ~ model 5
Trial1과 동일, validation없이. lasso coeff : 0.005, 0.0005, 0.00005, 0, 0.05

[183]Trial5 ( main5.py, model5.py, module2.py, train_bash5.sh ) / ~ model 1 ~ gene 89에서 cut
Trial4과 동일, FCN 인풋을 one-hot으로 하고 flatten해서 ( 길이 3배 ) 시도한 모델.

[194]Trial6 ( main6.py, model6.py, module2.py, train_bash6.sh ) / ~ model 1
Trial2와 동일, validation 없이, Trial2에서 가장 잘된 model1번만

[183]Trial7 ( main7.py, model7.py, module2.py, train_bash7.sh ) / ~ model ...
self-attention + FC로만 구성. (시도중 모델)

[183]Trial8 ( main8.py, model8.py, module2.py, train_bash8.sh ) / ~ model 5
Trial4과 동일, validation없이. Loss만 L1 loss

[194]Trial9 ( main9.py, model9.py, module2.py, train_bash9.sh ) / ~ model 3
Trial4과 동일, validation없이. lasso 없이. Dropout 0, 0.3, 0.5

