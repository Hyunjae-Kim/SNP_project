UTMOST 결과중 R square값, 유의미하다고 보기 힘든 0.1 이하 gene들 빼고
전반적인 경향 보기위해 0.1이상 500개 중 5개씩 건너뛰어서 100개
tissue : Liver
[194] : 168.188.90.194 서버 / [183] 192.168.1.183 서버  
(두개 서버에서 load하는 gene_num 순서가 달라서 pred_analysis할 때 matching 하는코드 on/off 확인) 
    
[183]Trial1 ( main1.py, model1.py, module.py, train_bash1.sh ) / ~ model 5
FCN 1개 hidden layer, Dropout 0.9, ReLU, Lasso 모델
hidden nodes num : 256 고정 / lasso coeff : 0.005, 0.0005, 0.00005, 0, 0.05 
validation없이. Loss만 L1 loss

[183]Trial2 ( main2.py, model2.py, train_bash2.sh ) / ~ model 3
Trial1의 model1과 hyper-parameter, model구조 같음.
model1 : Trial1중에서 R^2>0.4 인 gene 3개에 대해서 똑같이 돌림(확인차)
model2 : 위 세개 gene에 대해서 SNP포인트중 전부다 0인 포인트 없애고(module2.py)
model3 : 위 세개 gene에 대해서 input에 random noise살짝 섞어서 augmentation(module3.py)
    
[183]Trial3 ( main3.py, model3.py, module3.py, train_bash3.sh ) / ~ model 2
FCN2개, batch norm, lasso 없이, ReLU, hidden nodes수 : 2048 - 256 
model1 : Trial1중에서 R^2>0.4 인 gene 3개에 대해서, random noise augmentation 해서 학습

[183]Trial4 ( main4.py, model4.py, module.py, train_bash4.sh ) / ~ model 5
Linear Lasso model , lasso coeff : 0.005, 0.05, 0.1, 0.3, 0.5
Trial1중에서 R^2>0.4 인 gene 3개에 대해서, validation 없이. Loss만 L1 loss
    