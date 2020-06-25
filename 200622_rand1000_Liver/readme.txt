UTMOST 결과중 R square값, 유의미하다고 보기 힘든 0.1 이하 gene들 빼고
전반적인 경향 보기위해 0.1이상 500개 중 5개씩 건너뛰어서 100개
tissue : Liver
[194] : 168.188.90.194 서버 / [183] 192.168.1.183 서버  
(두개 서버에서 load하는 gene_num 순서가 달라서 pred_analysis할 때 matching 하는코드 on/off 확인) 
    
[194]Trial1 ( main1.py, model1.py, module.py, train_bash1.sh ) / ~ model 1
FCN 1개 hidden layer, Dropout 0.9, ReLU, Lasso 모델
hidden nodes num : 256 고정 / lasso coeff : 0.005
validation 없이. / Loss : L1 loss / 194에서 학습(gene num matching 필요)

