UTMOST 결과중 R square값, 전반적인 경향 보기위해 9916개 중 30개씩 건너뛰어서 331개
(2개 gene은 ENSG넘버가 하나씩 더있음, 그래서 데이터셋은 333개)
overall gene에 대해, tissue : Liver

top10 이나 low200에서는 module.py 가 validation없는거, module2.py가 validation 추가된거,
여기서는 module.py가 validation 추가된것.
module3.py는 validation + cross tissue 고려하도록 UTMOST observed 데이터에서 GX level 불러와서 매칭시킨것

Trial1 ( main1.py, model1.py, module.py, train_bash1.sh ) / ~ model 1
200524의 trial6에서, model3,4를 적용. (trial_num6 -> 1, module2 -> module로 이름 바꿈)
- CNN, n_channel : 16 / Lasso coeff : 0.03, 0.3 / 마지막 FCN에만 lasso 적용

Trial2 ( main2.py, model2.py, module.py, train_bash2.sh ) / ~ model 1
FCN layer 4개, Dropout=0.5, leakyReLU, hidden nodes : 2048 - 1024 - 512 
- lasso coeff : 0.3, 0.03, 0.003  ( 큰거부터 먼저 돌림 )
* 168.188.90.194에서 돌림

Trial3 ( main3.py, model3.py, module.py, train_bash3.sh ) / ~ model 2
hidden layer 1개, dropout=0.9, base node = 256
top10의 trial5와 동일. validation set 써서, 5fold - (3/1/1 : tr/vad/te)

Trial4 ( main4.py, model3.py, module2.py, train_bash4.sh ) / ~ model 3
Trial3와 동일. validation 없이 top10의 trial5와 같게끔.
여기서부터, 시간 절약을 위해, 333개 gene안하고, gene100까지만 확인.
(trial3에서, UTMOST보다 결과 저조, validation 때문인지 확인하기 위해 trial4)

Trial5 ( main5.py, model5.py, module.py, train_bash5.sh ) / ~ model 6
trial2,3이 성능 저조해서 확인을 위해 Linear모델로 UTMOST와 비교.
top10의 trial2와 동일한 셋업으로 학습. but validation set 추가해서.
* 168.188.90.194에서 돌림

Trial6 ( main6.py, model6.py, module3.py, train_bash6.sh ) / ~ model 3
hidden layer 1개, dropout=0.9, base node = 256
Trial3과 동일, Liver sample 153개에 대해서 44개 tissue의 GX level한번에 고려하여 training.

Trial7 ( main7.py, model7.py, module.py, train_bash7.sh ) / ~ model 6
trial5와 동일. Linear모델. 결과로 얻은 R^2값이 학습에의한 예측으로 얻은값인지 확인하기위해
학습은 정상적으로 진행하고, test에 대해서 SNP를 shuffle해서 넣었을때. 즉 SNP의 정보가 없이
학습된 weight들로 GX level prediction값을 얻어서 R^2값 계산.
* 168.188.90.194에서 돌림

'''
Sample_NAmesby_Tissue_by_Patient.csv : Tissue별 patient별 Tissue에 따른 sample 이름 620명. (matching용)
whole_ensg2name.json : ENSG번호에서 gene name으로 바꾸기 위한 dictionary. window상의 gtex_project에서 만듬.
'''
