Gene1 : ENSG00000206535.3   (LNP1)
Gene2 : ENSG00000217442.3   (SYCE3)
Gene3 : ENSG00000225972.1
Gene4 : ENSG00000007923.11
    
Trial description
모든 Trial은 model_check?.ipynb 에서 코드 돌아가는것 확인 후, model?.py, main?.py로 구성하여
train?.sh 를 통해 학습을 진행하였음. R2결과는 R2_measure?.ipynb로 얻음.

데이터는 전체 데이터에서 tissue개씩만 선택하여 가용한 데이터를 사용.
5-fold validation으로, 최종 R2값 구할때는 각 fold별로 얻은 test데이터를 종합하여 구함.
즉, 해당 tissue의 가용한 데이터 전체에 대해서 R2를 구하게 됨.
(ex, 1번 tissue observed data가 있는 sample이 500명이라고하면, 각 fold별로 400명은 train, 100은 test.
 test데이터가 겹치지 않도록 5-fold validation하면 결국 전체 500명 데이터에 대해서 R2를 구할 수 있음)

trial2 - Tissue 1 ~ 3에 대해서, input을 일정길이단위로 쪼개서 bidLSTM으로 학습시킴.
        각 time stamp별로 FC거쳐서 하나의 값 얻음. 전체에 대해서 FC하나 더 거쳐서 최종적으로 GX 예측값 얻음.
        Direct inference.
        model별로 input chunk의 길이, LSTM layer수가 달라짐.
        * Model 7은 Model1에서 hidden의 크기만 크게해본것. 학습시간 너무 길어서 하나만 진행하였음.
