UTMOST 결과중 R square값(Liver), 전반적인 경향 보기위해 9916개 중 30개씩 건너뛰어서 331개
(2개 gene은 ENSG넘버가 하나씩 더있음, 그래서 데이터셋은 333개)
overall gene에 대해, tissue : Liver

module.py 는 validation + 620 sample용 cross tissue 고려하도록 한 모듈

Trial1 ( main1.py, model1.py, module.py, train_bash1.sh ) / ~ model 3
hidden layer 1개, dropout=0.9, base node = 256
전체 sample 620개에 대해서 44개 tissue의 GX level한번에 고려하여 training.

'''
Sample_NAmesby_Tissue_by_Patient.csv : Tissue별 patient별 Tissue에 따른 sample 이름 620명. (matching용)
whole_ensg2name.json : ENSG번호에서 gene name으로 바꾸기 위한 dictionary. window상의 gtex_project에서 만듬.
'''
