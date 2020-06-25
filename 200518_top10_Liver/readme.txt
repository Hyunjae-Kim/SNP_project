UTMOST 결과중 R square값이 가장높은 10개 gene에 대해, tissue : Liver
    

Trial1 ( main1.py, model1.py, module.py, train_bash1.sh )
- Fully Connected Network, layer : 3개, base node : 512 / 1024
input : 0, 1, 2그대로 사용. , GX level : (x - x_mean)/x_mean 으로 normalize
5-fold validation
            
Trial2 ( main2.py, model2.py, module.py, train_bash2.sh )
- Linear model, 단일 layer, lasso coeff : 0.003, 0.03, 0.1, 0.3, 0.5, 1
input : 0, 1, 2그대로 사용. , GX level : (x - x_mean)/x_mean 으로 normalize
5-fold validation

Trial3 ( main3.py, model3.py, module.py, train_bash3.sh )
- Fully Connected Network, hidden layer, base node : 4092
Universal Approx thm 으로 되야한다.
input : 0, 1, 2그대로 사용. , GX level : (x - x_mean)/x_mean 으로 normalize
5-fold validation

Trial4 ( main4.py, model4.py, module.py, train_bash4.sh )
- Trial3과 동일, Dropout만 추가, drop prob = 0.5, 0.7, 0.9

Trial5 ( main5.py, model5.py, module.py, train_bash5.sh )
- 단일층 FCN, node 수 128 / 64 / 256, lasso coefficient = [0.0005, 0.005, 0.05, 0.1, 0.3] or [0.0005, 0.001, 0.003, 0.005, 0.01, 0.03]



check_top50_list.ipynb 
- 먼저 UTMOST predicted와 observed양쪽에 전부다 존재하는 patient선택 (184명)
- 그중에서 plink로 얻은 데이터에 존재하는 patient들만 선택 (155명)   -> sorted_sample_idx.npy
- UTMOST에서 predicted와 observed 양쪽에 전부다 존재하는 gene 선택 (10145개)
- expression에 nan값 있는 gene들 제외하고 (9916개) 그중에서 R^2값 가장 높은 top 10개 선택 (top10_name2rsqr.json)
    
-> 여기서 고른 top10 gene을, window상의 gtex_project의 gene_list에 넣고 데이터 추출 -> SNP데이터 획득

'''
Sample_NAmesby_Tissue_by_Patient.csv : Tissue별 patient별 Tissue에 따른 sample 이름 620명. (matching용)
whole_ensg2name.json : ENSG번호에서 gene name으로 바꾸기 위한 dictionary. window상의 gtex_project에서 만듬.
'''
