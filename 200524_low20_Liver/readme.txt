UTMOST 결과중 R square값이 가장 낮은 200개 gene에 대해, tissue : Liver

Trial2 ( main2.py, model2.py, module.py, train_bash2.sh )
- Linear model, 단일 layer, lasso coeff : 0.003, 0.03, 0.1, 0.3, 0.5, 1
input : 0, 1, 2그대로 사용. , GX level : (x - x_mean)/x_mean 으로 normalize
5-fold validation
model1번이 성능 가장 좋음. 2,6번은 시간절약 위해 중간에 끊음.(가능성 없음)

Trial3는 Trial2 그대로인데,  ( main3.py, model2.py, module2.py, train_bash3.sh)
validation, test 로 나눠서, validation loss가 가장 작을떄 test에 대한 결과 취하는방식.
168.188.90.194에서 돌렸음( npy파일만 가져와서 pred_analysis.ipynb 로 결과확인 )
model1만 돌려보고, trial2와 비교. 약간의 성능차이 나는것만 확인.

Trial5 ( main5.py, model5.py, module.py, train_bash5.sh )
- Fully connected Network, Dropout : 0.5 , Lasso coeff : 0.0001, 0.0005, nodes = 128
model3 중간에서 끊음.
194에서 돌렸음 ( npy파일만 가져와서 pred_analysis.ipynb 로 결과확인 )

Trial6 ( main6.py, model6.py, module.py, train_bash6.sh )
- CNN, n_channel : 8, 16 / Lasso coeff : 0.03, 0.3 / 마지막 FCN에만 lasso 적용

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
