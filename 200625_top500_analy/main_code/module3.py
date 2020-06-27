import numpy as np
import torch

def sort012_num(snp_data):
    len_data = len(snp_data)
    num0 = np.sum(snp_data==0)
    num1 = np.sum(snp_data==1)
    num2 = np.sum(snp_data==2)
    num_mut = num1+num2
    return num0, num1, num2, num_mut


def load_data(gene_data_name, gene_name, tissue_num, proc=True):
    np.random.seed(37)
    
    snp_name = '../../%s/%s/%s__snps.npy'%(gene_data_name, gene_name, gene_name)
    gx_name = '../../%s/%s/%s__gene_expression_levels.npy'%(gene_data_name, gene_name, gene_name)

    a = np.float32(np.load(snp_name).transpose())
    b = np.float32(np.load(gx_name).transpose())
    
    no_nan_arg = np.nan_to_num(np.abs(b[:,tissue_num-1])+1)>0
    no_nan_idx = np.where(no_nan_arg)[0]
    snp_data = a[no_nan_idx, :]
    gx_data = b[no_nan_idx, tissue_num-1,np.newaxis]
    
    if proc:
        tiss_mean = np.mean(gx_data)
        gx_data = (gx_data - tiss_mean)/tiss_mean
        
        s = np.arange(len(gx_data))
        np.random.shuffle(s)
    
        snp_data = snp_data[s]
        gx_data = gx_data[s]

    return snp_data, gx_data


def k_fold_data(snp_data, gx_data, div_num, k_num):
    if div_num>10:
        print("Error : div_num should be < 10")
        raise ValueError
        
    elif k_num<=0 or k_num>div_num:
        print("Error : k_num should > 1 and <= div_num")
        raise ValueError
        
    data_len = len(snp_data)
    div_len = int(data_len/div_num)
    
    start_te_idx = div_len*(k_num-1)
    if div_num==k_num:
        end_te_idx = data_len
    else:
        end_te_idx = div_len*k_num
    
    snp_train = np.vstack((snp_data[:start_te_idx,:], snp_data[end_te_idx:,:]))
    snp_test = snp_data[start_te_idx:end_te_idx]
    gx_train = np.vstack((gx_data[:start_te_idx,:], gx_data[end_te_idx:,:]))
    gx_test = gx_data[start_te_idx:end_te_idx]
    
    snp_train_ = snp_train.copy()
    gx_train_ = gx_train.copy()

    for k in range(9):
        a = np.random.normal(scale=0.5, size=np.shape(snp_train_))*0.1
        dsnp_train = snp_train_+a
        snp_train = np.vstack((snp_train, dsnp_train))
        gx_train = np.vstack((gx_train, gx_train_))
        
    return snp_train, snp_test, gx_train, gx_test
    
    
def sample_dist(input_data):
    data_sum = torch.sum(input_data**2, dim=1, keepdim=True)
    data_mat = -2*torch.matmul(input_data, input_data.transpose(1,0)) + data_sum
    data_dis = data_mat.transpose(1,0) + data_sum
    return data_dis
    
    