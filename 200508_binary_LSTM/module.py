import numpy as np
import torch

def sort012_num(snp_data):
    len_data = len(snp_data)
    num0 = np.sum(snp_data==0)
    num1 = np.sum(snp_data==1)
    num2 = np.sum(snp_data==2)
    num_mut = num1+num2
    return num0, num1, num2, num_mut

def binary_sort(snp_train, snp_test, gx_train, gx_test, mode, threshold=-0.5):
    gx_train_ = (gx_train.copy()>threshold).astype(int)
    gx_test_ = (gx_test.copy()>threshold).astype(int)
    
    train_nm_idx = np.where(gx_train_==0)[0]
    train_ab_idx = np.where(gx_train_==1)[0]

    snp_nm = snp_train[train_nm_idx]
    snp_ab = snp_train[train_ab_idx]
    
    idx_list_ab = []
    for i in range(len(snp_train[0])):
        sort_nm = snp_nm[:,i]
        sort_ab = snp_ab[:,i]
        nm0, nm1, nm2, nm_mut = sort012_num(sort_nm)
        ab0, ab1, ab2, ab_mut = sort012_num(sort_ab)

        if (ab1!=0 or ab2!=0) and (nm1==0 and nm2==0):
            idx_list_ab.append(i)
        
    snp_test_ = snp_test.copy()[:, idx_list_ab]

    test_nm_idx = []
    test_ab_idx = []
    for i in range(len(snp_test_)):
        if np.sum(snp_test_[i])==0:
            test_nm_idx.append(i)
        elif np.sum(snp_test_[i])!=0:
            test_ab_idx.append(i)
    
    train_nm_snp = snp_train[train_nm_idx]
    train_ab_snp = snp_train[train_ab_idx]
    train_nm_gx = gx_train[train_nm_idx]
    train_ab_gx = gx_train[train_ab_idx]
    test_nm_snp = snp_test[test_nm_idx]
    test_ab_snp = snp_test[test_ab_idx]
    test_nm_gx = gx_test[test_nm_idx]
    test_ab_gx = gx_test[test_ab_idx]
    
    if mode=='nm':
        return train_nm_snp, test_nm_snp, train_nm_gx, test_nm_gx
    elif mode=='ab':
        return train_ab_snp, test_ab_snp, train_ab_gx, test_ab_gx

def load_data(gene_name, tissue_num, proc=True):
    np.random.seed(37)
    
    snp_name = '../dataset/%s/x_np_snps_%s.npy'%(gene_name, gene_name)
    gx_name = '../dataset/%s/y_gene_expression_%s.npy'%(gene_name, gene_name)

    a = np.load(snp_name)
    b = np.load(gx_name)
    
    print(np.shape(a), np.shape(b))
    no_nan_arg = np.nan_to_num(np.abs(b[:,tissue_num-1])+1)>0
    no_nan_idx = np.where(no_nan_arg)[0]

    snp_data = a[no_nan_idx, :]
    gx_data = b[no_nan_idx, tissue_num-1,np.newaxis]
    
    if proc:
        snp_mask = (snp_data==2)*4
        snp_data = (snp_data + snp_mask*snp_data)/10
        
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
    
    return snp_train, snp_test, gx_train, gx_test
    
    
def sample_dist(input_data):
    data_sum = torch.sum(input_data**2, dim=1, keepdim=True)
    data_mat = -2*torch.matmul(input_data, input_data.transpose(1,0)) + data_sum
    data_dis = data_mat.transpose(1,0) + data_sum
    return data_dis
    
    