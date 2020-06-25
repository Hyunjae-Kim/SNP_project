import os
import time
import argparse
import module
import numpy as np
import matplotlib.pyplot as plt
from model2 import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init


trial_num = 2
sub_num = 1
parser = argparse.ArgumentParser(description = 'give num')
parser.add_argument('n_gene', type=int)
parser.add_argument('n_model', type=int)
parser.add_argument('n_tiss', type=int)
parser.add_argument('n_fold', type=int)
args = parser.parse_args()

gene_num = args.n_gene + 1
model_num = args.n_model + 1
tissue_num = args.n_tiss + 1
k_num = args.n_fold + 1

m_count = 0
for p1 in [100]:
    for p2 in [50]:
        for p3 in [1,2,3]:
            m_count += 1
            input_size = p1
            hidden_size = p2
            num_layers = p3
            if m_count==model_num:
                break
        if m_count==model_num:
            break
    if m_count==model_num:
        break
        
gene_list = ['ENSG00000206535.3', 'ENSG00000217442.3', 'ENSG00000225972.1', 'ENSG00000007923.11']
gene_name = gene_list[gene_num-1]

print('\n\n[Gene %d] Model %d ( tissue %d ) - %d/5 fold data'%(gene_num, model_num, tissue_num, k_num))
print('Option : %d input length   //  %d hidden length  //  %d layers'%(input_size, hidden_size, num_layers))

try:
    if not os.path.exists('npy/trial%d_%d/gene%d/'%(trial_num, sub_num, gene_num)): 
        os.mkdir('npy/trial%d_%d/gene%d/'%(trial_num, sub_num, gene_num))
except FileExistsError:
    print('Already exist folder')
    
try:
    if not os.path.exists('npy/trial%d_%d/gene%d/model%d'%(trial_num, sub_num, gene_num, model_num)): 
        os.mkdir('npy/trial%d_%d/gene%d/model%d'%(trial_num, sub_num, gene_num, model_num))
except FileExistsError:
    print('Already exist folder')

try:
    if not os.path.exists('img/trial%d_%d/gene%d'%(trial_num, sub_num, gene_num)): 
        os.mkdir('img/trial%d_%d/gene%d'%(trial_num, sub_num, gene_num))
except FileExistsError:
    print('Already exist folder')
    
try:
    if not os.path.exists('img/trial%d_%d/gene%d/loss_plot'%(trial_num, sub_num, gene_num)): 
        os.mkdir('img/trial%d_%d/gene%d/loss_plot'%(trial_num, sub_num, gene_num))
except FileExistsError:
    print('Already exist folder')
    
    
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

start_time = time.time()
np.random.seed(37)
torch.manual_seed(37)
torch.cuda.manual_seed_all(37)
torch.backends.cudnn.deterministic = True

snp, gx = module.load_data(gene_name, tissue_num, proc=True)
snp_len = np.shape(snp)[-1]
len_dif = snp_len%input_size
snp = snp[:, int(len_dif/2):int(len_dif/2)+snp_len-len_dif]

snp_train, snp_test, gx_train, gx_test = module.k_fold_data(snp, gx, 5, k_num)
# gx_train_ = (gx_train.copy()>-0.5).astype(int)
# gx_test_ = (gx_test.copy()>-0.5).astype(int)

snp_train = torch.Tensor(snp_train).to(device)
snp_test = torch.Tensor(snp_test).to(device)
gx_train = torch.Tensor(gx_train).to(device)
gx_test = torch.Tensor(gx_test).to(device)

snp_train = snp_train.view(snp_train.size()[0], -1, input_size)
snp_test = snp_test.view(snp_test.size()[0], -1, input_size)

print('\nData shape @@@@@@')
print('Train data : ', np.shape(snp_train),' / ', np.shape(snp_test))
print('Test data : ', np.shape(gx_train), ' / ', np.shape(gx_test))
print('\n')

learning_rate = 0.0002
model = Model(input_size, hidden_size, num_layers, snp_train.size()[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

tr_loss_list = []
te_loss_list = []

tr_loss_buff = 0
te_loss_buff = 0
min_iter = 0
for i in range(3001):
    optimizer.zero_grad()
    tr_loss, output = model(snp_train, gx_train)
    tr_loss.backward()
    optimizer.step()
    
    if i%100==0:
        te_loss, out_test = model(snp_test, gx_test)
        
        tr_loss_list.append(tr_loss.cpu().item())
        te_loss_list.append(te_loss.cpu().item())
        
    if i%100==0:
        print('iteration :', '%d/3000'%i, ' -  train loss :', \
              np.round(tr_loss.cpu().item(),3), '/  ', \
              'test loss :', np.round(te_loss.cpu().item(), 3))
        
        if te_loss_buff==0: te_loss_buff = te_loss.cpu().item(); continue
        
        if te_loss_buff>=te_loss.cpu().item():
            min_iter = i
            te_loss_buff = te_loss.cpu().item()
            tr_loss_buff = tr_loss.cpu().item()
            np.save('npy/trial%d/gene%d/model%d/%dtissue_trSNP2_k%d.npy'%(trial_num, gene_num, model_num, \
                                                    tissue_num, k_num), output.cpu().detach().numpy())
            np.save('npy/trial%d/gene%d/model%d/%dtissue_teSNP2_k%d.npy'%(trial_num, gene_num, model_num, \
                                                    tissue_num, k_num), out_test.cpu().detach().numpy())
    
np.save('npy/trial%d/gene%d/model%d/%dtissue_trSNP2_k%d_last.npy'%(trial_num, gene_num, model_num, \
                                                    tissue_num, k_num), output.cpu().detach().numpy())
np.save('npy/trial%d/gene%d/model%d/%dtissue_teSNP2_k%d_last.npy'%(trial_num, gene_num, model_num, \
                                                    tissue_num, k_num), out_test.cpu().detach().numpy())
np.save('npy/trial%d/gene%d/model%d/%dtissue_trGX_k%d.npy'%(trial_num, gene_num, model_num, tissue_num, \
                                                         k_num), gx_train.cpu().detach().numpy())
np.save('npy/trial%d/gene%d/model%d/%dtissue_teGX_k%d.npy'%(trial_num, gene_num, model_num, tissue_num, \
                                                         k_num), gx_test.cpu().detach().numpy())

plt.plot(tr_loss_list, 'k', label='train')
plt.plot(te_loss_list, 'r', label='test')
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Model %d , Tissue %d (Min loss %d th iter) - train : %.4f  /  test : %.4f'%(model_num, \
                                                         tissue_num, min_iter, tr_loss_buff, te_loss_buff))
plt.savefig('img/trial%d/gene%d/loss_plot/model%d_tissue%d_k%d.png'%(trial_num, gene_num, model_num, \
                                                                     tissue_num, k_num))
plt.clf()


print('\nTraining complete   //   Running time : %3.f  ------------'%(time.time()-start_time))