import os
import time
import argparse
import module2
import numpy as np
import matplotlib.pyplot as plt
from model2 import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init


trial_num = 3
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

lasso_list = [0.003, 0.03, 0.1, 0.3, 0.5, 1]
lasso_alpha = lasso_list[model_num-1]
        
gene_data_name = 'UTMOST_low200_Liver'
gene_list = os.listdir('../../%s/'%gene_data_name)
gene_name = gene_list[gene_num-1]

print('\n\n[Gene %d] Model %d ( tissue %d ) - %d/5 fold data'%(gene_num, model_num, tissue_num, k_num))
print('Option : lasso coeff : %.4f'%lasso_alpha)

try:
    if not os.path.exists('../ckpt/trial%d/gene%d/'%(trial_num, gene_num)): 
        os.mkdir('../ckpt/trial%d/gene%d/'%(trial_num, gene_num))
except FileExistsError:
    print('Already exist folder')
    
try:
    if not os.path.exists('../ckpt/trial%d/gene%d/model%d'%(trial_num, gene_num, model_num)): 
        os.mkdir('../ckpt/trial%d/gene%d/model%d'%(trial_num, gene_num, model_num))
except FileExistsError:
    print('Already exist folder')
    
try:
    if not os.path.exists('../npy/trial%d/gene%d/'%(trial_num, gene_num)): 
        os.mkdir('../npy/trial%d/gene%d/'%(trial_num, gene_num))
except FileExistsError:
    print('Already exist folder')
    
try:
    if not os.path.exists('../npy/trial%d/gene%d/model%d'%(trial_num, gene_num, model_num)): 
        os.mkdir('../npy/trial%d/gene%d/model%d'%(trial_num, gene_num, model_num))
except FileExistsError:
    print('Already exist folder')

try:
    if not os.path.exists('../img/trial%d/gene%d'%(trial_num, gene_num)): 
        os.mkdir('../img/trial%d/gene%d'%(trial_num, gene_num))
except FileExistsError:
    print('Already exist folder')
    
try:
    if not os.path.exists('../img/trial%d/gene%d/loss_plot'%(trial_num, gene_num)): 
        os.mkdir('../img/trial%d/gene%d/loss_plot'%(trial_num, gene_num))
except FileExistsError:
    print('Already exist folder')
    
    
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

start_time = time.time()
np.random.seed(37)
torch.manual_seed(37)
torch.cuda.manual_seed_all(37)
torch.backends.cudnn.deterministic = True

snp, gx = module2.load_data(gene_data_name, gene_name, tissue_num, proc=True)
snp_train, snp_valid, snp_test, gx_train, gx_valid, gx_test = module2.k_fold_data(snp, gx, 5, k_num)

snp_train = torch.Tensor(snp_train).to(device)
snp_valid = torch.Tensor(snp_valid).to(device)
snp_test = torch.Tensor(snp_test).to(device)

gx_train = torch.Tensor(gx_train).to(device)
gx_valid = torch.Tensor(gx_valid).to(device)
gx_test = torch.Tensor(gx_test).to(device)

print('\nData shape @@@@@@')
print('Train data : ', np.shape(snp_train),' / ', np.shape(gx_train))
print('Validation data : ', np.shape(snp_valid),' / ', np.shape(gx_valid))
print('Test data : ', np.shape(snp_test), ' / ', np.shape(gx_test))
print('\n')

learning_rate = 0.00003
model = Model(snp_len=snp_train.size()[-1], alpha=lasso_alpha).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

tr_loss_list = []
te_loss_list = []
vad_loss_list = []

tr_loss_buff = 0
te_loss_buff = 0
vad_loss_buff = 0
min_iter = 0

mb_div = 2
mb_idx = int(len(snp_train)/mb_div)
s = np.arange(len(snp_train))

for i in range(5001):
    np.random.shuffle(s)
    snp_train = snp_train[s]
    gx_train = gx_train[s]
    
    for mb in range(mb_div):
        dsnp_train = snp_train[mb*mb_idx:(mb+1)*mb_idx]
        dgx_train = gx_train[mb*mb_idx:(mb+1)*mb_idx]
        
        optimizer.zero_grad()
        tr_loss, output = model(dsnp_train, dgx_train)
        tr_loss.backward()
        optimizer.step()
    
    if i%100==0:
        tr_loss, output = model(snp_train, gx_train)
        te_loss, out_test = model(snp_test, gx_test)
        vad_loss, out_vad = model(snp_valid, gx_valid)
        
        tr_loss_list.append(tr_loss.cpu().item())
        te_loss_list.append(te_loss.cpu().item())
        vad_loss_list.append(vad_loss.cpu().item())
        
    if i%100==0:
        print('iteration :', '%d/5000'%i, ' -  train loss :', \
              np.round(tr_loss.cpu().item(),3), '/  ', \
              'valid loss :', np.round(vad_loss.cpu().item(), 3), '/ ', \
              'test loss :', np.round(te_loss.cpu().item(), 3))
        
        if vad_loss_buff==0: vad_loss_buff = vad_loss.cpu().item(); continue
        
        if vad_loss_buff>=vad_loss.cpu().item():
            min_iter = i
            vad_loss_buff = vad_loss.cpu().item()
            tr_loss_buff = tr_loss.cpu().item()
            np.save('../npy/trial%d/gene%d/model%d/%dtissue_trSNP2_k%d.npy'%(trial_num, gene_num, model_num, \
                                                    tissue_num, k_num), output.cpu().detach().numpy())
            np.save('../npy/trial%d/gene%d/model%d/%dtissue_teSNP2_k%d.npy'%(trial_num, gene_num, model_num, \
                                                    tissue_num, k_num), out_test.cpu().detach().numpy())
            if k_num==1:
                torch.save(model.state_dict(), '../ckpt/trial%d/gene%d/model%d/%dtissue_trSNP2_k%d.npy'\
                       %(trial_num, gene_num, model_num,tissue_num, k_num))
                
np.save('../npy/trial%d/gene%d/model%d/%dtissue_trSNP2_k%d_last.npy'%(trial_num, gene_num, model_num, \
                                                    tissue_num, k_num), output.cpu().detach().numpy())
np.save('../npy/trial%d/gene%d/model%d/%dtissue_teSNP2_k%d_last.npy'%(trial_num, gene_num, model_num, \
                                                    tissue_num, k_num), out_test.cpu().detach().numpy())
np.save('../npy/trial%d/gene%d/model%d/%dtissue_trGX_k%d.npy'%(trial_num, gene_num, model_num, tissue_num, \
                                                         k_num), gx_train.cpu().detach().numpy())
np.save('../npy/trial%d/gene%d/model%d/%dtissue_teGX_k%d.npy'%(trial_num, gene_num, model_num, tissue_num, \
                                                         k_num), gx_test.cpu().detach().numpy())

plt.plot(tr_loss_list, 'k', label='train')
plt.plot(te_loss_list, 'r', label='test')
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Model %d , Tissue %d (Min loss %d th iter) - train : %.4f  /  test : %.4f'%(model_num, \
                                                         tissue_num, min_iter, tr_loss_buff, te_loss_buff))
plt.savefig('../img/trial%d/gene%d/loss_plot/model%d_tissue%d_k%d.png'%(trial_num, gene_num, model_num, \
                                                                     tissue_num, k_num))
plt.clf()


print('\nTraining complete   //   Running time : %3.f  ------------'%(time.time()-start_time))