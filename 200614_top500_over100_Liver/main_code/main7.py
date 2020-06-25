import os
import time
import argparse
import module2 as module
import numpy as np
import matplotlib.pyplot as plt
from model7 import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init


trial_num = 7
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

lasso_alpha = 0.05
input_size = 64
attend_size = 64
hidden_size = 32

gene_data_name = 'UTMOST_top500_over100_Liver'
gene_list = os.listdir('../../%s/'%gene_data_name)
gene_name = gene_list[gene_num-1]

print('\n\n[Gene %d] Model %d ( tissue %d ) - %d/5 fold data'%(gene_num, model_num, tissue_num, k_num))
print('Option - input size : %d / attend size : %d / hidden size : %d / lasso coeff : %.5f '\
      %(input_size, attend_size, hidden_size, lasso_alpha))
    
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
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_time = time.time()
np.random.seed(37)
torch.manual_seed(37)
torch.cuda.manual_seed_all(37)
torch.backends.cudnn.deterministic = True

snp, gx = module.load_data(gene_data_name, gene_name, tissue_num, proc=True)
snp_len = np.shape(snp)[-1]
len_dif = snp_len%input_size
seq_len = int(snp_len/input_size)

snp = snp[:, int(len_dif/2):int(len_dif/2)+snp_len-len_dif]

snp_train, snp_test, gx_train, gx_test = module.k_fold_data(snp, gx, 5, k_num)

snp_train = torch.Tensor(snp_train).to(device).view(-1, seq_len, input_size)
snp_test = torch.Tensor(snp_test).to(device).view(-1, seq_len, input_size)

gx_train = torch.Tensor(gx_train).to(device)
gx_test = torch.Tensor(gx_test).to(device)

print('\nData shape @@@@@@')
print('Train data : ', np.shape(snp_train),' / ', np.shape(gx_train))
print('Test data : ', np.shape(snp_test), ' / ', np.shape(gx_test))
print('\n')


learning_rate = 0.00005
model = nn.DataParallel(Model(input_size=input_size, attend_size=attend_size, 
                              hidden_size=hidden_size, seq_len=seq_len)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=learning_rate*0.001)

def loss_func(output, target, weight_norm, lasso_coeff):
    l = torch.mean((output-target)**2)
    lasso_term = lasso_coeff*torch.mean(weight_norm)
    return l+lasso_term, l

tr_loss_list = []
te_loss_list = []

tr_loss_buff = 0
te_loss_buff = 0
min_iter = 0

mb_div = 3
mb_idx = int(len(snp_train)/mb_div)
s = np.arange(len(snp_train))

for i in range(2001):
    np.random.shuffle(s)
    snp_train = snp_train[s]
    gx_train = gx_train[s]
    
    model.train()
    
    for mb in range(mb_div):
        dsnp_train = snp_train[mb*mb_idx:(mb+1)*mb_idx]
        dgx_train = gx_train[mb*mb_idx:(mb+1)*mb_idx]
        
        optimizer.zero_grad()
        output, l1_tr = model(dsnp_train, dgx_train)
        tr_loss, _ = loss_func(output, dgx_train, l1_tr, lasso_alpha)
        tr_loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    if i%100==0:
        model.eval()
        output, l1_tr = model(snp_train, gx_train)
        tr_loss, tr_loss2 = loss_func(output, gx_train, l1_tr, lasso_alpha)
        out_test, l1_te = model(snp_test, gx_test)
        te_loss, te_loss2 = loss_func(out_test, gx_test, l1_te, lasso_alpha)
        
        tr_loss_list.append(tr_loss2.cpu().item())
        te_loss_list.append(te_loss2.cpu().item())
        
    if i%100==0:
        print('iteration :', '%d/2000'%i, ' -  train loss :', \
              np.round(tr_loss2.cpu().item(),3), '/  ', \
              'test loss :', np.round(te_loss2.cpu().item(), 3))
        
        if te_loss_buff==0: te_loss_buff = te_loss2.cpu().item(); continue
        
        if te_loss_buff>=te_loss2.cpu().item():
            min_iter = i
            te_loss_buff = te_loss2.cpu().item()
            tr_loss_buff = tr_loss2.cpu().item()
            np.save('../npy/trial%d/gene%d/model%d/%dtissue_trSNP2_k%d.npy'%(trial_num, gene_num, model_num, \
                                                    tissue_num, k_num), output.cpu().detach().numpy())
            np.save('../npy/trial%d/gene%d/model%d/%dtissue_teSNP2_k%d.npy'%(trial_num, gene_num, model_num, \
                                                    tissue_num, k_num), out_test.cpu().detach().numpy())
            
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