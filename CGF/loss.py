import torch
import torch.nn.functional as F
import numpy as np
from config import read_config
config_dict = read_config()
class StationsLoss(torch.nn.Module):
    def __init__(self, target_gas,  mode):
        super(StationsLoss, self).__init__()
        self.target_gas = target_gas
        self.mode = mode

    def forward(self, sta_pre, sta_tgt):
        sta_tgt = sta_tgt[:, :, :1]
        tgt = sta_tgt
        if self.mode == 'MSE':
                        loss = (tgt - sta_pre)**2
                        loss = loss[(sta_tgt == sta_tgt)&(sta_tgt!=0)]
                        if loss.shape[0]==0:
                            loss = torch.tensor([0], device=sta_pre.device, dtype=sta_pre.dtype)
                        else:
                            loss = torch.mean(loss)
        elif self.mode == 'MAE':
                        loss = torch.abs(tgt - sta_pre)
                        loss = loss[(sta_tgt == sta_tgt)&(sta_tgt!=0)]
                        if loss.shape[0]==0:
                            loss = torch.tensor([0], device=sta_pre.device, dtype=sta_pre.dtype)
                        else:
                            loss = torch.mean(loss)
        return loss


class StationsLoss_spl(torch.nn.Module):
    def __init__(self, target_gas, mode):
        super(StationsLoss_spl, self).__init__()
        self.target_gas = target_gas
        self.mode = mode
        self.threshold =config_dict['spl_param']["lambda"]#0.05
        self.gamma =config_dict['expParam']["gamma"]
        self.growing_factor = config_dict['spl_param']["growth"]


    def forward(self, sta_pre, sta_tgt, sta_level):
        sta_tgt = sta_tgt[:, :, :1]
        tgt = sta_tgt
        sta_level = sta_level
        loss_smape = 2 * torch.abs(tgt - sta_pre) / (torch.abs(tgt) + torch.abs(sta_pre))
        loss_zero = torch.ones_like(sta_tgt) * 3
        loss_smape = torch.where((sta_tgt == sta_tgt) & (sta_tgt != 0), loss_smape, loss_zero)
        loss_smape = loss_smape.view(-1)
        sta_level = sta_level.reshape(-1)
        v = self.spl_loss(loss_smape, sta_level).type(torch.cuda.FloatTensor)

        return 1 - v


    def increase_threshold(self):
        self.threshold *= self.growing_factor
        return self.threshold

    def spl_loss(self, super_loss, sta_level):
        index_order = super_loss.sort()[1]
        list_order=[list(index_order[sta_level ==0].sort()[0].cpu().numpy()) if len(index_order[sta_level == 0])!=0 else [],
        list(index_order[sta_level ==1].sort()[0].cpu().numpy()) if len(index_order[sta_level == 1])!=0 else [],
        list(index_order[sta_level == 2].sort()[0].cpu().numpy()) if len(index_order[sta_level == 2])!=0 else [],
        list(index_order[sta_level == 3].sort()[0].cpu().numpy()) if len(index_order[sta_level == 3])!=0 else [],
        list(index_order[sta_level == 4].sort()[0].cpu().numpy()) if len(index_order[sta_level == 4])!=0 else []
                    ]
        list_order_len = max([len(list_order[i]) for i in range(len(list_order))])
        list_order = [list_order[0] + [-9 for i in range(list_order_len - len(list_order[0]))] if len(list_order[0]) < list_order_len else list_order[0],
                      list_order[1] + [-9 for i in range(list_order_len - len(list_order[1]))] if len(list_order[1]) < list_order_len else list_order[1],
                      list_order[2] + [-9 for i in range(list_order_len - len(list_order[2]))] if len(list_order[2]) < list_order_len else list_order[2],
                      list_order[3] + [-9 for i in range(list_order_len - len(list_order[3]))] if len(list_order[3]) < list_order_len else list_order[3],
                      list_order[4] + [-9 for i in range(list_order_len - len(list_order[4]))] if len(list_order[4]) < list_order_len else list_order[4]]
        sta_level_list = sta_level.cpu().numpy().tolist()
        res_index= np.array(list_order)[np.array(sta_level_list).astype(np.int).tolist()]
        index_order_array = index_order.cpu().numpy()
        tmp_a= np.expand_dims(index_order_array,-1).repeat(len(res_index[0]),-1)

        tmp = np.expand_dims(np.array(range(len(res_index[0]))),0).repeat(len(res_index),0)
        spl_array = tmp[res_index==tmp_a]
        spl_array = torch.tensor(spl_array).type(torch.FloatTensor).cuda()

        v = super_loss < (self.threshold +self.gamma/(torch.sqrt(spl_array+1)+torch.sqrt(spl_array)))

        return v.int()

class Contrastive(torch.nn.Module):
    def __init__(self):
        super(Contrastive, self).__init__()

    def forward(self, inputs, pos, C_emb):
        tau = config_dict['expParam']["tau"]
        batch_size, K_size, feature_size  = C_emb.shape
        _pos = torch.exp(F.cosine_similarity(inputs, pos[:,0,:], -1, 1e-6)/tau)
        Anchor_n = inputs.unsqueeze(1).expand(inputs.shape[0], K_size, feature_size)
        triloss = torch.sum(torch.exp(F.cosine_similarity(Anchor_n, C_emb, -1, 1e-6) / tau), -1)
        conloss = torch.mean(-1*torch.log(_pos/(triloss)))
        return conloss
