import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import Contrastive
import  numpy as np
class EncoderRNN(nn.Module):
    def __init__(self, config):
        super(EncoderRNN, self).__init__()
        self.hidR = config['hidRNN'];
        self.GRU1 = nn.GRU(1, self.hidR);
        self.dropout = nn.Dropout(p=config['dropout']);
    def forward(self, x):
        xn = x.permute(1, 0, 2)
        _, r = self.GRU1(xn);
        r = self.dropout(r);
        return _, r;

class DecoderRNN(nn.Module):
    def __init__(self,  config):
        super(DecoderRNN, self).__init__()
        self.feature_size = 13#12auxiliary features and a PM2.5 concentration
        self.config = config
        self.prediction_length = self.config['PredictionTimePeriods']
        self.hidden_size = config['hidRNN']
        self.output_size = config['output_size']
        self.gru_cell = nn.GRUCell(self.feature_size, self.hidden_size)
        self.input_array = np.array([0, 1, 2, 3, 4])
        self.embedding = nn.Embedding(len(self.input_array),self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.ConLoss = Contrastive()

    def crl_emb(self, st_level):
        batch_size, time, _ = st_level.shape
        st_level_index = st_level.tolist()
        inputs = torch.Tensor([0, 1, 2, 3, 4]).to(dtype=torch.long).cuda()
        index_array = np.expand_dims(np.expand_dims(self.input_array, 0), 0).repeat(batch_size, 0).repeat(time, 1)
        whole_list = np.array(index_array)
        sample = self.embedding(inputs)
        pos_embedding = sample[st_level_index, :]
        C_embedding = sample[whole_list, :]
        return pos_embedding, C_embedding

    def csl(self, tgt_seq, current_pm25, v):
        sta_tgt = tgt_seq[:, :, :1]
        bs, ts, feature = sta_tgt.shape
        tgt = sta_tgt
        #the previous timestamps of real PM2.5
        tgt = torch.cat((current_pm25.unsqueeze(-1), tgt[:, :-1, :]), 1)

        #to filter some timestamps with nan values or 0 values
        mask = v.view(bs, ts, 1)
        zero_mask = torch.zeros_like(mask)
        mask = torch.where((tgt == tgt) & (tgt != 0), mask, zero_mask)
        return tgt, mask

    def forward(self, auxiliary, current_pm25, hn, st_level, tgt_seq, mode, v=0):
        if mode == "train":

            pos_embedding, C_embedding = self.crl_emb(st_level)
            tgt, mask = self.csl(tgt_seq, current_pm25, v)
            pm25_pred = []
            hn = hn[0,:,:]
            loss_con = torch.tensor([0], device=auxiliary.device, dtype=auxiliary.dtype)
            for i in range(self.prediction_length):
                pos_emb = pos_embedding[:,i,:,:]
                C_emb = C_embedding[:,i,:,:]

                if i >=1:
                    dec_input = torch.where(mask[:,i,:]==1, tgt[:,i,:], current_pm25.detach())
                else:
                    dec_input = current_pm25
                xn = torch.cat([dec_input, auxiliary[:, i, :]], -1)
                hn = self.gru_cell(xn,hn)
                loss_con += self.ConLoss(hn, pos_emb, C_emb)
                output = self.out(hn)
                pm25_pred.append(output)
                current_pm25 = output
            pm25_pred = torch.stack(pm25_pred, dim=1)

            return pm25_pred, loss_con/self.prediction_length
        else:
            pm25_pred = []
            hn = hn[0, :, :]


            for i in range(self.prediction_length):
                xn = torch.cat([current_pm25, auxiliary[:, i, :]], -1)
                hn = self.gru_cell(xn, hn)
                output = self.out(hn)
                pm25_pred.append(output)
                current_pm25 = output
            pm25_pred = torch.stack(pm25_pred, dim=1)

            return pm25_pred

    def initHidden(self):
        return torch.zeros(1, 128, self.hidden_size)

class CGF_En_De(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = EncoderRNN(config).cuda()
        self.decoder = DecoderRNN(config).cuda()

    def forward(self, auxiliary, st_his, st_level, tgt_seq, mode, v=0):
        encoder_outputs, encoder_hidden = self.encoder(st_his)
        # decoder_input = auxiliary  #X_is_wrf_pr;edict
        decoder_hidden = encoder_hidden
        current_pm25 = st_his[:, -1, :1]
        if mode == "train":
            output,loss_con = self.decoder(
                auxiliary, current_pm25, decoder_hidden, st_level, tgt_seq, mode, v)

            return output, loss_con
        else:
            output = self.decoder(
                auxiliary, current_pm25, decoder_hidden, st_level, tgt_seq, mode)

            return output



