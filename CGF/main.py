import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import torch
import datetime
from config import read_config
from loss import StationsLoss
from utils import Plot_res, Model_select, Eval
from layers.CGF_En_De import CGF_En_De
from loss import StationsLoss_spl
from dataset import HazeData


import warnings
warnings.filterwarnings('ignore')

class Opt_model(object):
    def __init__(self, config_dict,  exp_idx):
        self.config_dict = config_dict
        # self.graph = graph
        self.hist_len = self.config_dict['HistoryTimePeriods']
        self.pred_len = self.config_dict['PredictionTimePeriods']
        self.train_data = HazeData(self.hist_len, self.pred_len,  flag='Train')
        self.in_dim = self.train_data.feature.shape[-1] + self.train_data.pm25.shape[-1]
        self.val_data = HazeData( self.hist_len, self.pred_len,  flag='Val')
        self.wind_mean, self.wind_std = self.train_data.wind_mean, self.train_data.wind_std
        self.test_data = HazeData(self.hist_len, self.pred_len, flag='Test')
        self.pm25_mean, self.pm25_std = self.test_data.pm25_mean, self.test_data.pm25_std

        self.model = self._select_model()
        self.loss_plot = Plot_res('./plot', ['train', 'val'], config_dict['Target'], 'iter', 'loss', enable=True)
        model_savepath = os.path.join(config_dict['ModelSavePath'], config_dict['SaveName'])
        if not os.path.exists(model_savepath):
            os.makedirs(model_savepath)
        self.model_select = Model_select(model_savepath,
                                         save_modelname='{}_{}'.format(config_dict['Target'], str(exp_idx)))


    def _select_model(self):
        if self.config_dict['NetName'] == 'CGF_En_De':
            model = CGF_En_De(self.config_dict)
        else:
            print('`{}` not support'.format(self.config_dict['NetName']))
            assert False
        return model


    def _get_data(self):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.config_dict['Batchsize'], shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=self.config_dict['Batchsize'], shuffle=False, drop_last=True)
        return train_loader, val_loader

    def _get_data_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.config_dict['Batchsize'], shuffle=False, drop_last=False)
        return test_loader



    def _select_optimizer(self):
        model_optim = torch.optim.RMSprop(self.model.parameters(), lr=0.0005, weight_decay=0.0005)
        return model_optim

    def _select_criterion(self):
        criterion = StationsLoss(target_gas=self.config_dict['Target'],
                                 mode='MSE')
        return criterion


    def train(self):
        mode='train'
        train_loader, val_loader=self._get_data()

        optimizer = self._select_optimizer()
        criterion = StationsLoss_spl(target_gas=self.config_dict['Target'],
                                     mode='MAE')
        criterion_val = StationsLoss(target_gas=self.config_dict['Target'],
                                     mode='MSE')

        print(self.config_dict)
        print('Target gas is {}. Model is {}'.format(self.config_dict['Target'], self.config_dict['NetName']))
        print('{} beginning train!'.format(datetime.datetime.now()))

        for epoch in range(1, self.config_dict['EpochNum']+1):
            total_loss = []
            self.model.train()
            for i, data in enumerate(train_loader):
                pm25, pm25_level, feature, time_arr = data
                pm25 = pm25.to(self.config_dict['Device'])
                feature = feature.to(self.config_dict['Device'])
                pm25_level = pm25_level.to(self.config_dict['Device'])
                pm25_label = pm25[:, self.hist_len:]
                pm25_level = pm25_level[:, self.hist_len:]
                pm25_hist = pm25[:, :self.hist_len]
                feature = feature[:, self.hist_len:]
                if epoch == 1:
                    v = torch.ones(self.config_dict["PredictionTimePeriods"] * self.config_dict['Batchsize']).cuda()
                else:
                    pre_frames_val = self.val_v(feature, pm25_hist, pm25_level, pm25_label)
                    v = criterion(pre_frames_val, pm25_label, pm25_level)

                pre_frames,loss_con = self.model(feature, pm25_hist, pm25_level, pm25_label,mode, v)
                optimizer.zero_grad()
                loss_seq = criterion_val(pre_frames, pm25_label)
                loss = loss_seq+self.config_dict['expParam']["alpha"]*loss_con

                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()

                print('TRAIN INFO: epoch:{} ({}/{}) loss_con:{:.5f} loss_seq:{:.5f}'.format(epoch, i + 1, len(train_loader),loss_con.item(),
                                                                               loss_seq.item()))
                total_loss.append(loss.item())

            thres = criterion.increase_threshold()
            train_loss = sum(total_loss) / len(total_loss)
            val_loss = self.val(val_loader, criterion_val, epoch=epoch)
            self.loss_plot.step([train_loss, val_loss])
            print('thre={} train loss={}  val loss={} '.format(thres, train_loss, val_loss))

    def val_v(self, feature, pm25_hist, pm25_level, pm25_label):
        self.model.eval()
        mode='val'
        with torch.no_grad():
                pre_frames = self.model(feature, pm25_hist, pm25_level, pm25_label,mode )
        self.model.train()
        return  pre_frames


    def val(self, val_loader, criterion, epoch):
        self.model.eval()
        mode='val'
        total_loss = []
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                pm25, pm25_level, feature, time_arr = data
                pm25 = pm25.cuda()
                feature = feature.cuda()
                pm25_level = pm25_level.to(self.config_dict['Device'])
                pm25_level = pm25_level[:, self.hist_len:]
                pm25_label = pm25[:, self.hist_len:]
                pm25_hist = pm25[:, :self.hist_len]
                feature = feature[:, self.hist_len:]
                pre_frames = self.model(feature, pm25_hist, pm25_level, pm25_label, mode)
                loss = criterion(pre_frames, pm25_label)
                total_loss.append(loss.item())
                print('VAL INFO: epoch:{} ({}/{}) loss:{:.5f}'.format(epoch, i + 1, len(val_loader), loss.item()))

        total_loss = sum(total_loss) / len(total_loss)
        self.model_select.step(total_loss, self.model, epoch=epoch)
        self.model.train()
        return total_loss


    def test(self, exp_idx):
            print('{} beginning test!'.format(datetime.datetime.now()))
            print('Test target gas is {}'.format(self.config_dict['Target']))
            mode='test'

            test_model = self._select_model()
            test_model_file = self.model_select.load_model(name=self.config_dict['Target'], exp_idx=exp_idx)
            test_model.load_state_dict(test_model_file)
            test_model = test_model.to(self.config_dict['Device'])
            test_model.eval()
            test_loader = self._get_data_test()
            sta_id = 0
            eval = Eval(self.config_dict['Target'], sta_id,  exp_idx,
                        is_save_results=True)
            id_size = 0
            for i, data in enumerate(test_loader):


                    pm25, pm25_level, feature, time_arr = data
                    # wrf_his = wrf_his.to(self.config_dict['Device'])
                    pm25 = pm25.to(self.config_dict['Device'])
                    feature = feature.to(self.config_dict['Device'])
                    pm25_level = pm25_level.to(self.config_dict['Device'])
                    pm25_label = pm25[:, self.hist_len:]
                    pm25_hist = pm25[:, :self.hist_len]
                    feature = feature[:, self.hist_len:]
                    pm25_level = pm25_level[:, self.hist_len:]
                    pre_frames = test_model(feature, pm25_hist, pm25_level, pm25_label, mode)
                    id_size += pm25.size(0)
                    sta_id_list = [sta_id for i in range(pm25.size(0))]
                    print(str(i), str(id_size), pm25.size)
                    import math
                    sample_num = len(self.test_data) / len(self.config_dict['region_index'])
                    if math.floor(id_size / sample_num) != sta_id:
                        mode_len = int(id_size % sample_num)
                        head_len = int(pm25.size(0) - mode_len)
                        sta_id_list_head = [sta_id for i in range(head_len)]
                        sta_id += 1
                        sta_id_list_mode = [sta_id for i in range(mode_len)]
                        sta_id_list = sta_id_list_head + sta_id_list_mode
                    eval.step(pre_frames.reshape(1, -1, 1), pm25_label.reshape(1, -1, 1),
                              list(np.array(sta_id_list).repeat(24)), self.pm25_mean, self.pm25_std)
            eval.get_scores()



if __name__ == "__main__":
    config_dict = read_config()
    exp_repeat = config_dict['exp_repeat']
    if config_dict['IsTrain']:

        for exp_idx in range(exp_repeat):
            opt = Opt_model(config_dict=config_dict, exp_idx=exp_idx)
            opt.train()
    elif config_dict['IsTrain'] == False:

        for exp_idx in [0]:#range(exp_repeat):
            opt = Opt_model(config_dict=config_dict, exp_idx=exp_idx)
            opt.test(exp_idx)


