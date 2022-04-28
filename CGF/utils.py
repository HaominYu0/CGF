import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import config


class Plot_res(object):
    def __init__(self, plot_save_dir, plot_datainfo, plot_title, plot_xname, plot_yname, enable=False):
        self.data = []
        self.plot_save_dir = plot_save_dir
        self.plot_datainfo = plot_datainfo
        self.plot_title = plot_title
        self.plot_xname = plot_xname
        self.plot_yname = plot_yname
        self.enable = enable
        self.runtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    def step(self, data):
        assert len(data) == len(self.plot_datainfo)
        self.data.append(data)
        self.save()

    def save(self):
        if self.enable:
            data = np.array(self.data)
            for i in range(len(self.plot_datainfo)):
                x = range(len(self.data))
                plt.plot(x, data[:, i], label=self.plot_datainfo[i])
            plt.title(self.plot_title)
            # plt.ylim([0, 0.2])
            plt.xlabel(self.plot_xname)
            plt.ylabel(self.plot_yname)
            if not os.path.exists(self.plot_save_dir):
                os.makedirs(self.plot_save_dir)
            plt.legend()
            plt.savefig(os.path.join(self.plot_save_dir, '{}.jpg'.format(self.runtime)))
            plt.close()


class Model_select(object):
    def __init__(self, save_path, save_modelname):
        self.save_path = save_path
        self.save_modelname = save_modelname
        self.loss = 10000000
        self.minloss_epoch = -1
        if not os.path.isdir(save_path):
            os.makedirs(save_path)



    def step(self, loss, model, epoch):
        if loss < self.loss:
            self.loss = loss
            self.minloss_epoch = epoch
            self.save_model(model, name=self.save_modelname)

    def save_model(self, model, name='default'):
        torch.save(model.state_dict(), os.path.join(self.save_path, '{}.pkl'.format(name)))

    def load_model(self, name='default', exp_idx=0):
        model_file = torch.load(os.path.join(self.save_path, '{}_{}.pkl'.format(name, exp_idx)))
        return model_file




class Eval(object):
    def __init__(self, target_gas, sta_id,
                 exp_idx=0,
                 is_save_results=False,
                 results_save_path='./results/ori_results',
                 scores_save_path='./results'):
        self.target_gas = target_gas
        self.exp_idx = exp_idx
        self.sta_id = sta_id
        self.is_save_results = is_save_results
        self.results_save_path = results_save_path
        self.scores_save_path = os.path.join(scores_save_path, '{}-scores.txt'.format(target_gas))
        self.result = []


        if not os.path.exists(scores_save_path):
            os.makedirs(scores_save_path)

    def __del__(self):
        csv_file = pd.DataFrame(self.result, columns=['station_code', self.target_gas, '{}_pre'.format(self.target_gas)])
        if self.is_save_results:
           for sta_id in range(len(config.read_config()["region_index"])):
               config_dict = config.read_config()
               path = os.path.join(self.results_save_path, config_dict['SaveName'])
               if not os.path.exists(path):
                   os.makedirs(path)
               path = os.path.join(path, str(sta_id))
               if not os.path.exists(path):
                   os.makedirs(path)
               csv_file_sta = csv_file[csv_file['station_code'] == sta_id]
               csv_file_sta.to_csv(os.path.join(path, self.target_gas + '_' + str(self.exp_idx) + '.csv'), index=False)

    def step(self, sta_pre, sta_tgt, sta_id, pm_mean,pm_std):
            sta_tgt = sta_tgt
            sta_pre = sta_pre[0].detach().cpu()
            sta_tgt = sta_tgt[0].detach().cpu()
            time_periods, params_num = sta_tgt.shape
            for t in range(time_periods):
                    sta_pre_atom = sta_pre[t, 0] * pm_std + \
                                   pm_mean
                    sta_tgt_atom = sta_tgt[t, 0]*pm_std+pm_mean
                    self.result.append([sta_id[t], sta_tgt_atom.item(),
                                        sta_pre_atom.item()])

    def get_scores(self):

        tmp = np.array(self.result)
        obs = np.array(tmp[:, -2])
        pre = np.array(tmp[:, -1])
        mae = self.MAE(obs, pre)
        smape = self.SMAPE(obs, pre)
        with open(self.scores_save_path, 'w') as f:
            info = '{}:\nMAE = {}\nSMAPE = {}\n'\
                .format(self.target_gas,   np.mean(mae), np.mean(smape))
            f.write(info)
            print(info)

    def SMAPE(self, obs, pre):
        pre = pre.flatten()
        obs = obs.flatten()
        cnt = 0
        err = 0
        for i in range(pre.shape[0]):
            if not (obs[i] >= 0) and not (obs[i] <= 0):
                continue
            cnt += 1
            err += np.abs(obs[i] - pre[i]) / ((np.abs(obs[i]) + np.abs(pre[i])) / 2)
        if cnt == 0:
            err = 0
        else:
            err = err / cnt
        return err

    def MAE(self, obs, pre):
        pre = pre.flatten()
        obs = obs.flatten()
        cnt = 0
        err = 0
        for i in range(pre.shape[0]):
            if not (obs[i] >= 0) and not (obs[i] <= 0):
                continue
            cnt += 1
            err += np.abs(obs[i] - pre[i])
        if cnt != 0:
            err = err / cnt
        return err



if __name__ == "__main__":
    pass


