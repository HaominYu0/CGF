import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
# from util import config, file_dir

from datetime import datetime
import numpy as np
import arrow
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils import data
from config import read_config

config_dict = read_config()

class HazeData(data.Dataset):

    def __init__(self, hist_len=1,
                       pred_len=24,
                       flag='Train',
                       ):

        if flag == 'Train':
            start_time_str = 'train_start'
            end_time_str = 'train_end'
        elif flag == 'Val':
            start_time_str = 'val_start'
            end_time_str = 'val_end'
        elif flag == 'Test':
            start_time_str = 'test_start'
            end_time_str = 'test_end'
        else:
            raise Exception('Wrong Flag!')
        self.config_dict = config_dict


        self.start_time = self._get_time(self.config_dict["dataset"][start_time_str])
        self.end_time = self._get_time(self.config_dict["dataset"][end_time_str])
        self.data_start = self._get_time(self.config_dict['data_start'])
        self.data_end = self._get_time(self.config_dict['data_end'])



        self.knowair_fp = self.config_dict["File_source"]
        self.region_index = self.config_dict["region_index"]
        self.city_num = len(self.region_index)

        self._load_npy()
        self._gen_time_arr()
        self._process_time()
        self._process_feature()

        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)
        self.pm25_level = self._pm25_level(self.pm25, self.city_num)
        self._calc_mean_std()
        seq_len = hist_len + pred_len
        self._add_time_dim(seq_len)
        self._norm()
        self.data_list = self._generate_list(len(self.pm25),self.pm25.shape[2])

    def _norm(self):
        self.feature = (self.feature - self.feature_mean) / self.feature_std
        self.pm25 = (self.pm25 - self.pm25_mean) / self.pm25_std

    def _pm25_level(self, pm25, pred_len):
        pm25_level = np.zeros([len(self.pm25),pred_len, 1], dtype=np.float32)
        pm25_level0 = 0*np.ones([len(self.pm25),pred_len, 1], dtype=np.float32)
        pm25_level1= np.ones([len(self.pm25),pred_len, 1], dtype=np.float32)
        pm25_level2 = 2*np.ones([len(self.pm25),pred_len, 1], dtype=np.float32)
        pm25_level3 = 3*np.ones([len(self.pm25),pred_len, 1], dtype=np.float32)
        pm25_level4 = 4*np.ones([len(self.pm25),pred_len, 1], dtype=np.float32)
        pm25_level5 = 5*np.ones([len(self.pm25),pred_len, 1], dtype=np.float32)
        level_list = config_dict['AirLevel']['PM25']
        pm25_level = np.where(pm25!=pm25, pm25_level5, pm25_level)
        pm25_level = np.where((pm25<level_list[1])&(pm25>=level_list[0]),pm25_level0, pm25_level )
        pm25_level = np.where((pm25<level_list[2])&(pm25>=level_list[1]),pm25_level1, pm25_level )
        pm25_level = np.where((pm25<level_list[3])&(pm25>=level_list[2]),pm25_level2, pm25_level )
        pm25_level = np.where((pm25<level_list[4])&(pm25>=level_list[3]),pm25_level3, pm25_level )
        pm25_level = np.where(pm25>=level_list[4],pm25_level4, pm25_level )
        return pm25_level


    def _generate_list(self, sam_num, city_num):
        data_list = []
        for sd in range(sam_num):
            for cd in range(city_num):
                data_index=str(sd)+"_"+str(cd)
                data_list.append(data_index)
        return data_list



    def _add_time_dim(self, seq_len):

        def _add_t(arr, seq_len):
            t_len = arr.shape[0]
            assert t_len > seq_len
            arr_ts = []
            for i in range(seq_len, t_len):
                arr_t = arr[i-seq_len:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts

        self.pm25 = _add_t(self.pm25, seq_len)
        self.pm25_level = _add_t(self.pm25_level, seq_len)
        self.feature = _add_t(self.feature, seq_len)
        self.time_arr = _add_t(self.time_arr, seq_len)

    def _calc_mean_std(self):
        self.feature_mean = self.feature.mean(axis=(0,1))
        self.feature_std = self.feature.std(axis=(0,1))
        self.wind_mean = self.feature_mean[-2:]
        self.wind_std = self.feature_std[-2:]
        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()

    def _process_feature(self):
        metero_var = self.config_dict['metero_var']
        metero_use = self.config_dict['metero_use']
        metero_idx = [metero_var.index(var) for var in metero_use]
        self.feature = self.feature[:,:,metero_idx]

        u = self.feature[:, :, -2] * units.meter / units.second
        v = self.feature[:, :, -1] * units.meter / units.second
        speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude
        direc = mpcalc.wind_direction(u, v)._magnitude

        h_arr = []
        w_arr = []
        for i in self.time_arrow:
            h_arr.append(i.hour)
            w_arr.append(i.isoweekday())
        h_arr = np.stack(h_arr, axis=-1)
        w_arr = np.stack(w_arr, axis=-1)
        h_arr = np.repeat(h_arr[:, None], self.city_num, axis=1)
        w_arr = np.repeat(w_arr[:, None], self.city_num, axis=1)

        self.feature = np.concatenate([self.feature, h_arr[:, :, None], w_arr[:, :, None],
                                       speed[:, :, None], direc[:, :, None]
                                       ], axis=-1)

    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        self.pm25 = self.pm25[start_idx: end_idx+1, :]
        self.feature = self.feature[start_idx: end_idx+1, :]
        self.time_arr = self.time_arr[start_idx: end_idx+1]
        self.time_arrow = self.time_arrow[start_idx: end_idx + 1]

    def _gen_time_arr(self):
        self.time_arrow = []
        self.time_arr = []
        for time_arrow in arrow.Arrow.interval('hour', self.data_start, self.data_end.shift(hours=+3), 3):
            self.time_arrow.append(time_arrow[0])
            self.time_arr.append(time_arrow[0].timestamp)
        self.time_arr = np.stack(self.time_arr, axis=-1)

    def _load_npy(self):
        self.knowair = np.load(self.knowair_fp)
        self.knowair = self.knowair[:, self.region_index, :]
        self.feature = self.knowair[:,:,:-1]
        self.pm25 = self.knowair[:,:,-1:]

    def _get_idx(self, t):
        t0 = self.data_start
        return int((t.timestamp - t0.timestamp) / (60 * 60 * 3))

    def _get_time(self, time_yaml):
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
        return arrow_time

    def __len__(self):
        return len(self.pm25)*self.pm25.shape[2]

    def __getitem__(self, index):
        data = int(self.data_list[index].split('_')[0])
        city = int(self.data_list[index].split('_')[1])

        return self.pm25[data, :,city,:],self.pm25_level[data, :,city,:], self.feature[data, :,city,:], self.time_arr[data]


