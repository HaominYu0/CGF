def read_config():
    config_dict = {}
    config_dict["File_source"] = "data/KnowAir.npy"
    config_dict['Target'] = 'PM25'
    config_dict['NetName'] = 'CGF_En_De'

    config_dict['SaveName'] = config_dict['NetName']

    config_dict['IsTrain'] = False
    config_dict['expParam'] = {"tau": 0.3, "alpha": 1, "gamma": 0.15}
    config_dict['AirLevel'] = {'PM25': [0, 25, 50, 100, 300]}

    config_dict['exp_repeat'] = 5
    config_dict['dataset_num']=1
    config_dict['data_start']=[[2015, 1, 1, 0, 0],  'GMT']
    config_dict['data_end']=[[2018, 12, 31, 21, 0],  'GMT']
    config_dict["dataset"]={'train_start':[[2015, 1, 1], 'GMT'],"train_end":[[2016, 12, 31], 'GMT'],
                            "val_start":[[2017, 1, 1],  'GMT'],"val_end":[[2017, 12, 31],  'GMT'],
                            "test_start":[[2018, 1, 1],  'GMT'],"test_end":[[2018, 12, 31],  'GMT']}

    config_dict["region_index"]=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    config_dict['spl_param'] = {"lambda": 0.1, "growth": 1.1}


    config_dict['ModelSavePath'] = './checkpoints'

    config_dict['HistoryTimePeriods'] = 168
    config_dict['PredictionTimePeriods'] = 24


    config_dict['EpochNum'] = 80
    config_dict['Batchsize'] = 512
    config_dict['LearningRate'] = 0.01
    config_dict['Device'] = 'cuda'
    config_dict['metero_var']=['100m_u_component_of_wind',
                                 '100m_v_component_of_wind',
                                 '2m_dewpoint_temperature',
                                 '2m_temperature',
                                 'boundary_layer_height',
                                 'k_index',
                                 'relative_humidity+950',
                                 'relative_humidity+975',
                                 'specific_humidity+950',
                                 'surface_pressure',
                                 'temperature+925',
                                 'temperature+950',
                                 'total_precipitation',
                                 'u_component_of_wind+950',
                                 'v_component_of_wind+950',
                                 'vertical_velocity+950',
                                 'vorticity+950']
    config_dict['metero_use']=['2m_temperature',
                               'boundary_layer_height',
                               'k_index',
                               'relative_humidity+950',
                               'surface_pressure',
                               'total_precipitation',
                               'u_component_of_wind+950',
                               'v_component_of_wind+950',]



    config_dict["hidRNN"] = 50

    config_dict['dropout'] = 0.2
    config_dict['output_size'] = 1






    return config_dict

