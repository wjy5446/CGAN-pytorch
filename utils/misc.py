import os
import re
import datetime
import numpy as np

import torch
import torchvision.utils as tutils

####
# save/load model
####
def save_model(root, name, model, dict_postfix=None):
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    path_save_dir = os.path.join(root, date_str)

    if not os.path.isdir(path_save_dir):
        os.mkdir(path_save_dir)

    filename = name
    if dict_postfix is not None:
        for key, value in dict_postfix.items():
            filename += ("_" + key + "-" + str(value))

    filename += '.pth'

    path_save_file = os.path.join(path_save_dir, filename)

    torch.save(model.state_dict(), path_save_file)

    print('Save Model !!')

def load_model(root, name, key, ascending=True, dir=None, device='cpu'):
    if dir is None:
        pattern = '^[0-9]{4}\-[0-9]{2}\-[0-9]{2}$'

        list_dir = [datetime.datetime.strptime(name, '%Y-%m-%d') for name in os.listdir(root) if re.match(pattern, name)]
        sorted(list_dir, reverse=True)
        dir = list_dir[0].strftime('%Y-%m-%d')

        path_folder = os.path.join(root, dir)
    else:
        path_folder = os.path.join(root, dir)

    # make dict info
    list_info = []
    list_filename = []
    for filename in os.listdir(path_folder):
        if filename.endswith('.pth'):
            dict_info = {}

            list_filename.append(filename)
            list_info_str = filename.split('.pth')[0].split('_')

            for info in list_info_str:
                if '-' in info:
                    tmp = info.split('-')
                    dict_info[tmp[0]] = float(tmp[1])
                else:
                    dict_info['name'] = info

            list_info.append(dict_info)

    # sort value by key
    list_sort = []
    for info in list_info:
        if info.get(key):
            list_sort.append(info.get(key))

    if ascending:
        idx = np.argsort(list_sort)[0]
    else:
        idx = np.argsort(list_sort)[-1]

    filename_load = list_filename[idx]

    print('Load Model : dir({dir}), filename({filename})'.format(dir=dir, filename=filename_load))


    if device == 'cpu':
        return torch.load(os.path.join(path_folder, filename_load), map_location=lambda storage, loc: storage)
    else:
        return torch.load(os.path.join(path_folder, filename_load))

