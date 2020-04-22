from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import torch



from .AttModel_MAD_SAP import *

def setup(opt):
    if opt.caption_model == 'lstm_MAD_SAP':
        model = LSTM_MAD_SAP(opt)                   
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    save_id_real = getattr(opt, 'save_id', '')
    if save_id_real == '':
        save_id_real = opt.id

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.checkpoint_path)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.checkpoint_path, 'infos_'
                + save_id_real + format(int(opt.start_from),'04') + '.pkl')),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(
            opt.checkpoint_path, 'model' +save_id_real+ format(int(opt.start_from),'04') + '.pth')))

    return model
