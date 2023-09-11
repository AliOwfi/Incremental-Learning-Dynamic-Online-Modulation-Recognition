
from bic.trainer import Trainer
from utils import *
from data_utils  import *
import pickle as pkl    

def train_bic(scenario_name, task_num, rnd_order, dataset_name, exemp_num, lr, n_epochs, seed=0):


    batch_size = 128

    ds_dict, task_order, im_sz, class_num, emb_fact, total_cls = get_dataset_specs_class_inc(seed=seed, task_num=task_num, 
                                                                                rnd_order=False, dataset=dataset_name,
                                                                                eval_ratio=0.1, order=None) 

    save_dict = {}  
    save_dict['scenario'] = scenario_name
    save_dict['model_type'] = 'resnet'    
    save_dict['dataset'] = 'split_cifar100'
    save_dict['optim_name'] = 'sgd'    
    save_dict['class_num'] = class_num  
    save_dict['bs'] = batch_size
    save_dict['lr'] = 1. 
    save_dict['n_epochs'] = n_epochs
    save_dict['task_num'] = task_num    
    save_dict['task_order'] = task_order
    save_dict['seed'] = seed    
    save_dict['emb_fact'] = emb_fact  


    trainer = Trainer(total_cls, ds_dict=ds_dict)

    trainer.train(batch_size, n_epochs, lr, exemp_num)

    save_dict['model'] = trainer.model.state_dict()
    save_name = generate_save_name(save_dict)
    pkl.dump(save_dict, open(f'{save_name}.pkl', 'wb'))
    

train_bic(scenario_name='clean_bic', task_num=10, rnd_order=False, 
          dataset_name='split_modulation', exemp_num=2000, lr=0.1, n_epochs=10, seed=0)
