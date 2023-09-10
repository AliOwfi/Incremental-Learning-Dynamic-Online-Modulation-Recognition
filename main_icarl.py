from models import * 
from utils import *
import pickle as pkl    
from torch.utils.data import DataLoader 
from data_utils import CustomTenDataset, get_dataset_specs_class_inc
import warnings


warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


class iCaRLNet(nn.Module):
    def __init__(self, feature_extractor, feature_size, n_classes):
        # Network architecture
        super(iCaRLNet, self).__init__()
        self.feature_extractor = feature_extractor  
        

        self.fc = nn.Linear(feature_size, n_classes, bias=True)

        self.n_classes = n_classes
        self.n_known = 0

        # List containing exemplar_sets
        # Each exemplar_set is a np.array of N images
        # with shape (N, C, H, W)
        self.exemplar_sets = []

        # Learning method
        self.cls_loss = nn.CrossEntropyLoss()
        self.dist_loss = nn.BCELoss()
    
        # Means of exemplars
        self.exemplar_means = []

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

    def increment_classes(self, n):
        """Add n classes in the final fc layer"""
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data
        bias  = self.fc.bias.data   

        self.fc = nn.Linear(in_features, out_features+n, bias=True)
        self.fc.weight.data[:out_features] = weight
        self.fc.bias.data[:out_features] = bias 

        self.n_classes += n

    def update_means(self): 
        self.feature_extractor.eval()   
        exemplar_means = []

        for P_y in self.exemplar_sets:
            features = []

            # Extract feature for each exemplar in P_y
            for ex in P_y:
                ex = ex.to(device)
                feature = self.feature_extractor(ex.unsqueeze(0))
                feature = feature.squeeze()
                feature.data = feature.data / feature.data.norm() # Normalize
                features.append(feature)

            features = torch.stack(features)
            mu_y = features.mean(0).squeeze()
            mu_y.data = mu_y.data / mu_y.data.norm() # Normalize
            exemplar_means.append(mu_y)

        self.exemplar_means = exemplar_means


    def classify(self, x, mode):
        """Classify images by neares-means-of-exemplars

        Args:
            x: input image batch
        Returns:
            preds: Tensor of size (batch_size,)
        """
        if mode == 0:
            self.feature_extractor.eval()   

            batch_size = x.size(0)
        
            exemplar_means = self.exemplar_means
            means = torch.stack(exemplar_means) # (n_classes, feature_size)
            means = torch.stack([means] * batch_size) # (batch_size, n_classes, feature_size)
            means = means.permute(0, 2, 1) # (batch_size, feature_size, n_classes)


            feature = self.feature_extractor(x) # (batch_size, feature_size)
            for i in range(feature.size(0)): # Normalize
                feature.data[i] = feature.data[i] / feature.data[i].norm()

            feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
            feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)

            dists = (feature - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
            _, preds = dists.min(1)

        else:
            self.eval()
            preds = self.forward(x)
            preds = torch.argmax(preds, dim=1)  

        return preds
    

    def construct_exemplar_set(self, ds, m):
        """Construct an exemplar set for image set

        Args:
            images: np.array containing images of a class
        """
        # Compute and cache features for each example
        self.feature_extractor.eval()   

        for cls in range(self.n_known, self.n_classes):
            features = []
            images = ds.data[ds.targets == cls]  
            for img in images:
                x = img.to(device)
                feature = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy()
                feature = feature / np.linalg.norm(feature) # Normalize
                features.append(feature[0])


            features = np.array(features)
            class_mean = np.mean(features, axis=0)
            class_mean = class_mean / np.linalg.norm(class_mean) # Normalize

            exemplar_set = []
            exemplar_features = [] # list of Variables of shape (feature_size,)
            for k in range(m):
                S = np.sum(exemplar_features, axis=0)
                phi = features
                mu = class_mean
                mu_p = 1.0/(k+1) * (phi + S)
                mu_p = mu_p / np.linalg.norm(mu_p)
                i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))

                exemplar_set.append(images[i])
                exemplar_features.append(features[i])
                """
                print "Selected example", i
                print "|exemplar_mean - class_mean|:",
                print np.linalg.norm((np.mean(exemplar_features, axis=0) - class_mean))
                #features = np.delete(features, i, axis=0)
                """
            
            self.exemplar_sets.append(torch.stack(exemplar_set))    
                

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]


    def combine_dataset_with_exemplars(self, dataset):
        if len(self.exemplar_sets) == 0:
            return dataset
        
        ex_data = []
        ex_targets = []
        for cls in range(self.n_known): 
            ex_data.append(self.exemplar_sets[cls])
            ex_targets.append(torch.ones(len(self.exemplar_sets[cls])) * cls)   

        ex_data = torch.cat(ex_data, dim=0) 
        ex_targets = torch.cat(ex_targets, dim=0)   

        combined_data = torch.cat((dataset.data, ex_data), dim=0)         
        combined_targets = torch.cat((dataset.targets, ex_targets), dim=0)  
        new_ds = CustomTenDataset(combined_data, combined_targets)  
        return new_ds   
    
    def serialize_variables(self):
        self.register_buffer('buffer_exemplar_sets', torch.stack(self.exemplar_sets))   
        self.register_buffer('buffer_n_classes', torch.tensor(self.n_classes)) 
        self.register_buffer('buffer_n_known', torch.tensor(self.n_known))
        self.register_buffer('buffer_exemplar_means', torch.stack(self.exemplar_means))   

    def register_variables_first_time(self, state_dict):
        self.register_buffer('buffer_exemplar_sets', state_dict['buffer_exemplar_sets'])
        self.register_buffer('buffer_n_classes', state_dict['buffer_n_classes'])
        self.register_buffer('buffer_n_known', state_dict['buffer_n_known'])
        self.register_buffer('buffer_exemplar_means', state_dict['buffer_exemplar_means'])

        

    def load_from_buffer(self):
        self.n_classes = getattr(self, 'buffer_n_classes').item()   
        self.n_known = getattr(self, 'buffer_n_known').item()

        self.exemplar_sets_ten = getattr(self, 'buffer_exemplar_sets')  
        self.exemplar_sets = [] 
        for ex in self.exemplar_sets_ten:
            self.exemplar_sets.append(ex)   
        
        self.exemplar_means_ten = getattr(self, 'buffer_exemplar_means')    
        self.exemplar_means = []
        for ex in self.exemplar_means_ten:
            self.exemplar_means.append(ex)  



def create_and_load_old_icarl(model, **kwargs):
    feat_ext_old = create_model_class_inc(task_num=1, class_num=None, 
                                          model_type=kwargs['model_type'], emb_fact=kwargs['emb_fact'], include_head=False, 
                                          nf=kwargs['nf'], final_feat_sz=kwargs['final_feat_sz'])
    
    model_old = iCaRLNet(feat_ext_old, kwargs['emb_dim'], model.n_classes)
    model_old.register_variables_first_time(model.state_dict())
    model_old.load_state_dict(model.state_dict(), strict=True)   
    model_old.load_from_buffer()    
    model_old.eval()    

    return model_old    


def construct_ds_from_all_tsks(ds, n_classes):
    ds_data = []
    ds_targets = []

    
    for cls in range(n_classes):    
        
        tmp_idx = ds.targets == cls 
        ds_data.append(ds.data[tmp_idx])    
        ds_targets.append(ds.targets[tmp_idx])

    ds_data = torch.cat(ds_data, dim=0)
    ds_targets = torch.cat(ds_targets, dim=0)
    acc_ds = CustomTenDataset(ds_data, ds_targets)  

    return acc_ds


def construct_ds_from_cls_per_task(ds_lst, task_ind):
    ds_data = []
    ds_targets = []

    for t_id in range(task_ind+1):
        ds_data.append(ds_lst[t_id].data)   
        ds_targets.append(ds_lst[t_id].targets) 

    ds_data = torch.cat(ds_data, dim=0)
    ds_targets = torch.cat(ds_targets, dim=0)
    acc_ds = CustomTenDataset(ds_data, ds_targets)  

    return acc_ds

def icarl_lr_sch(epoch, init_lr, optim):
    if epoch < 48:  
        new_lr = init_lr

    elif epoch >= 48 and epoch < 62:
        new_lr = init_lr * 0.2

    elif epoch >= 62 and epoch < 80:
        new_lr = init_lr * 0.04

    elif epoch >= 80:
        new_lr =  init_lr / 125.
    
    for param_group in optim.param_groups:
        param_group['lr'] = new_lr
        

    return  optim  


def train_icarl(scenario_name, task_num, n_epochs, seed=0, dataset='pmnist', model_type='mlp', lr=1e-2, 
              optim_name='sgd', bs=16, emb_dim=2048, exemp_num=2000):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    order = np.random.permutation(10)
    ds_dict, task_order, im_sz, class_num, emb_fact = get_dataset_specs_class_inc(task_num=task_num, order=order, dataset=dataset, seed=seed)
    feat_ext = create_model_class_inc(task_num=1, nf=64, final_feat_sz=1, 
                                      class_num=None, model_type=model_type, emb_fact=emb_fact, include_head=False)
    


    feat_ext = feat_ext.to(device)
    model = iCaRLNet(feat_ext, emb_dim, class_num)
    model = model.to(device)      
    model.train()     

    accumulative_acc_lst = []  
    acc_mat = np.zeros((task_num, task_num))    
  

    save_dict = {}  
    save_dict['scenario'] = scenario_name
    save_dict['model_type'] = model_type    
    save_dict['dataset'] = dataset
    save_dict['optim_name'] = optim_name    
    save_dict['class_num'] = class_num  
    save_dict['bs'] = bs
    save_dict['lr'] = lr
    save_dict['n_epochs'] = n_epochs
    save_dict['model'] = model.state_dict()
    save_dict['model_name'] = model.__class__.__name__
    save_dict['task_num'] = task_num    
    save_dict['task_order'] = task_order
    save_dict['seed'] = seed    
    save_dict['emb_fact'] = emb_fact  
    save_dict['emb_dim'] = emb_dim  

    cont_method_args = {'method':'icarl', 'total_exemplar_num':exemp_num} 
    save_dict['cont_method_args'] = cont_method_args    
    

    for task_ind in range(task_num): 
        ds_train = ds_dict['train'][task_ind]
        ds_tst = ds_dict['test'][task_ind]

        comb_train_ds = model.combine_dataset_with_exemplars(ds_train)  
        dl_train = DataLoader(comb_train_ds, batch_size=bs, shuffle=True)    
        
        if task_ind > 0:    
            old_model = create_and_load_old_icarl(model, model_type=model_type, emb_fact=emb_fact, 
                                                  emb_dim=emb_dim, nf=64, final_feat_sz=1)    
            
            old_model.increment_classes(len(task_order[task_ind]))  
            old_model.to(device)    
            model.increment_classes(len(task_order[task_ind]))  

        model = model.to(device)    
        optim = create_optimizer(model, optim_name, lr, weight_decay=1e-5) 
        loss_ = []

        for epoch in range(n_epochs):
            model.train()
            # optim = icarl_lr_sch(epoch, lr, optim)
            for i, (x, y) in enumerate(dl_train):
                x, y = x.to(device), y.type(torch.int64).to(device)           
    
                y_hat = model(x)
                y_one_hot = torch.nn.functional.one_hot(y, num_classes=model.n_classes).float().to(device) 

                if task_ind > 0:
                    y_hat_old = torch.sigmoid(old_model(x)).detach()    
                    y_one_hot[:, :model.n_known] = y_hat_old[:, :model.n_known]
                
                
                loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y_one_hot)   

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)   
                optim.step()

                loss_.append(loss.item())
            
            
            print(f'task {task_ind} epoch {epoch} loss: {loss.item()}')

        exemp_per_cls = exemp_num // model.n_classes
        model.reduce_exemplar_sets(exemp_per_cls)
        model.construct_exemplar_set(ds_train, exemp_per_cls)
        model.update_means()   

        model.n_known = model.n_classes

        
        model.serialize_variables()
    
        save_dict['model'] = model.state_dict()

        accumulaive_ds = construct_ds_from_cls_per_task(ds_dict['test'], task_ind)
        dl_tst = DataLoader(accumulaive_ds, batch_size=bs, shuffle=True)
        accumulative_acc = eval_dl(model, dl_tst, verbose=False, task_id=task_ind, class_inc=True)
        accumulative_acc_lst.append(accumulative_acc) 

        for test_id in range(task_ind+1):
            ds_tst = ds_dict['test'][test_id]
            dl_tst = DataLoader(ds_tst, batch_size=bs, shuffle=True)
            acc = eval_dl(model, dl_tst, verbose=False, task_id=test_id, class_inc=True)
            acc_mat[task_ind, test_id] = acc    

        with np.printoptions(precision=2, suppress=True):
            print(f'accumulative acc: {np.array(accumulative_acc_lst)}')  
            print(acc_mat)    


    avg_acc = np.mean(accumulative_acc_lst)

    print(f'avg acc: {avg_acc}')

    save_dict['accumulative_acc_lst'] = accumulative_acc_lst
    save_dict['acc_mat'] = acc_mat  
    save_dict['avg_acc'] = avg_acc
    save_dict['model'] = model.state_dict()
    save_dict['optim'] = optim.state_dict()
    
    save_name = generate_save_name(save_dict)
    pkl.dump(save_dict, open(f'{save_name}.pkl', 'wb'))


train_icarl('icarl_clean', task_num=5, n_epochs=1, seed=0, dataset='split_modulation', model_type='cnn1d', 
            lr=0.01, optim_name='sgd', bs=16, emb_dim=128, exemp_num=2000)

