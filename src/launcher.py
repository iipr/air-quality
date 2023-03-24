import sys, os, time
sys.path.append('../src')
import tables as tb
import numpy as np
import learner


# Dataset conversion
kind = '35x30'
#dset_convs = ['J']
n_x_n_y = [(4, 4)] # [(4 * i, 4) for i in range(1, 7)]
shifts = [1]
# Optimizers: ['Adadelta', 'Nadam', 'SGD', 'RMSprop', 'Adagrad', 'Adamax', 'Adam', 'Ftrl']
#optimizers = ['Adadelta', 'Nadam', 'SGD', 'RMSprop', 'Adagrad', 'Adamax', 'Adam', 'Ftrl']
optimizers = ['Adamax']
# Loss function: ['mse', 'mae', 'msle']
#losses = ['mse', 'mae', 'msle']#['mse']
losses = ['mse']
# Learning rates:
#lrs = [0.001, 0.0001, 0.00001, 0.000005]
lrs = [0.001]
# Neural networks: ['ST-AQF-star', 'ST-AQF-w', 'ST-AQF-t', 'ST-AQF', 'persistence', 'naive', 'fc', 'lstm', 'biLstm']
nns = ['ST-AQF']
# Scaling method: ['raw', 'stand', 'norm']
scalings = ['stand']
# Interpolation method: ['nearest', 'linear']
interps = ['nearest']
# Jacobi's epsilon (0 = no Jacobi):
jacobis = [0.]#,0.1, 1.]
# Number of epochs
epochs = 100
# Experiment (e.g., 'compare-opt-loss-lr', 'single-variable', 'experiment-integrability')
experiment = 'test'
# Indexes of the variables: None for all, or [[idx] for idx in range(11)]
indexes_v = None # [[idx] for idx in range(11)]

def load_learner(idx_v):    
    # Create the learner
    l = learner.DeepLearner()
    if idx_v:
        l.set_idx_v(idx_v)
    if len(l.vn_map.keys()) == 11:
        l.gases = 'all'
    else:
        l.gases = ''
        for g in l.vn_map.keys():
            l.gases += g + '-'
        l.gases = l.gases[:-1]
    return l
   # l.models.set_lr(0)
   # l.set_idx_v([0]) # --> Only SO2   --> ('01', 'SO2'),
   # l.set_idx_v([1]) # --> Only CO    --> ('06', 'CO'),
   # l.set_idx_v([2]) # --> Only NO    --> ('07', 'NO'),
   # l.set_idx_v([3]) # --> Only NO2   --> ('08', 'NO2'),
   # l.set_idx_v([4]) # --> Only PM25  --> ('09', 'PM25'),
   # l.set_idx_v([5]) # --> Only PM10  --> ('10', 'PM10'),
   # l.set_idx_v([6]) # --> Only NOx   --> ('12', 'NOx'),
   # l.set_idx_v([7]) # --> Only O3    --> ('14', 'O3'),
   # l.set_idx_v([8]) # --> Only TOL   --> ('20', 'TOL'),
   # l.set_idx_v([9]) # --> Only BEN   --> ('30', 'BEN'),
   # l.set_idx_v([10]) # --> Only EBE  --> ('35', 'EBE')])
   # l.set_idx_v([0, 1, 3, 5]) #       --> SO2, CO, NO2, PM10
   # l.set_n_y(4)

def train(l, model, epochs):
    if not os.path.exists(os.path.join(l.models_path, model.mod_name)):
        os.mkdir(os.path.join(l.models_path, model.mod_name))
    tic = time.time()
    model = l._DeepLearner__train(model, epochs)
    train_duration_s = time.time() - tic
    fname = os.path.join(l.models_path, model.mod_name, model.mod_name + '.h5')
    # If model already existed, add those previous epochs
    if os.path.exists(fname):
        with tb.open_file(fname, 'r') as h5_mod:
            node = h5_mod.get_node('/')
            epochs += node._v_attrs['epochs']
    model.save(fname)
    # For reproducibility:
    l._DeepLearner__add_meta(fname, model.mod_name, epochs, train_duration_s)
    # Do tests
    l._DeepLearner__plot_model(model)
    l._DeepLearner__plot_loss(model)
    #l._DeepLearner__add_attr(model.mod_name, 'train_duration', 0)
    #l._DeepLearner__test(model)
    l._DeepLearner__test_truth(model, save_preds=False, make_boxplots=True)
    l._DeepLearner__test_date(model, date='15/01/2019', n_days=7)
    l._DeepLearner__test_date(model, date='15/04/2019', n_days=7)
    l._DeepLearner__test_date(model, date='15/07/2019', n_days=7)
    l._DeepLearner__test_date(model, date='15/11/2019', n_days=7)
    #l._DeepLearner__robustness_test(model, n_reps=20)

def load_and_rob(mod_name):
    model = l.models.load_model(os.path.join(l.models_path, mod_name, mod_name + '.h5'))
    model.mod_name = mod_name
    if 'PM25' in model.mod_name: # Special case, need to remove one sensor (sensor 20)
        l.vi_si_map = {0: [1, 6, 10, 13, 14, 16, 19]}
    l._DeepLearner__robustness_test(model, n_reps=20)

for idx_v in indexes_v:
    l = load_learner(idx_v)
    l.set_experiment(experiment)
    if not os.path.exists(l.models_path):
        os.mkdir(os.path.join(l.models_path))
    for jacobi in jacobis:
        l.set_jacobi(jacobi)
        for interp in interps:
            l.set_interp(interp)
            for scaling in scalings:
                l.set_scaling(scaling)
                for loss in losses:
                    l.models.set_loss(loss)
                    for lr in lrs:
                        l.models.set_lr(lr)
                        for optimizer in optimizers:
                            l.models.set_optimizer(optimizer)
                            for n_x, n_y in n_x_n_y:
                                l.set_n_x(n_x)
                                l.set_n_y(n_y)
                                for shift in shifts:
                                    l.set_shift(shift)
                                    for nn in nns:
                                        l.set_calendar_aware('True' if '-t' in nn or 'star' in nn else 'False')
                                        l.set_weather_aware('True' if '-w' in nn or 'star' in nn else 'False')
                                        model = l.models.model(nn)
                                        strs = (nn, scaling, interp, jacobi, n_x, n_y, shift,
                                                loss, lr, optimizer, epochs, l.gases)
                                        model.mod_name = '{}_{}_{}_J{}_nx{:02}_ny{:02}_sh{:02}_{}_lr{}_{}_ep{:03}_{}'.format(*strs)
                                        train(l, model, epochs)
                                        #load_and_rob(model.mod_name)
# Produce table with results when finished
#l.results_table('results_{}.csv'.format(experiment), auto=False)

