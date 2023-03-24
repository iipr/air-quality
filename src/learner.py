import os, time, re, socket
import numpy as np
import pandas as pd
import tables as tb
import datetime as dt
import deep_playground, modelUtils, plotUtils
from trainUtils import HistoryCallback, DataGenerator
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.interpolate import griddata
from pandas.plotting import register_matplotlib_converters
# The following is for the RTX 2070 GPU
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class DeepLearner():
    '''
    Class that encapsulates common tasks when training models.
    '''

    def __init__(self, work_path='..', city='madrid', time_gran='01h',
                 kind='35x30', scaling='raw', interp='nearest', jacobi=0., group='/gases',
                 experiment='test', n_x=4, n_y=4, shift=1, batch_size=7*24, # 1 week as batch_size
                 idx_v=None, calendar_aware=False, weather_aware=False
                ):
        # For files and paths
        self.work_path = work_path
        self.time_gran = time_gran
        self.models_path = os.path.join(self.work_path, 'models', city, experiment)
        self.data_path = os.path.join(self.work_path, 'data', city, 'clean')
        self.other_path = os.path.join(self.work_path, 'data', city, 'other')
        self.dataset = '{}_{}_{}_{}_J{:0.2}.h5'.format(time_gran, kind, scaling, interp, jacobi)
        self.dataset_d = {'time_series': os.path.join(self.data_path, self.dataset)}
        self.dataset_raw = '{}_{}_{}.h5'.format(time_gran, 'flat', 'raw')
        self.dataset_d_raw = {'time_series': os.path.join(self.data_path, self.dataset_raw)}
        self.group = group
        if calendar_aware:
            self.dataset_d['holidays'] = os.path.join(self.other_path, 'holidays.csv')
        if weather_aware:
            self.dataset_d['weather'] = os.path.join(self.data_path, '01h_weather_filledNorm.h5')
        self.city, self.kind, self.scaling, self.interp, self.jacobi = city, kind, scaling, interp, jacobi
        # For training and prediction
        self.n_x = n_x
        self.n_y = n_y
        self.shift = shift
        self.idx_v = idx_v
        self.batch_size = batch_size
        self.calendar_aware, self.weather_aware = calendar_aware, weather_aware
        with tb.open_file(self.dataset_d['time_series'], mode='r') as h5_file:
            self.intial_dt = h5_file.root._v_attrs['initial_dt']
            # Preferably, work with OrderedDict so that keys are sorted in a certain order
            self.vc_map = h5_file.root._v_attrs['vc_map'] # vc_map: variable code --> variable name
            self.vn_map = h5_file.root._v_attrs['vn_map'] # vn_map: variable name --> variable code
            self.sc_map = h5_file.root._v_attrs['sc_map'] # sc_map: station code --> station name
            self.sn_map = h5_file.root._v_attrs['sn_map'] # sn_map: station name --> station code
            self.n_s = len(self.sc_map)
            self.vc_sc_map = h5_file.root._v_attrs['vc_sc_map'] # vc_sc_map: variable code --> list of station codes
            self.vi_si_map = {list(self.vc_map.keys()).index(vc): [list(self.sc_map.keys()).index(sc) for sc in scs]
                              for vc, scs in self.vc_sc_map.items()} # vi_si_map: variable index --> list of station indexes
            #self.sc_vc_map = h5_file.root._v_attrs['sc_vc_map'] # vc_sc_map: station code --> list of variable codes (not used)
            self.n_v = len(self.vc_sc_map)
            self.gases_in_mug = h5_file.root._v_attrs['gases_in_mug'] # Gases measured in \mug
            self.gases_in_mg = h5_file.root._v_attrs['gases_in_mg'] # Gases measured in mg
            self.vc_map_pretty = h5_file.root._v_attrs['vc_map_pretty'] # Pretty names of variables for plotting
        # Mobility: train = [2013, 2017]; val = [2018]; test = [2019, 2020]
        # AirQ:     train = [2010, 2016]; val = [2017]; test = [2018, 2019]
        # Caveat! Leap years: 2012, 2016 and 2020
        self.train_extent = (0, self.__datetime_to_idx(dt.datetime(2017, 1, 1)))
        self.val_extent = (self.train_extent[1], self.__datetime_to_idx(dt.datetime(2018, 1, 1)))
        self.test_extent = (self.val_extent[1], self.__datetime_to_idx(dt.datetime(2020, 1, 1)))
        # Or, begining of COVID pandemic:
        #self.test_extent = (self.val_extent[1], self_datetime_to_idx(dt.datetime(2020, 3, 16)))
        # For quick tests, shorter extents:
        #self.train_extent, self.val_extent, self.test_extent = (0, 1250), (1250, 1650), (1650, 2000)
        # Additional stuff
        self.menu = deep_playground.Menu()
        self.helper = deep_playground.Helper()
        self.models = modelUtils.Models(n_x=self.n_x, n_y=self.n_y, n_v=self.n_v,
                                        act_out='sigmoid' if 'norm' in scaling else 'linear')
        self.set_kind(self.kind)
        self.update_scaling()
        self.set_idx_v(self.idx_v)
        self.plotter = plotUtils.Plotter()
        self.mod_queue = []
        register_matplotlib_converters() # <- to avoid a warning

    def __idx_to_datetime(self, idx, base=None, freq=None):
        if base is None:
            base = self.intial_dt
        # Frequency expressed in hours
        if freq is None:
            freq = int(self.time_gran.replace('h', ''))
        return base + dt.timedelta(hours=freq * idx)

    def __datetime_to_idx(self, date, base=None, freq=None):
        if base is None:
            base = self.intial_dt
        # Frequency expressed in hours
        if freq is None:
            freq = int(self.time_gran.replace('h', ''))
        return int((date - base).total_seconds() / (3600 * freq))

    #################################################
    #                  TRAINING                     #
    #################################################

    def prepare_training(self):
        title = 'Training phase:'
        opts = {'Create new model.': self.__create_model,
                'Load existing model.': self.__load_model,
                'Check models that will be trained.': self.__models_to_train,
                'Ready for training.': self.__train_models,
                'Back to main menu.': self.menu.exit
               }
        stop = False
        while not stop:
            opt = self.menu.run(list(opts.keys()), title=title)
            if opt:
                stop = opts[opt]()

    def __create_model(self):
        title = 'Available models are:'
        mod_list = self.models.get_models_list()
        while True:
            opt = self.menu.run(mod_list, title=title)
            if opt:
                break
        model = self.models.model(opt)
        model.mod_name = self.helper.read('Please enter a new model name')
        self.__prep_model(model)

    def __load_model(self, prep_train=True):
        title = 'Available models are:'
        mod_list = sorted([m for m in os.listdir(self.models_path) 
                           if m[0] != '.' and os.path.isdir(os.path.join(self.models_path, m))])
        while True:
            opt = self.menu.run(mod_list, title=title)
            if opt:
                mod_path = os.path.join(self.models_path, opt, opt + '.h5')
                if os.path.exists(mod_path):
                    break
                else:
                    self.helper._print('Wrong file name! -> {}'.format(mod_path))
        do_compile = self.helper.read('Compile model? [y/n]')
        model = self.models.load_model(mod_path, True if do_compile == 'y' else False)
        model.mod_name = opt
        if self.helper.read('See model summary? [y/n]') == 'y':
            model.summary()
            self.helper._continue()
        if prep_train: # for training
            self.__prep_model(model)
        else:          # for testing
            return model

    def __prep_model(self, model):
        epochs = self.helper.read('Number of epochs', cast=int)
        self.mod_queue.append((model, epochs))
        self.helper._print('Model added to queue.')
        self.helper._continue()

    def __models_to_train(self):
        for idx, (model, epochs) in enumerate(self.mod_queue):
            self.helper._print('\t{}. {}: {} epochs'.format(idx+1, model.mod_name, epochs))
        self.helper._continue()

    def __train_models(self, do_test=True):
        idx = 0
        while len(self.mod_queue) > 0:
            idx +=1
            model, epochs = self.mod_queue.pop(0)
            self.helper._print('{}. Starting to train model {}...'.format(idx, model.mod_name))
            if not os.path.exists(os.path.join(self.models_path, model.mod_name)):
                os.mkdir(os.path.join(self.models_path, model.mod_name))
            tic = time.time()
            model = self.__train(model, epochs)
            train_duration_s = time.time() - tic
            fname = os.path.join(self.models_path, model.mod_name, model.mod_name + '.h5')
            # If model already existed, add those previous epochs
            if os.path.exists(fname):
                with tb.open_file(fname, 'r') as h5_mod:
                    node = h5_mod.get_node('/')
                    epochs += node._v_attrs['epochs']
            model.save(fname)
            # For reproducibility:
            self.__add_meta(fname, model.mod_name, epochs, train_duration_s)
            # After training, do tests
            if do_test:
                self.__plot_model(model)
                self.__plot_loss(model)
                self.__test(model)

    def __train(self, model, epochs):
        # Define parameters for the generator
        params = {'n_x': self.n_x,
                  'n_y': self.n_y,
                  'shift': self.shift,
                  'dataset_d': self.dataset_d,
                  'batch_size': self.batch_size,
                  'group': self.group,
                  'idx_v': self.idx_v
                 }
        train_generator = DataGenerator(extent=self.train_extent, **params)
        val_generator = DataGenerator(extent=self.val_extent, **params)
        log_file = os.path.join(self.models_path, model.mod_name, model.mod_name + '_logs.txt') 
        print(model.summary())
        model.fit(x=train_generator,
                  validation_data=val_generator,
                  epochs=epochs,
                  callbacks=[HistoryCallback(mod_name=model.mod_name,
                                             log_file=log_file,
                                             ),
                             EarlyStopping(monitor='val_loss',
                                           patience=10)],
                  use_multiprocessing=True, workers=0)
        return model

    def __add_meta(self, fname, mod_name, epochs, train_duration_s=0):
        # Add metadata to the model
        with tb.open_file(fname, 'a') as h5_mod:
            node = h5_mod.get_node('/')
            node._v_attrs['name'] = mod_name
            node._v_attrs['epochs'] = epochs
            node._v_attrs['train_duration_s'] = train_duration_s
            node._v_attrs['trained_by'] = os.environ.get('LOGNAME')
            node._v_attrs['trained_in'] = socket.gethostname()
            node._v_attrs['trained_with'] = [tf.config.experimental.get_device_details(gpu)['device_name'] for gpu in gpus] \
                                            if len(gpus) > 0 else ['CPU']
            # Data
            node._v_attrs['time_gran'] = self.time_gran
            node._v_attrs['dataset'] = self.dataset
            node._v_attrs['city'] = self.city
            node._v_attrs['kind'] = self.kind
            node._v_attrs['interp'] = self.interp
            node._v_attrs['scaling'] = self.scaling
            #node._v_attrs['zone_ids'] = self.zone_ids
            # Model
            node._v_attrs['batch_size'] = self.batch_size
            node._v_attrs['n_x'] = self.n_x
            node._v_attrs['n_y'] = self.n_y
            node._v_attrs['n_v'] = self.n_v
            node._v_attrs['shift'] = self.shift
            node._v_attrs['idx_v'] = self.idx_v
            node._v_attrs['calendar_aware'] = self.calendar_aware
            node._v_attrs['weather_aware'] = self.weather_aware
            node._v_attrs['train_extent'] = self.train_extent
            node._v_attrs['val_extent'] = self.val_extent
            node._v_attrs['test_extent'] = self.test_extent
            # Optimizer
            node._v_attrs['optimizer'] = self.models.get_optimizer()
            node._v_attrs['loss'] = self.models.get_loss()
            node._v_attrs['metrics'] = self.models.get_metrics()
            node._v_attrs['lr'] = self.models.get_lr()

    def __add_attr(self, mod_name, attr, value):
        # Add single attribute
        fname = os.path.join(self.models_path, mod_name, mod_name + '.h5')
        with tb.open_file(fname, 'a') as h5_mod:
            node = h5_mod.get_node('/')
            node._v_attrs[attr] = value

    def __get_attr(self, mod_name, attr):
        # Get a single attribute
        fname = os.path.join(self.models_path, mod_name, mod_name + '.h5')
        with tb.open_file(fname, 'r') as h5_mod:
            node = h5_mod.get_node('/')
            return node._v_attrs[attr]


    #################################################
    #                  TESTING                      #
    #################################################

    def prepare_test(self):
        model = self.__load_model(prep_train=False)
        title = 'Testing phase for {}:'.format(model.mod_name)
        opts = {'Plot model graph.': self.__plot_model,
                'Check model hyperparameters.': self.__model_hyperparams,
                'Plot model loss and metrics.': self.__plot_loss,
                'Test on (unseen) years.': self.__test,
                'Test on (unseen) years using true (raw) data.': self.__test_truth,
                'Test and plot a certain day.': self.__test_date,
                'Robustness test.': self.__robustness_test,
                'Back to main menu.': self.menu.exit,
               }
        stop = False
        while not stop:
            opt = self.menu.run(list(opts.keys()), title=title)
            if opt:
                stop = opts[opt](model)
                self.helper._continue()

    def __plot_model(self, model):
        aux_path = os.path.join(self.models_path, model.mod_name, model.mod_name + '.pdf')
        self.models.plot_model(model, aux_path)

    def __model_hyperparams(self, model):
        fname = os.path.join(self.models_path, model.mod_name, model.mod_name + '.h5')
        with tb.open_file(fname, 'a') as h5_mod:
            node = h5_mod.get_node('/')
            for idx, attr in enumerate(node._v_attrs._v_attrnames):
                if '_config' not in attr:
                    self.helper._print('\t{:02}. {}: {}'.format(idx+1, attr, node._v_attrs[attr]))

    def __plot_loss(self, model):
        # Load history file
        aux_path = os.path.join(self.models_path, model.mod_name, model.mod_name)
        hist = pd.read_csv(aux_path + '_hist.csv')
        days = hist['duration [s]'].sum() // (24 * 3600)
        train_duration = '{} days and {}'.format(days, time.strftime('%H:%M:%S', 
                                                                     time.gmtime(hist['duration [s]'].sum())))
        self.helper._print('Total training time for model {} was {}'.format(model.mod_name, train_duration))
        self.__add_attr(model.mod_name, 'train_duration', train_duration)
        loss_names = {'mse': 'Mean Squared Error',
                      'val_mse': 'Mean Squared Error',
                      'mae': 'Mean Absolute Error',
                      'val_mae': 'Mean Absolute Error',
                      'acc': 'Accuracy',
                      'val_acc': 'Accuracy',
                      'categorical_crossentropy': 'Categorical Crossentropy',
                      'msle': 'Mean Squared Logarithmic Error',
                      'val_msle': 'Mean Squared Logarithmic Error'
                     }
        series_d = dict()
        # Here we reverse the columns for a better order in the plot and legend
        print(hist.columns[2:])
        for hkey in reversed(hist.columns[3:]): # after epoch & duration
            if 'loss' in hkey: # Skip loss due to the keras bug
                continue
            metric_name = loss_names[hkey]
            if self.models.get_loss() in hkey:
                metric_name += ' - Loss'
            t_set = 'Validation set' if 'val_' in hkey else 'Training set'
            if metric_name in series_d.keys():
                series_d[metric_name][t_set] = hist.loc[:, hkey].values
            else:
                series_d[metric_name] = {t_set: hist.loc[:, hkey].values}
        self.plotter.series(series_d, ('epoch', ''), int_ticker=True,
                                 out_path=aux_path + '_loss.pdf')#yscale='log'

    def __test(self, model):
        # Define parameters for the generator
        params = {'n_x': self.n_x,
                  'n_y': self.n_y,
                  'shift': self.shift,
                  'dataset_d': self.dataset_d,
                  'batch_size': self.batch_size,
                  'idx_v': self.idx_v
                 }
        test_generator = DataGenerator(extent=self.test_extent, **params)
        ret_d = model.evaluate(test_generator, verbose=1, return_dict=True,
                             use_multiprocessing=True, workers=0)
        for metric, value in ret_d.items():
            self.__add_attr(model.mod_name, 'test_{}'.format(metric), round(value, 5))

    def __scale(self, y, id_v):
        # id_v indicates the index of the variable that is passed, to take the corresponding min/max/avg/std
        if 'raw' in self.scaling:
            pass
        elif 'stand' in self.scaling:
            vc = list(self.vc_map.keys())[id_v]
            y = (y - self.vc_mean_map[vc]) / self.vc_std_map[vc]
        elif 'norm' in self.scaling:
            vc = list(self.vc_map.keys())[id_v]
            y = (y - self.vc_min_map[vc]) / (self.vc_max_map[vc] - self.vc_min_map[vc])
        return y

    def __de_scale(self, y, id_v):
        # id_v indicates the index of the variable that is passed, to take the corresponding min/max/avg/std
        if 'raw' in self.scaling:
            pass
        elif 'stand' in self.scaling:
            vc = list(self.vc_map.keys())[id_v]
            y = y * self.vc_std_map[vc] + self.vc_mean_map[vc]
        elif 'norm' in self.scaling:
            vc = list(self.vc_map.keys())[id_v]
            y = y * (self.vc_max_map[vc] - self.vc_min_map[vc]) + self.vc_min_map[vc]
        return y

    def __test_truth(self, model, save_preds=True, make_boxplots=True):
        # Define parameters for the generator
        params = {'n_x': self.n_x,
                  'n_y': self.n_y,
                  'shift': self.shift,
                  'dataset_d': self.dataset_d,
                  'batch_size': self.batch_size,
                  'go': True
                 }
        params_truth = params.copy()
        params_truth['dataset_d'] = self.dataset_d_raw
        pred_generator = DataGenerator(extent=self.test_extent, group=self.group, idx_v=self.idx_v, **params)
        preds = np.empty((pred_generator.n_preds, self.n_y, self.n_s, self.n_v))
        truth = np.empty((pred_generator.n_preds, self.n_y, self.n_s, self.n_v))
        preds[:], truth[:] = np.nan, np.nan
        # As many truth_generators as variables to evaluate
        truth_generators = [DataGenerator(extent=self.test_extent, group='/v{}'.format(v), **params_truth)
                            for v in self.vc_sc_map.keys()]
        # Number of stations per variable
        n_s_per_v = {vi: len(si) for vi, si in self.vi_si_map.items()} # variable index --> number of stations
        for n_batch, batch in enumerate(zip(pred_generator, *truth_generators)): # ((X_b, y_b), (X_v1, y_v1), ..., (X_vn, y_vn))
            # If the model works with grids, dimensions of X_b --> (batch_size, n_y, n_r, n_c, n_v), indexing='ij'
            X_b, y_t = batch[0][0], [b[1] for b in batch[1:]]
            y_pred = model.predict_on_batch(X_b)
            for id_v, y_v in enumerate(y_t): # (de)interpolate the result for each target variable
                # First, place map axes first and flatten along the remaining dimensions
                y_pred_v = y_pred[..., id_v].transpose((2, 3, 0, 1)).reshape((self.grid_shape[1] * self.grid_shape[0], -1))
                y_pred_v = griddata(self.grid, y_pred_v, self.ij[self.vi_si_map[id_v]], method=self.interp,
                                    fill_value=np.nan).reshape((n_s_per_v[id_v], -1, self.n_y)).transpose((1, 2, 0))
                y_pred_v = self.__de_scale(y_pred_v, id_v) # De-scale predictions
                slc = slice(n_batch * self.batch_size, n_batch * self.batch_size + y_pred_v.shape[0])
                preds[slc, :, self.vi_si_map[id_v], id_v] = y_pred_v
                # The raw data includes a time_since_epoch column, that's why we use y_v[..., 1:] to exclude it
                truth[slc, :, self.vi_si_map[id_v], id_v] = y_v[..., 1:]
        # Compute and save the MAE and RMSE of the model
        mae_mod = np.nanmean(np.abs(preds - truth), axis=0)
        rmse_mod = np.sqrt(np.nanmean((preds - truth)**2, axis=0))
        self.__add_attr(model.mod_name, 'mae_mod', mae_mod)
        self.__add_attr(model.mod_name, 'rmse_mod', rmse_mod)
        if save_preds:
            fname = os.path.join(self.models_path, model.mod_name, 'pred_truth.h5')
            with tb.open_file(fname, mode='w') as h5:
                h5.create_carray(where='/', name='truth', atom=tb.Float32Atom(), obj=np.float32(truth))
                h5.create_carray(where='/', name='preds', atom=tb.Float32Atom(), obj=np.float32(preds))
            self.helper._print('Predictions and truth values saved to {}'.format(fname))
        if make_boxplots:
            for id_v, vc in enumerate(self.vc_sc_map.keys()):
                # If there are NaNs, filter them out
                mae_vc, rmse_vc = mae_mod[:, self.vi_si_map[id_v], id_v], rmse_mod[:, self.vi_si_map[id_v], id_v]
                mmask_vc, rmask_vc = ~np.isnan(mae_vc), ~np.isnan(rmse_vc)
                units = '$\mu$g/m$^3$' if vc in self.gases_in_mug else 'mg/m$^3$'
                var, var_pretty = self.vc_map[vc], self.vc_map_pretty[vc]
                self.plotter.boxplot([d[m] for d, m in zip(mae_vc, mmask_vc)], x_labels=range(1, self.n_y + 1),
                                     labels=('Forecast horizon [{}]'.format(self.time_gran[1:]), 'MAE for {} [{}]'.format(var_pretty, units)),
                                     title='MAE variation among sensors for each horizon', #yscale='log',
                                     out_path=os.path.join(self.models_path, model.mod_name, '{}_mae_{}.pdf'.format(model.mod_name, var)))
                self.plotter.boxplot([d[m] for d, m in zip(rmse_vc, rmask_vc)], x_labels=range(1, self.n_y + 1),
                                     labels=('Forecast horizon [{}]'.format(self.time_gran[1:]), 'RMSE for {} [{}]'.format(var_pretty, units)),
                                     title='RMSE variation among sensors for each horizon', #yscale='log',
                                     out_path=os.path.join(self.models_path, model.mod_name, '{}_rmse_{}.pdf'.format(model.mod_name, var)))

    def __predict(self, model, day, month, year, n_days=1):
        # Define parameters for the generator
        params = {'n_x': self.n_x,
                  'n_y': self.n_y,
                  'shift': self.shift,
                  'dataset_d': self.dataset_d,
                  'batch_size': self.batch_size,
                  'go': True
                 }
        # Parameters for the truth generators
        params_truth = params.copy()
        params_truth['dataset_d'] = self.dataset_d_raw
        # Extent that the data generators need
        start_id = self.__datetime_to_idx(dt.datetime(year, month, day, 0, 0)) - self.n_x - self.shift + 1
        end_id = self.__datetime_to_idx(dt.datetime(year, month, day, 23, 0)) + 2 + (n_days - 1) * 24
        pred_generator = DataGenerator(extent=(start_id, end_id), group=self.group, idx_v=self.idx_v, **params)
        preds = np.empty((pred_generator.n_preds, self.n_y, self.n_s, self.n_v))
        truth = np.empty((pred_generator.n_preds, self.n_y, self.n_s, self.n_v))
        preds[:], truth[:] = np.nan, np.nan
        # As many truth_generators as variables to evaluate
        truth_generators = [DataGenerator(extent=(start_id, end_id), group='/v{}'.format(v), **params_truth)
                            for v in self.vc_sc_map.keys()]
        # Generate hours based on the horizon
        hours = np.empty((pred_generator.n_preds, self.n_y), dtype=object)
        y_l = self.__datetime_to_idx(dt.datetime(year, month, day, 0, 0))
        slc = slice(y_l, y_l + pred_generator.n_preds)
        for h in range(self.n_y):
            hours[:, h] = np.array([self.__idx_to_datetime(id_t + h) for id_t in range(slc.start, slc.stop)])
        # Number of stations per variable
        n_s_per_v = {vi: len(si) for vi, si in self.vi_si_map.items()} # variable index --> number of stations
        for n_batch, batch in enumerate(zip(pred_generator, *truth_generators)): # ((X_b, y_b), (X_v1, y_v1), ..., (X_vn, y_vn))
            # If the model works with grids, dimensions of X_b --> (batch_size, n_y, n_r, n_c, n_v), indexing='ij'
            X_b, y_t = batch[0][0], [b[1] for b in batch[1:]]
            y_pred = model.predict_on_batch(X_b)
            for id_v, y_v in enumerate(y_t): # (de)interpolate the result for each target variable
                # First, place map axes first and flatten along the remaining dimensions
                y_pred_v = y_pred[..., id_v].transpose((2, 3, 0, 1)).reshape((self.grid_shape[1] * self.grid_shape[0], -1))
                y_pred_v = griddata(self.grid, y_pred_v, self.ij[self.vi_si_map[id_v]], method=self.interp,
                                    fill_value=np.nan).reshape((n_s_per_v[id_v], -1, self.n_y)).transpose((1, 2, 0))
                y_pred_v = self.__de_scale(y_pred_v, id_v) # De-scale predictions
                slc = slice(n_batch * self.batch_size, n_batch * self.batch_size + y_pred_v.shape[0])
                preds[slc, :, self.vi_si_map[id_v], id_v] = y_pred_v
                truth[slc, :, self.vi_si_map[id_v], id_v] = y_v[..., 1:]
        return hours, preds, truth

    def __test_date(self, model, date=None, n_days=7):
        if not date:
            date = self.helper.read('Date (dd/mm/yyyy)')
        d, m, y = date.split('/')
        hours, preds, truth = self.__predict(model, *[int(n) for n in (d, m, y)], n_days)
        # Create folder for plots
        plt_path = os.path.join(self.models_path, model.mod_name, 'day_plots')
        if not os.path.exists(plt_path):
            os.mkdir(plt_path)
        xlim = (dt.datetime(*[int(n) for n in (y, m, d)], hour=0, minute=0),
                dt.datetime(*[int(n) for n in (y, m, d)], hour=23, minute=59) + dt.timedelta(days=n_days - 1))
        for vi in range(self.n_v):
            ylim = (min(np.nanmin(preds[..., vi]), np.nanmin(truth[..., vi])) - 1,
                    max(np.nanmax(preds[..., vi]), np.nanmax(truth[..., vi])) + 1)
            vc = list(self.vc_map.keys())[vi]
            units = '$\mu$g/m$^3$' if vc in self.gases_in_mug else 'mg/m$^3$'
            var, var_pretty = self.vc_map[vc], self.vc_map_pretty[vc]
            mae = np.nanmean(np.abs(preds - truth), axis=0)
            rmse = np.sqrt(np.nanmean((preds - truth)**2, axis=0))
            series = dict()
            for horizon in range(self.n_y):
                for idx, si in enumerate(self.vi_si_map[vi]):
                    title = 'True vs predicted {} for sensor {} from {}, \nHorizon = {}h, RMSE = {:0.2}, MAE = {:0.2}'.format(
                        var_pretty, self.vc_sc_map[vc][idx], date, (self.shift + horizon)*1,
                        rmse[horizon, si, vi], mae[horizon, si, vi])
                    series[title] = {'pred': preds[:, horizon, si, vi],
                                     'true': truth[:, horizon, si, vi]}
            self.plotter.series(series, ('Hour', '{} [{}]'.format(var_pretty, units)), date_ticker=hours, scale=0.5,
                                     dims=(self.n_y, len(self.vi_si_map[vi])), xlim=xlim, ylim=ylim,
                                     out_path=plt_path + '/{}_series_{}.png'.format(y+m+d, var))

    def __robustness_test(self, model, up_to=None, n_reps=10):
        # Define parameters for the generator
        params = {'n_x': self.n_x,
                  'n_y': self.n_y,
                  'shift': self.shift,
                  'dataset_d': self.dataset_d_raw,
                  'batch_size': self.batch_size,
                  'go': True
                 }
        # Parameters for the truth generators
        vc = list(self.vc_map.keys())[0]
        reps = ['rep_{:02}'.format(idx) for idx in range(n_reps)]
        s_idx = self.vi_si_map[0]
        max_s = len(s_idx)
        excl = np.arange(start=1, stop=int(np.ceil(max_s / 2)) + 1 if up_to is None else up_to+1)
        rob_path = os.path.join(self.models_path, model.mod_name, 'robustness-test')
        if not os.path.exists(rob_path):
            os.mkdir(rob_path)
        rmse_base = self.__get_attr(model.mod_name, 'rmse_mod')[:, s_idx, 0]
        # To keep the results
        arr_err = np.empty(shape=(len(excl), n_reps, self.n_y, max_s))
        arr_sks = np.empty(shape=(len(excl), n_reps, self.n_y, max_s)) # --> (5, 1, 4, 10)
        print('Working on {}...'.format(self.vc_map[vc]))
        for id_excl, n_excl in enumerate(excl):
            print('\tfor {:02} sensor(s) excluded...'.format(n_excl))
            for id_rep, n_rep in enumerate(reps):
                print('\t\ton {}...'.format(n_rep), end='\r')
                chosen = np.sort(np.random.choice(range(max_s), size=max_s - n_excl, replace=False))
                sse_mod = np.zeros((self.n_y, max_s))
                n_samp = np.zeros((self.n_y, max_s)) # <-- Occhio! The sample count depends on the sensor
                generator = DataGenerator(extent=self.test_extent, group='/v{}'.format(vc), idx_v=None, **params)
                for n_batch, (X_b, y_t) in enumerate(generator):
                    X_b['time_series'] = X_b['time_series'][..., 1:max_s+1]
                    y_t = y_t[..., 1:max_s+1]
                    bs = y_t.shape[0]
                    # For [batch, n_x, sensor]
                    aux = X_b['time_series'][:, :, chosen]
                    aux = self.__scale(aux, 0) # Scale inputs
                    grids = np.empty((bs, self.n_x, *self.grid_shape), dtype=np.float32)
                    for idx in range(bs):
                        for step in range(self.n_x):
                            mask = ~np.isnan(aux[idx, step])
                            if mask.sum() == 0:
                                # Note that [idx - 1, step] is the previous of the current one for all step
                                aux[idx, step] = aux[idx - 1, step]
                                grids[idx, step] = grids[idx - 1, step]
                            else:
                                curr = aux[idx, step].copy()
                                curr[~mask] = aux[idx - 1, step, ~mask]
                                mask = ~np.isnan(curr)
                                grids[idx, step] = griddata(self.ij[chosen[mask]], curr[mask], (self.I, self.J),
                                             method='nearest')#.transpose((2, 0, 1))
                    X_b['time_series'] = np.expand_dims(grids, axis=-1)
                    # Predict and de-interpolate
                    y_pred = model.predict_on_batch(X_b)
                    # Dimensions of y_pred: (batch, n_x, lat, lng, n_v)
                    # First, place map axes first and flatten along the remaining dimensions
                    y_pred = y_pred[:, :, :, :, 0].transpose((2, 3, 0, 1)).reshape((self.grid_shape[0] * self.grid_shape[1], -1))
                    # Then interpolate, undo the flatten and the transpose
                    y_pred = griddata(self.grid, y_pred, self.ij[s_idx], method='nearest',
                                           fill_value=0).reshape((max_s, bs, self.n_y)).transpose((1, 2, 0))
                    y_pred = self.__de_scale(y_pred, 0) # De-scale predictions
                    # Evaluate error
                    sse_mod += np.nansum(((y_t - y_pred)**2), axis=0)
                    n_samp += bs - np.isnan(y_t).sum(axis=0)
                # Compute and save the RMSE for the model
                rmse_mod = np.sqrt(sse_mod / n_samp)
                arr_err[id_excl, id_rep] = rmse_mod
                arr_sks[id_excl, id_rep] = (1 - rmse_mod / rmse_base) * 100
        np.save(os.path.join(rob_path, 'rmse_per_horizon_{}reps.npy'.format(n_reps)), arr_err)
        np.save(os.path.join(rob_path, 'skill_per_horizon_{}reps.npy'.format(n_reps)), arr_sks)
        self.__plot_robustness(arr_sks, name=self.vc_map_pretty[vc], tickers=[i for i in excl],
                               plt_path=os.path.join(rob_path, model.mod_name + '_rob_test_{}reps.pdf'.format(n_reps)))

    def __plot_robustness(self, arr, name, tickers, plt_path):
        series_d = dict(zip(['{}h'.format(1 * (self.shift + h)) for h in range(self.n_y)],
                            (*np.median(arr.mean(axis=1), axis=-1).T,)))
        print(series_d)
        self.plotter.series({'Skill with respect to the original model - {}'.format(name): series_d},
                            ('Number of excluded sensors', 'Worsening [%]'), tickers=tickers, style='o-',
                            out_path=plt_path)


    #################################################
    #                 UPDATE OPTIONS                #
    #################################################

    def update_options(self):
        title = 'The options that can be changed are:'
        stop = False
        while not stop:
            opts = {'Time granularity: {}'.format(self.get_time_gran()): self.set_time_gran,
                    'City: {}'.format(self.get_city()): self.set_city,
                    'Experiment: {}'.format(self.get_experiment()): self.set_experiment,
                    'Scaling method: {}'.format(self.get_scaling()): self.set_scaling,
                    'Interpolation method: {}'.format(self.get_interp()): self.set_interp,
                    #'Indexes of the zones: {} in total'.format(self.grid_shape): self.set_zone_ids,
                    'Batch size: {}'.format(self.get_batch_size()): self.set_batch_size,
                    'n_x: {}'.format(self.get_n_x()): self.set_n_x,
                    'n_y: {}'.format(self.get_n_y()): self.set_n_y,
                    'shift: {}'.format(self.get_shift()): self.set_shift,
                    'Calendar aware: {}'.format(self.get_calendar_aware()): self.set_calendar_aware,
                    'Weather aware: {}'.format(self.get_weather_aware()): self.set_weather_aware,
                    'Optimizer: {}'.format(self.models.get_optimizer()): self.models.set_optimizer,
                    'Loss: {}'.format(self.models.get_loss()): self.models.set_loss,
                    'Metrics: {}'.format(self.models.get_metrics()): self.models.set_metrics,
                    'Learning rate: {}'.format(self.models.get_lr()): self.models.set_lr,
                    'Back to main menu.': self.menu.exit,
                }
            opt = self.menu.run(list(opts.keys()), title=title)
            value = self.helper.read('Value')
            if opt:
                stop = opts[opt](value)


    #################################################
    #                 RESULTS TABLE                 #
    #################################################

    def results_table(self, fname='results.csv', auto=True):
        df = pd.DataFrame(columns=['Model', 'Target',
                                   'Time granularity',
                                   'n_x', 'n_y', 'shift',
                                   'MSE test', 'MAE test',
                                   'MMWC taxi', 'MMMWC taxi',
                                   'MMWC bike', 'MMMWC bike',
                                   'MRWC taxi', 'MMRWC taxi',
                                   'MRWC bike', 'MMRWC bike',
                                   'Epochs', 'Loss', 'Optimizer',
                                   'Learning rate', 'Train duration', 'Train duration [s]'])
        for idx, model in enumerate(os.listdir(self.models_path)):
            model_path = os.path.join(self.models_path, model, model + '.h5')
            if not os.path.exists(model_path):
                continue
            with tb.open_file(model_path, 'r') as h5_mod:
                node = h5_mod.root
                if 'test_mse' not in node._v_attrs:
                    continue
                name, zone = 'unknown', ''
                df.loc[idx] = [node._v_attrs['name'],
                               self.target_d[node._v_attrs['kind'] + '_' + node._v_attrs['scaling']] + zone,
                               node._v_attrs['time_gran'],
                               node._v_attrs['n_x'], node._v_attrs['n_y'], node._v_attrs['shift'],
                               node._v_attrs['test_mse'], node._v_attrs['test_mae'],
                               node._v_attrs['mae_w_count_taxi'].mean(axis=1).round(3),
                               node._v_attrs['mae_w_count_taxi'].mean(axis=1).mean(),
                               node._v_attrs['mae_w_count_bike'].mean(axis=1).round(3),
                               node._v_attrs['mae_w_count_bike'].mean(axis=1).mean(),
                               node._v_attrs['rmse_w_count_taxi'].mean(axis=1).round(3),
                               node._v_attrs['rmse_w_count_taxi'].mean(axis=1).mean(),
                               node._v_attrs['rmse_w_count_bike'].mean(axis=1).round(3),
                               node._v_attrs['rmse_w_count_bike'].mean(axis=1).mean(),
                               node._v_attrs['epochs'], node._v_attrs['loss'],
                               node._v_attrs['optimizer'].__name__, node._v_attrs['lr'],
                               node._v_attrs['train_duration'],
                               node._v_attrs['train_duration_s']
                              ]
        df.sort_values(['Time granularity', 'Target', 'n_x', 'n_y', 'shift', 'Model', 'Optimizer', 'Loss'], inplace=True)
        df.to_csv(os.path.join(self.models_path, fname), index=False)
        self.helper._print('{} saved at {}'.format(fname, self.models_path))
        if auto:
            self.helper._continue()


    #################################################
    #              GETTERS & SETTERS                #
    #################################################

    # n_x
    def get_n_x(self):
        return self.n_x
    def set_n_x(self, x):
        self.n_x = int(x)
        self.models.set_n_x(self.n_x)

    # n_y
    def get_n_y(self):
        return self.n_y
    def set_n_y(self, x):
        self.n_y = int(x)
        self.models.set_n_y(self.n_y)

    # shift
    def get_shift(self):
        return self.shift
    def set_shift(self, x):
        self.shift = int(x)

    # idx_v: indexes of the chosen variables
    def get_idx_v(self):
        return self.idx_v
    def set_idx_v(self, x):
        self.idx_v = x
        if self.idx_v is not None:
            vc_del = set(self.vc_map.keys()).difference(set([list(self.vc_map.keys())[idx] for idx in self.idx_v]))
            #vi_del = list(self.vc_map.keys()).index(vc)
            for vc in vc_del:
                vn_aux = self.vc_map[vc]
                self.vc_map.pop(vc)
                self.vn_map.pop(vn_aux)
                self.vc_sc_map.pop(vc)
                #self.gases_in_mug.pop(vc)
                #self.gases_in_mg.pop(vc)
                self.vc_map_pretty.pop(vc)
                if self.scaling == 'std':
                    self.vc_mean_map.pop(vc)
                    self.vc_std_map.pop(vc)
                elif self.scaling == 'norm':
                    self.vc_min_map.pop(vc)
                    self.vc_max_map.pop(vc)
            self.vi_si_map = {list(self.vc_map.keys()).index(vc): [list(self.sc_map.keys()).index(sc) for sc in scs]
                              for vc, scs in self.vc_sc_map.items()} # vc_si_map: variable code --> list of station indexes
            self.n_v = len(self.vc_sc_map)
            self.models.set_n_v(self.n_v)

    # Map shape for convolutionals
    def get_kind(self):
        return self.kind
    def set_kind(self, x):
        if 'x' in x: # e.g. '35x30'
            # Load longitude and latitude for the interpolation (we assume they are sorted according to their IDs)
            with tb.open_file(self.dataset_d['time_series'], mode='r') as h5_file:
                self.grid_shape = h5_file.root._v_attrs['grid_shape']
                self.ij = h5_file.root._v_attrs['ij']    
                self.I, self.J = h5_file.root._v_attrs['IJ']
            self.n_sensors = self.ij.shape[0]
            self.lat = self.ij[:, 0]
            self.lng = self.ij[:, 1]
            self.grid = np.stack((self.I.ravel(), self.J.ravel())).T
        else:
            self.grid_shape = (self.n_s,)
        self.models.set_grid_shape(self.grid_shape)

    # City
    def get_city(self):
        return self.city
    def set_city(self, x):
        curr_city = self.city
        self.city = x
        self.other_path.replace(curr_city, self.city)
        self.models_path = self.models_path.replace(curr_city, self.city)
        self.data_path.replace(curr_city, self.city)
        self.update_datasets(curr_city, self.city, update_raw=True)

    # Time granularity
    def get_time_gran(self):
        return self.time_gran
    def set_time_gran(self, x):
        curr_time_gran = self.time_gran
        self.time_gran = x
        self.models_path = self.models_path.replace(curr_time_gran, self.time_gran)
        self.update_datasets(curr_time_gran, self.time_gran, update_raw=True)

    # Scaling method (raw, normalization, standarization, etc.)
    def get_scaling(self):
        return self.scaling
    def set_scaling(self, x):
        curr_scaling = self.scaling
        self.scaling = x
        self.models.set_act_out('sigmoid' if 'norm' in self.scaling else 'linear')
        self.update_datasets(curr_scaling, self.scaling)
        self.update_scaling()
        self.update_datasets(curr_scaling, self.scaling)

    # Interpolation method
    def get_interp(self):
        return self.interp
    def set_interp(self, x):
        curr_interp = self.interp
        self.interp = x
        self.update_datasets(curr_interp, self.interp)

    # Jacobi epsilon
    def get_jacobi(self):
        return self.jacobi
    def set_jacobi(self, x):
        curr_jacobi = self.jacobi
        self.jacobi = x
        self.update_datasets(curr_jacobi, self.jacobi)

    # Group path
    def set_group(self,group):
        self.group = group
    def get_group(self):
        return self.group

    # Experiment (models folder)
    def get_experiment(self):
        return os.path.basename(self.models_path)
    def set_experiment(self, x):
        self.models_path = self.models_path.replace(os.path.basename(self.models_path), x)

    # Batch size
    def get_batch_size(self):
        return self.batch_size
    def set_batch_size(self, x):
        self.batch_size = int(x)

    # Calendar aware (input with time, day of the week...)
    def get_calendar_aware(self):
        return self.calendar_aware
    def set_calendar_aware(self, x):
        calendar = eval(x)
        if calendar:
            self.dataset_d['holidays'] = os.path.join(self.other_path, 'holidays.csv')
        elif not calendar and 'calendar' in self.dataset_d.keys():
            _ = self.dataset_d.pop('holidays')
        self.calendar_aware = calendar

    # Weather aware (input with weather information)
    def get_weather_aware(self):
        return self.weather_aware
    def set_weather_aware(self, x):
        weather = eval(x)
        if weather:
            self.dataset_d['weather'] = os.path.join(self.data_path, '01h_weather_filledNorm.h5')
        elif not weather and 'weather' in self.dataset_d.keys():
            _ = self.dataset_d.pop('weather')
        self.weather_aware = weather

    def update_datasets(self, current, new, update_raw=False):
        self.dataset = self.dataset.replace(str(current), str(new))
        self.dataset_d['time_series'] = self.dataset_d['time_series'].replace(str(current), str(new))
        if update_raw:
            self.dataset_raw = self.dataset.replace(str(current), str(new))
            self.dataset_d_raw['time_series'] = self.dataset_d_raw['time_series'].replace(str(current), str(new))

    def update_scaling(self):
        with tb.open_file(self.dataset_d['time_series'], mode='r') as h5_file:
            if self.scaling == 'stand':
                self.vc_mean_map = h5_file.root._v_attrs['vc_mean_map']
                self.vc_std_map = h5_file.root._v_attrs['vc_std_map']
            elif self.scaling == 'norm':
                self.vc_min_map = h5_file.root._v_attrs['vc_min_map']
                self.vc_max_map = h5_file.root._v_attrs['vc_max_map']
