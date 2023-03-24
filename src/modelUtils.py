from keras import layers, models, optimizers, utils
from keras import Model as kModel


class Models():
    '''
    Class that contains the hardcoded models.
    Also allows to load a pre-trained model from an HDF5 file.
    '''
    
    def __init__(self, n_x, n_y, n_v,
                 grid_shape=tuple(), act_out='sigmoid',
                 loss='mse', metrics=['mae', 'mse'],
                 optimizer=optimizers.Adadelta, lr=1.0):
        self.n_x = n_x
        self.n_y = n_y
        self.n_v = n_v
        self.grid_shape = grid_shape
        self.act_out = act_out
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr = lr
        self.models_d = {'ST-AQF': self.__ST_AQF,       'ST-AQF-w': self.__ST_AQF_w,
                         'ST-AQF-t': self.__ST_AQF_t,   'ST-AQF-star': self.__ST_AQF_star,
                         'ST-MDF-star-0': self.__ST_MDF_star_0, 'ST-MDF-star-1': self.__ST_MDF_star_1,
                         'fc': self.__fc, 'FC': self.__FC, 'Fc': self.__Fc,
                         'lstm': self.__lstm,  'biLstm': self.__biLstm,  'rnn': self.__rnn,
                         'conv3d': self.__conv3d, 'persistence': self.__persistence, 'naive': self.__naive
                        }

    def get_models_list(self):
        return list(self.models_d.keys())
    
    def model(self, mod_name):
        if mod_name in self.models_d.keys():
            return self.models_d[mod_name]()
        print('Model not found!')
        return None

    def load_model(self, model_path, do_compile=False):
        model = models.load_model(model_path)
        if do_compile:
                self.__compile_model(model)
        return model

    def plot_model(self, model, fpath):
        utils.plot_model(model, to_file=fpath,
                         show_shapes=True, show_layer_names=False)
        print('\n\tModel graph saved to {}'.format(fpath))

    def __compile_model(self, model):
        opt = self.optimizer(lr=self.lr) if self.lr > 0 else self.optimizer()
        model.compile(optimizer=opt, loss=self.loss, metrics=self.metrics)

    def __ST_AQF(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, self.n_v), name='time_series')
        x0 = layers.ConvLSTM2D(filters=4, kernel_size=(4, 3),
                                                             return_sequences=False)(input_maps)
        x0 = layers.MaxPool2D((2, 2))(x0)
        x0 = layers.Flatten()(x0)

        x3 = layers.Dense(1 * 32 * 28 * 1)(x0)
        x3 = layers.Reshape((1, 32, 28, 1))(x3)
        x3 = layers.convolutional.Conv3DTranspose(filters=self.n_v, kernel_size=(self.n_y, 4, 3))(x3)
        model = kModel(inputs=[input_maps], outputs=x3, name='ST-MDF')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __ST_AQF_w(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, self.n_v), name='time_series')
        x0 = layers.ConvLSTM2D(filters=4, kernel_size=(4, 3),
                                                             return_sequences=False)(input_maps)
        x0 = layers.MaxPool2D((2, 2))(x0)
        x0 = layers.Flatten()(x0)

        weather = layers.Input(shape=(self.n_x, 30), name='weather')
        x2 = layers.LSTM(128)(weather)

        x3 = layers.Concatenate()([x0, x2])
        x3 = layers.Dense(1 * 32 * 28 * 1)(x3)
        x3 = layers.Reshape((1, 32, 28, 1))(x3)
        x3 = layers.convolutional.Conv3DTranspose(filters=self.n_v, kernel_size=(self.n_y, 4, 3))(x3)
        model = kModel(inputs=[input_maps, weather], outputs=x3, name='ST-MDF-w')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __ST_AQF_t(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, self.n_v), name='time_series')
        x0 = layers.ConvLSTM2D(filters=4, kernel_size=(4, 3),
                                                             return_sequences=False)(input_maps)
        x0 = layers.MaxPool2D((2, 2))(x0)
        x0 = layers.Flatten()(x0)

        features = layers.Input(shape=(8, ), name='time')
        x1 = layers.Dense(128)(features)

        x3 = layers.Concatenate()([x0, x1])
        x3 = layers.Dense(1 * 32 * 28 * 1)(x3)
        x3 = layers.Reshape((1, 32, 28, 1))(x3)
        x3 = layers.convolutional.Conv3DTranspose(filters=self.n_v, kernel_size=(self.n_y, 4, 3))(x3)
        model = kModel(inputs=[input_maps, features], outputs=x3, name='ST-MDF-t')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __ST_AQF_star(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, self.n_v), name='time_series')
        x0 = layers.ConvLSTM2D(filters=4, kernel_size=(4, 3),
                                                             return_sequences=False)(input_maps)
        x0 = layers.MaxPool2D((2, 2))(x0)
        x0 = layers.Flatten()(x0)

        features = layers.Input(shape=(8, ), name='time')
        x1 = layers.Dense(128)(features)

        weather = layers.Input(shape=(self.n_x, 30), name='weather')
        x2 = layers.LSTM(128)(weather)

        x3 = layers.Concatenate()([x0, x1, x2])
        x3 = layers.Dense(1 * 32 * 28 * 1)(x3)
        x3 = layers.Reshape((1, 32, 28, 1))(x3)
        x3 = layers.convolutional.Conv3DTranspose(filters=self.n_v, kernel_size=(self.n_y, 4, 3))(x3)
        model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='ST-MDF-star')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __ST_MDF_star_0(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, 2), name='time_series')
        x0 = layers.ConvLSTM2D(filters=3, kernel_size=(8, 8),
                                                             return_sequences=False)(input_maps)
        x0 = layers.MaxPool2D((4, 4))(x0)
        x0 = layers.Flatten()(x0)

        features = layers.Input(shape=(8, ), name='time')
        #x1 = layers.Dense(16)(features)

        weather = layers.Input(shape=(self.n_x, 8), name='weather')
        x2 = layers.Flatten()(weather)
        #x2 = layers.LSTM(16)(weather)

        x3 = layers.Concatenate()([x0, features, x2])
        x3 = layers.Dense(1 * 43 * 28 * 1)(x3)
        x3 = layers.Reshape((1, 43, 28, 1))(x3)
        x3 = layers.convolutional.Conv3DTranspose(filters=2, kernel_size=(self.n_y, 6, 6), strides=(2,2,2))(x3)
        model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='ST-MDF-star-0')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __ST_MDF_star_1(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, 2), name='time_series')
        x0 = layers.ConvLSTM2D(filters=3, kernel_size=(8, 8),
                                                             return_sequences=False)(input_maps)
        x0 = layers.MaxPool2D((4, 4))(x0)
        x0 = layers.Flatten()(x0)

        features = layers.Input(shape=(8, ), name='time')
        #x1 = layers.Dense(16)(features)

        weather = layers.Input(shape=(self.n_x, 8), name='weather')
        x2 = layers.Flatten()(weather)
        #x2 = layers.LSTM(16)(weather)

        x3 = layers.Concatenate()([x0, features, x2])
        x3 = layers.Dense(256)(x3)
        x3 = layers.Dense(1 * 43 * 28 * 1)(x3)
        x3 = layers.Reshape((1, 43, 28, 1))(x3)
        x3 = layers.convolutional.Conv3DTranspose(filters=2, kernel_size=(self.n_y, 6, 6), strides=(2,2,2))(x3)
        model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='ST-MDF-star-1')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __fc(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, 2), name='time_series')
        x0 = layers.Flatten()(input_maps)
        
        features = layers.Input(shape=(8, ), name='time')

        weather = layers.Input(shape=(self.n_x, 8), name='weather')
        x2 = layers.Flatten()(weather)

        x3 = layers.Concatenate()([x0, features, x2])
        x3 = layers.Dense(32)(x3)
        x3 = layers.Dense(self.n_y * self.grid_shape[0] * self.grid_shape[1] * 2)(x3)
        x3 = layers.Reshape((self.n_y, self.grid_shape[0], self.grid_shape[1], 2))(x3)
        model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='fc')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __Fc(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, 2), name='time_series')
        x0 = layers.Flatten()(input_maps)

        x3 = layers.Dense(32)(x0)
        x3 = layers.Dense(self.n_y * self.grid_shape[0] * self.grid_shape[1] * 2)(x3)
        x3 = layers.Reshape((self.n_y, self.grid_shape[0], self.grid_shape[1], 2))(x3)
        model = kModel(inputs=[input_maps], outputs=x3, name='Fc')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __FC(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, 2), name='time_series')
        x0 = layers.Flatten()(input_maps)

        features = layers.Input(shape=(8, ), name='time')

        weather = layers.Input(shape=(self.n_x, 8), name='weather')
        x2 = layers.Flatten()(weather)

        x3 = layers.Concatenate()([x0, features, x2])
        x3 = layers.Dense(32)(x3)
        x3 = layers.Dense(32)(x3)
        x3 = layers.Dense(self.n_y * self.grid_shape[0] * self.grid_shape[1] * 2)(x3)
        x3 = layers.Reshape((self.n_y, self.grid_shape[0], self.grid_shape[1], 2))(x3)
        model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='FC')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __lstm(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, self.n_v), name='time_series')
        x0 = layers.Reshape((self.n_x, self.grid_shape[0] * self.grid_shape[1] * self.n_v))(input_maps)
        x3 = layers.LSTM(16, return_sequences=False)(x0)

        #features = layers.Input(shape=(8, ), name='time')
        #x1 = layers.Dense(16)(features)
        #weather = layers.Input(shape=(self.n_x, 8), name='weather')
        #x2 = layers.LSTM(16)(weather)
        #x2 = layers.Flatten()(x2)

        #x3 = layers.Concatenate()([x3, x1, x2])
        x3 = layers.Dense(32)(x3)
        x3 = layers.Dense(self.n_y * self.grid_shape[0] * self.grid_shape[1] * self.n_v)(x3)
        x3 = layers.Reshape((self.n_y, self.grid_shape[0], self.grid_shape[1], self.n_v))(x3)
        #model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='lstm')
        model = kModel(inputs=[input_maps], outputs=x3, name='lstm')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __biLstm(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, self.n_v), name='time_series')
        x0 = layers.Reshape((self.n_x, self.grid_shape[0] * self.grid_shape[1] * self.n_v))(input_maps)
        x3 = layers.Bidirectional(layers.LSTM(16, return_sequences=False))(x0)

        #features = layers.Input(shape=(8, ), name='time')
        #weather = layers.Input(shape=(self.n_x, 8), name='weather')
        #x2 = layers.Flatten()(weather)

        #x3 = layers.Concatenate()([x3, features, x2])
        x3 = layers.Dense(self.n_y * self.grid_shape[0] * self.grid_shape[1] * self.n_v)(x3)
        x3 = layers.Reshape((self.n_y, self.grid_shape[0], self.grid_shape[1], self.n_v))(x3)
        #model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='biLstm')
        model = kModel(inputs=[input_maps], outputs=x3, name='biLstm')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __rnn(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, 2), name='time_series')
        x0 = layers.Reshape((self.n_x, self.grid_shape[0] * self.grid_shape[1] * 2))(input_maps)
        x3 = layers.SimpleRNN(16, return_sequences=False)(x0)

        features = layers.Input(shape=(8, ), name='time')
        x1 = layers.Dense(16)(features)
        weather = layers.Input(shape=(self.n_x, 8), name='weather')
        x2 = layers.LSTM(16)(weather)
        x2 = layers.Flatten()(x2)

        x3 = layers.Concatenate()([x3, x1, x2])
        x3 = layers.Dense(self.n_y * self.grid_shape[0] * self.grid_shape[1] * 2)(x3)
        x3 = layers.Reshape((self.n_y, self.grid_shape[0], self.grid_shape[1], 2))(x3)
        model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='rnn')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __conv3d(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, self.n_v), name='time_series')
        x0 = layers.Conv3D(filters=3, kernel_size=(8, 8, 8), padding='same')(input_maps)
        x0 = layers.MaxPool3D((1, 4, 4))(x0)
        x0 = layers.Flatten()(x0)

        #features = layers.Input(shape=(8, ), name='time')
        #x1 = layers.Dense(16)(features)

        #weather = layers.Input(shape=(self.n_x, 8), name='weather')
        #x2 = layers.LSTM(16)(weather)

        #x3 = layers.Concatenate()([x0, x1, x2])
        #x3 = layers.Dense(1 * 43 * 28 * 1)(x3)
        #x3 = layers.Dense(1 * 43 * 28 * 1)(x0)
        #x3 = layers.Reshape((1, 43, 28, 1))(x3)
        x3 = layers.Dense(1 * 32 * 28 * 1)(x0)
        x3 = layers.Reshape((1, 32, 28, 1))(x3)
        #x3 = layers.convolutional.Conv3DTranspose(filters=2, kernel_size=(self.n_y, 6, 6), strides=(2,2,2))(x3)
        x3 = layers.convolutional.Conv3DTranspose(filters=self.n_v, kernel_size=(self.n_y, 4, 3))(x3)
        #model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='conv3d')
        model = kModel(inputs=[input_maps], outputs=x3, name='conv3d')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __persistence(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, self.n_v), name='time_series')
        x0 = input_maps[:, -1:]
        x0 = layers.Concatenate(axis=1)([x0 for i in range(self.n_y)])
        model = kModel(inputs=[input_maps], outputs=x0, name='persistence')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __naive(self):
        input_maps = layers.Input(shape=(self.n_x, *self.grid_shape, self.n_v), name='time_series')
        x0 = layers.Add()([input_maps[:, i] for i in range(self.n_x)]) / self.n_x
        x0 = layers.Reshape((1, *self.grid_shape, self.n_v))(x0)
        x0 = layers.Concatenate(axis=1)([x0 for i in range(self.n_y)])
        model = kModel(inputs=[input_maps], outputs=x0, name='naive')
        # Compile model and return
        self.__compile_model(model)
        return model

    # Size of the input window
    def get_n_x(self):
        return self.n_x 
    def set_n_x(self, x):
        self.n_x = int(x)

    # Size of the output window (number of forecast horizons)
    def get_n_y(self):
        return self.n_y
    def set_n_y(self, x):
        self.n_y = int(x)

    # Number of target variables
    def get_n_v(self):
        return self.n_v
    def set_n_v(self, x):
        self.n_v = int(x)

    # Map shape
    def get_grid_shape(self):
        return self.grid_shape
    def set_grid_shape(self, x):
        self.grid_shape = x

    # Activation function for the last layer
    def get_act_out(self):
        return self.act_out
    def set_act_out(self, x):
        self.act_out = x

    # Optimizer
    def get_optimizer(self):
        return self.optimizer
    def set_optimizer(self, x):
        self.optimizer = eval('optimizers.' + x)

    # Loss
    def get_loss(self):
        return self.loss
    def set_loss(self, x):
        self.loss = x

    # Metrics
    def get_metrics(self):
        return self.metrics
    def set_metrics(self, x):
        self.metrics = eval(x)

    # Learning rate
    def get_lr(self):
        return self.lr
    def set_lr(self, x):
        self.lr = float(x)
