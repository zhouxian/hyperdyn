import modules.dynamics.core.keras as keras
from modules.dynamics.core.layers import MLP
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from modules.utils.serializable import Serializable
from modules.utils import tensor_utils
from modules.logger import logger
import time


class MetaMLPDynamicsModel(keras.Net):
    """
    Class for MLP continous dynamics model
    """

    _activations = {
        None: None,
        "relu": tf.nn.relu,
        "tanh": tf.tanh,
        "sigmoid": tf.sigmoid,
        "softmax": tf.nn.softmax,
        "swish": lambda x: x * tf.sigmoid(x)
    }

    def __init__(self,
                 name,
                 env,
                 hidden_sizes=(512, 512),
                 meta_batch_size=10,
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 batch_size=500,
                 learning_rate=0.001,
                 inner_learning_rate=0.1,
                 normalize_input=True,
                 optimizer=tf.keras.optimizers.Adam,
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 p_name_scope="",
                 max_data_size=None,
                 num_steps_per_epoch_mult=None,
                 ):

        super(MetaMLPDynamicsModel, self).__init__(name=name)
        self.update_name_scope(p_name_scope)

        self.normalization = None
        self.normalize_input = normalize_input
        self.next_batch = None
        self.meta_batch_size = meta_batch_size

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.inner_learning_rate = inner_learning_rate
        self._full_dataset_train = None
        self._full_dataset_test = None
        self._adapted = False
        self.max_data_size = max_data_size
        self.num_steps_per_epoch_mult = num_steps_per_epoch_mult
        if self.max_data_size is not None:
            self.max_train_size = int(max_data_size * (1 - valid_split_ratio))
            self.max_test_size = max_data_size - self.max_train_size
        else:
            self.max_train_size = 0
            self.max_test_size = 0

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env.action_space.shape[0]

        hidden_nonlinearity = self._activations[hidden_nonlinearity]
        output_nonlinearity = self._activations[output_nonlinearity]


        self.net = MLP(name=name,
                      output_dim=obs_space_dims,
                      hidden_sizes=hidden_sizes,
                      hidden_nonlinearity=hidden_nonlinearity,
                      output_nonlinearity=output_nonlinearity)

        self.net_copies = []
        for idx in range(self.meta_batch_size):
            self.net_copies.append(MLP(name=f'{name}_copy_{idx}',
                                      output_dim=obs_space_dims,
                                      hidden_sizes=hidden_sizes,
                                      hidden_nonlinearity=hidden_nonlinearity,
                                      output_nonlinearity=output_nonlinearity))
        self.weight_uninited = [True] * self.meta_batch_size

        self.optimizer = optimizer(self.learning_rate)

    def forward(self, obs, act, training, net=None, numpy=False):
        if numpy:
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)
            act = tf.convert_to_tensor(act, dtype=tf.float32)

        if net is None:
            net = self.net

        nn_input = tf.concat([obs, act], axis=1)
        delta_pred = net(nn_input, training=training)
        return delta_pred



    def fit(self, obs, act, obs_next, env_info, epochs=1000, compute_normalization=True,
            valid_split_ratio=None, rolling_average_persitency=None, verbose=False, log_tabular=False, itr=0, tb_writer=None):

        assert obs.ndim == 3 and obs.shape[2] == self.obs_space_dims
        assert obs_next.ndim == 3 and obs_next.shape[2] == self.obs_space_dims
        assert act.ndim == 3 and act.shape[2] == self.action_space_dims

        if valid_split_ratio is None: valid_split_ratio = self.valid_split_ratio
        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        assert 1 > valid_split_ratio >= 0

        if (self.normalization is None or compute_normalization) and self.normalize_input:
            self.compute_normalization(obs, act, obs_next)

        if self.normalize_input:
            # Normalize data
            obs, act, delta = self._normalize_data(obs, act, obs_next)
            assert obs.ndim == act.ndim == obs_next.ndim == 3
        else:
            delta = obs_next - obs

        # Split into valid and test set
        obs_train, act_train, delta_train, obs_test, act_test, delta_test = train_test_split(obs, act, delta,
                                                                                             test_split_ratio=valid_split_ratio)
        if self._full_dataset_test is None:
            self._full_dataset_test = dict(obs=obs_test, act=act_test, delta=delta_test)
            self._full_dataset_train = dict(obs=obs_train, act=act_train, delta=delta_train)
        else:
            self._full_dataset_test['obs'] = np.concatenate([self._full_dataset_test['obs'], obs_test])
            self._full_dataset_test['act'] = np.concatenate([self._full_dataset_test['act'], act_test])
            self._full_dataset_test['delta'] = np.concatenate([self._full_dataset_test['delta'], delta_test])

            self._full_dataset_train['obs'] = np.concatenate([self._full_dataset_train['obs'], obs_train])
            self._full_dataset_train['act'] = np.concatenate([self._full_dataset_train['act'], act_train])
            self._full_dataset_train['delta'] = np.concatenate([self._full_dataset_train['delta'], delta_train])

        if self.max_data_size is not None and len(self._full_dataset_train['obs']) > self.max_train_size:
            train_rand_ids = np.random.randint(0, len(self._full_dataset_train['obs']), self.max_train_size)
            self._dataset_train = dict(obs=self._full_dataset_train['obs'][train_rand_ids],
                                        act=self._full_dataset_train['act'][train_rand_ids],
                                        delta=self._full_dataset_train['delta'][train_rand_ids])

            test_rand_ids = np.random.randint(0, len(self._full_dataset_test['obs']), self.max_test_size)
            self._dataset_test = dict(obs=self._full_dataset_test['obs'][test_rand_ids],
                                        act=self._full_dataset_test['act'][test_rand_ids],
                                        delta=self._full_dataset_test['delta'][test_rand_ids])
        else:
            self._dataset_train = self._full_dataset_train
            self._dataset_test = self._full_dataset_test


        valid_loss_rolling_average = None
        epoch_times = []

        """ ------- Looping over training epochs ------- """
        num_steps_per_epoch = max(int(np.prod(self._dataset_train['obs'].shape[:2])
                                  / (self.meta_batch_size * self.batch_size * 2)), 1)
        if self.num_steps_per_epoch_mult is not None:
            num_steps_per_epoch = int(num_steps_per_epoch * self.num_steps_per_epoch_mult)

        num_steps_test = max(int(np.prod(self._dataset_test['obs'].shape[:2])
                                 / (self.meta_batch_size * self.batch_size * 2)), 1)


        for epoch in range(epochs):
            total_epoch = itr * epochs + epoch

            # preparations for recording training stats
            pre_batch_losses = []
            post_batch_losses = []
            t0 = time.time()

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""
            for _ in range(num_steps_per_epoch):
                obs_batch, act_batch, delta_batch = self._get_batch(train=True)
                obs_per_task = tf.split(obs_batch, self.meta_batch_size, axis=0)
                act_per_task = tf.split(act_batch, self.meta_batch_size, axis=0)
                delta_per_task = tf.split(delta_batch, self.meta_batch_size, axis=0)

                pre_obs_per_task, post_obs_per_task = zip(*[tf.split(obs, 2, axis=0) for obs in obs_per_task])
                pre_act_per_task, post_act_per_task = zip(*[tf.split(act, 2, axis=0) for act in act_per_task])
                pre_delta_per_task, post_delta_per_task = zip(*[tf.split(delta, 2, axis=0) for delta in delta_per_task])

                pre_losses = []
                post_losses = []
                with tf.GradientTape() as test_tape:

                    for idx in range(self.meta_batch_size):
                        with tf.GradientTape() as train_tape:
                            pre_delta_pred = self.forward(pre_obs_per_task[idx], pre_act_per_task[idx], training=True, net=self.net)
                            pre_loss = tf.reduce_mean(tf.square(pre_delta_per_task[idx] - pre_delta_pred))

                        gradients = train_tape.gradient(pre_loss, self.net.trainable_variables)

                        # copy net and train
                        if self.weight_uninited[idx]: # create weights
                            self.forward(pre_obs_per_task[idx], pre_act_per_task[idx], training=True, net=self.net_copies[idx])
                            self.weight_uninited[idx] = False

                        # self.net_copies[idx].set_weights(self.net.get_weights()) # NB: i think this step is unnecessary, and it causes problem

                        k = 0
                        for j in range(len(self.net_copies[idx].net.layers)):
                            self.net_copies[idx].net.layers[j].kernel = tf.subtract(self.net.net.layers[j].kernel,
                                        tf.multiply(self.inner_learning_rate, gradients[k]))
                            self.net_copies[idx].net.layers[j].bias = tf.subtract(self.net.net.layers[j].bias,
                                        tf.multiply(self.inner_learning_rate, gradients[k+1]))
                            k += 2

                        post_delta_pred = self.forward(post_obs_per_task[idx], post_act_per_task[idx], training=True, net=self.net_copies[idx])
                        post_loss = tf.reduce_mean(tf.square(post_delta_per_task[idx] - post_delta_pred))

                        pre_losses.append(pre_loss)
                        post_losses.append(post_loss)

                    pre_batch_loss = tf.reduce_mean(pre_losses)
                    post_batch_loss = tf.reduce_mean(post_losses)

                gradients = test_tape.gradient(post_batch_loss, self.net.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))
                
                pre_batch_losses.append(pre_batch_loss)
                post_batch_losses.append(post_batch_loss)


            valid_losses = []
            for _ in range(num_steps_test):
                obs_test, act_test, delta_test = self._get_batch(train=False)

                delta_pred = self.forward(obs_test, act_test, training=False)
                valid_loss = tf.reduce_mean(tf.square(delta_test - delta_pred))
                valid_losses.append(valid_loss)

            valid_loss = np.mean(valid_losses)
            if valid_loss_rolling_average is None:
                valid_loss_rolling_average = 1.5 * valid_loss  # set initial rolling to a higher value avoid too early stopping
                valid_loss_rolling_average_prev = 2 * valid_loss
                if valid_loss < 0:
                    valid_loss_rolling_average = valid_loss/1.5  # set initial rolling to a higher value avoid too early stopping
                    valid_loss_rolling_average_prev = valid_loss/2

            valid_loss_rolling_average = rolling_average_persitency*valid_loss_rolling_average \
                                         + (1.0-rolling_average_persitency)*valid_loss

            epoch_times.append(time.time() - t0)

            if verbose:
                logger.log("Training DynamicsModel - finished epoch %i - "
                           "train loss: %.4f   valid loss: %.4f   valid_loss_mov_avg: %.4f   epoch time: %.2f"
                           % (epoch, np.mean(post_batch_losses), valid_loss, valid_loss_rolling_average,
                              time.time() - t0))

            # write to tb
            with tb_writer.as_default():
                tf.summary.scalar('epoch/train_loss', np.mean(post_batch_losses), step=total_epoch)
                tf.summary.scalar('epoch/pre_loss', np.mean(pre_batch_losses), step=total_epoch)
                tf.summary.scalar('epoch/valid_loss', valid_loss, step=total_epoch)
                tf.summary.scalar('epoch/valid_loss_rolling_average', valid_loss_rolling_average, step=total_epoch)
                tf.summary.scalar('epoch/epoch_time', time.time() - t0, step=total_epoch)


            # if valid_loss_rolling_average_prev < valid_loss_rolling_average or epoch == epochs - 1:
            #     logger.log('Stopping Training of Model since its valid_loss_rolling_average decreased')
            if epoch == epochs - 1:
                logger.log('Iteration done')
                break
            valid_loss_rolling_average_prev = valid_loss_rolling_average


        """ ------- Tabular Logging ------- """
        if log_tabular:
            logger.logkv('AvgModelEpochTime', np.mean(epoch_times))
            logger.logkv('Post-Loss', np.mean(post_batch_losses))
            logger.logkv('Pre-Loss', np.mean(pre_batch_losses))
            logger.logkv('Epochs', epoch)

    def predict(self, obs, act, env_info):

        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs

        if self.normalize_input:
            obs, act = self._normalize_data(obs, act)
            delta = self._predict(obs, act)
            delta = denormalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
        else:
            delta = self._predict(obs, act)

        assert delta.ndim == 2
        pred_obs = obs_original + delta

        return pred_obs

    def _predict(self, obs, act):
        if self._adapted:
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)
            act = tf.convert_to_tensor(act, dtype=tf.float32)
            obs_per_task = tf.split(obs, self._num_adapted_models, axis=0)
            act_per_task = tf.split(act, self._num_adapted_models, axis=0)

            deltas = []
            for idx in range(self._num_adapted_models):
                deltas.append(self.forward(obs_per_task[idx], act_per_task[idx], training=False, net=self.net_copies[idx]).numpy())
            delta = np.concatenate(deltas, axis=0)
        else:
            delta = self.forward(obs, act, numpy=True, training=False).numpy()

        return delta

    def _pad_inputs(self, obs, act, obs_next=None):
        if self._num_adapted_models < self.meta_batch_size:
            pad = int(obs.shape[0] / self._num_adapted_models * (self.meta_batch_size - self._num_adapted_models))
            obs = np.concatenate([obs, np.zeros((pad,) + obs.shape[1:])], axis=0)
            act = np.concatenate([act, np.zeros((pad,) + act.shape[1:])], axis=0)
            if obs_next is not None:
                obs_next = np.concatenate([obs_next, np.zeros((pad,) + obs_next.shape[1:])], axis=0)

        if obs_next is not None:
            return obs, act, obs_next
        else:
            return obs, act

    def adapt(self, obs, act, obs_next):
        self._num_adapted_models = len(obs)
        assert len(obs) == len(act) == len(obs_next)
        obs = np.concatenate(obs, axis=0)
        act = np.concatenate(act, axis=0)
        obs_next = np.concatenate(obs_next, axis=0)

        assert self._num_adapted_models <= self.meta_batch_size
        assert obs.shape[0] == act.shape[0] == obs_next.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims
        assert obs_next.ndim == 2 and obs_next.shape[1] == self.obs_space_dims

        if self.normalize_input:
            # Normalize data
            obs, act, delta = self._normalize_data(obs, act, obs_next)
            assert obs.ndim == act.ndim == obs_next.ndim == 2
        else:
            delta = obs_next - obs

        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        act = tf.convert_to_tensor(act, dtype=tf.float32)
        delta = tf.convert_to_tensor(delta, dtype=tf.float32)

        obs_per_task = tf.split(obs, self._num_adapted_models, axis=0)
        act_per_task = tf.split(act, self._num_adapted_models, axis=0)
        delta_per_task = tf.split(delta, self._num_adapted_models, axis=0)

        for idx in range(self._num_adapted_models):
            with tf.GradientTape() as train_tape:
                pre_delta_pred = self.forward(obs_per_task[idx], act_per_task[idx], training=False, net=self.net)
                pre_loss = tf.reduce_mean(tf.square(delta_per_task[idx] - pre_delta_pred))

            gradients = train_tape.gradient(pre_loss, self.net.trainable_variables)

            k = 0
            for j in range(len(self.net_copies[idx].net.layers)):
                self.net_copies[idx].net.layers[j].kernel = tf.subtract(self.net.net.layers[j].kernel,
                            tf.multiply(self.inner_learning_rate, gradients[k]))
                self.net_copies[idx].net.layers[j].bias = tf.subtract(self.net.net.layers[j].bias,
                            tf.multiply(self.inner_learning_rate, gradients[k+1]))
                k += 2

        self._adapted = True

    def switch_to_pre_adapt(self):
        self._adapted = False

    def _get_batch(self, train=True):
        if train:
            num_paths, len_path = self._dataset_train['obs'].shape[:2]
            idx_path = np.random.randint(0, num_paths, size=self.meta_batch_size)
            idx_batch = np.random.randint(self.batch_size, len_path - self.batch_size, size=self.meta_batch_size)

            obs_batch = np.concatenate([self._dataset_train['obs'][ip,
                                        ib - self.batch_size:ib + self.batch_size, :]
                                        for ip, ib in zip(idx_path, idx_batch)], axis=0)
            act_batch = np.concatenate([self._dataset_train['act'][ip,
                                        ib - self.batch_size:ib + self.batch_size, :]
                                        for ip, ib in zip(idx_path, idx_batch)], axis=0)
            delta_batch = np.concatenate([self._dataset_train['delta'][ip,
                                          ib - self.batch_size:ib + self.batch_size, :]
                                          for ip, ib in zip(idx_path, idx_batch)], axis=0)

        else:
            num_paths, len_path = self._dataset_test['obs'].shape[:2]
            idx_path = np.random.randint(0, num_paths, size=self.meta_batch_size)
            idx_batch = np.random.randint(self.batch_size, len_path - self.batch_size, size=self.meta_batch_size)

            obs_batch = np.concatenate([self._dataset_test['obs'][ip,
                                        ib - self.batch_size:ib + self.batch_size, :]
                                        for ip, ib in zip(idx_path, idx_batch)], axis=0)
            act_batch = np.concatenate([self._dataset_test['act'][ip,
                                        ib - self.batch_size:ib + self.batch_size, :]
                                        for ip, ib in zip(idx_path, idx_batch)], axis=0)
            delta_batch = np.concatenate([self._dataset_test['delta'][ip,
                                          ib - self.batch_size:ib + self.batch_size, :]
                                          for ip, ib in zip(idx_path, idx_batch)], axis=0)

        obs_batch = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
        act_batch = tf.convert_to_tensor(act_batch, dtype=tf.float32)
        delta_batch = tf.convert_to_tensor(delta_batch, dtype=tf.float32)

        return obs_batch, act_batch, delta_batch

    def _normalize_data(self, obs, act, obs_next=None):
        obs_normalized = normalize(obs, self.normalization['obs'][0], self.normalization['obs'][1])
        actions_normalized = normalize(act, self.normalization['act'][0], self.normalization['act'][1])

        if obs_next is not None:
            delta = obs_next - obs
            deltas_normalized = normalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
            return obs_normalized, actions_normalized, deltas_normalized
        else:
            return obs_normalized, actions_normalized

    def compute_normalization(self, obs, act, obs_next):
        assert obs.shape[0] == obs_next.shape[0] == act.shape[0]
        assert obs.shape[1] == obs_next.shape[1] == act.shape[1]
        delta = obs_next - obs

        assert delta.ndim == 3 and delta.shape[2] == obs_next.shape[2] == obs.shape[2]

        # store means and std in dict
        self.normalization = OrderedDict()
        self.normalization['obs'] = (np.mean(obs, axis=(0, 1)), np.std(obs, axis=(0, 1)))
        self.normalization['delta'] = (np.mean(delta, axis=(0, 1)), np.std(delta, axis=(0, 1)))
        self.normalization['act'] = (np.mean(act, axis=(0, 1)), np.std(act, axis=(0, 1)))

    def _adapt_sym(self, loss, params_var):
        update_param_keys = list(params_var.keys())

        grads = tf.gradients(loss, [params_var[key] for key in update_param_keys])
        gradients = dict(zip(update_param_keys, grads))

        # Gradient descent
        adapted_policy_params = [params_var[key] - tf.multiply(self.inner_learning_rate, gradients[key])
                          for key in update_param_keys]

        adapted_policy_params_dict = OrderedDict(zip(update_param_keys, adapted_policy_params))

        return adapted_policy_params_dict

    def _create_placeholders_for_vars(self, vars):
        placeholders = OrderedDict()
        for key, var in vars.items():
            placeholders[key] = tf.placeholder(tf.float32, shape=var.shape, name=key + '_ph')
        return OrderedDict(placeholders)

    @property
    def network_params_feed_dict(self):
        return dict(list((self.network_phs_meta_batch[i][key], self._adapted_param_values[i][key])
                         for key in self._adapted_param_values[0].keys() for i in range(self._num_adapted_models)))

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        state['normalization'] = self.normalization
        state['networks'] = [nn.__getstate__() for nn in self._networks]
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
        self.normalization = state['normalization']
        for i in range(len(self._networks)):
            self._networks[i].__setstate__(state['networks'][i])


def normalize(data_array, mean, std):
    return (data_array - mean) / (std + 1e-10)


def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean


def train_test_split(obs, act, delta, test_split_ratio=0.2):
    assert obs.shape[0] == act.shape[0] == delta.shape[0]
    dataset_size = obs.shape[0]
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split_idx = int(dataset_size * (1-test_split_ratio))

    idx_train = indices[:split_idx]
    idx_test = indices[split_idx:]
    assert len(idx_train) + len(idx_test) == dataset_size

    return obs[idx_train, :], act[idx_train, :], delta[idx_train, :], \
           obs[idx_test, :], act[idx_test, :], delta[idx_test, :]
