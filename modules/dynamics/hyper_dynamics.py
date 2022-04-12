import modules.dynamics.core.keras as keras
from modules.dynamics.core.layers import *
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from modules.utils.serializable import Serializable
from modules.utils import tensor_utils
from modules.logger import logger
import time



class HyperDynamicsModel(keras.Net):
    """
    Class for MLP continous dynamics model
    """
    
    _activations = {
        None: tf.identity,
        "relu": tf.nn.relu,
        "tanh": tf.tanh,
        "sigmoid": tf.sigmoid,
        "softmax": tf.nn.softmax,
        "swish": lambda x: x * tf.sigmoid(x)
    }

    def __init__(self,
                 name,
                 env,
                 extra,
                 hidden_sizes=(512, 512),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 batch_size=500,
                 learning_rate=0.001,
                 normalize_input=True,
                 optimizer=tf.keras.optimizers.Adam,
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 p_name_scope="",
                 max_data_size=None,
                 ):

        super(HyperDynamicsModel, self).__init__(name=name)
        self.update_name_scope(p_name_scope)

        self.normalization = None
        self.normalize_input = normalize_input
        self.next_batch = None

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

        self.batch_size = batch_size
        self.adapt_batch_size = extra['adapt_batch_size']
        self.learning_rate = learning_rate
        self._full_dataset_train = None
        self._full_dataset_test = None
        self.max_data_size = max_data_size
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

        hypernet_init_std = extra['init_std']
        targetnet_n_params = self.compute_n_params(self.obs_space_dims+self.action_space_dims, hidden_sizes, obs_space_dims)

        self.encoder = MLP(name=name+'_encoder',
                      output_dim=extra['enc_out_dim'],
                      hidden_sizes=extra['hidden_sizes'])

        self.train_dec = extra['train_dec']

        self.freeze_enc = extra.get('freeze_enc', False)
        self.freeze_enc_itr = extra.get('freeze_enc_itr', 0)

        self.hypernet = HyperNet(name=name, output_dim=targetnet_n_params, init_std=hypernet_init_std)

        self.targetnet = HyperMLP(name=name,
                      output_dim=obs_space_dims,
                      hidden_sizes=hidden_sizes,
                      hidden_nonlinearity=hidden_nonlinearity,
                      output_nonlinearity=output_nonlinearity)

        grad_clipvalue = extra.get('grad_clipvalue', None)
        if grad_clipvalue is not None:
            self.optimizer = optimizer(self.learning_rate, clipvalue=grad_clipvalue)
        else:
            self.optimizer = optimizer(self.learning_rate)

    def compute_n_params(self, n_in, hidden_sizes, n_out):
        layer_sizes = [n_in] + list(hidden_sizes) + [n_out]
        n_params = 0
        for i in range(len(layer_sizes)-1):
            n_params += (layer_sizes[i] + 1) * layer_sizes[i+1]
        return n_params

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
            # normalize data
            obs, act, delta = self._normalize_data(obs, act, obs_next)
            assert obs.ndim == act.ndim == obs_next.ndim == 3
        else:
            delta = obs_next - obs
        # split into valid and test set
        obs_train, act_train, delta_train, info_train, obs_test, act_test, delta_test, info_test = train_test_split(obs, act, delta, env_info, test_split_ratio=valid_split_ratio)
        if self._full_dataset_test is None:
            self._full_dataset_test = dict(obs=obs_test, act=act_test, delta=delta_test, info=info_test)
            self._full_dataset_train = dict(obs=obs_train, act=act_train, delta=delta_train, info=info_train)
        else:
            self._full_dataset_test['obs'] = np.concatenate([self._full_dataset_test['obs'], obs_test])
            self._full_dataset_test['act'] = np.concatenate([self._full_dataset_test['act'], act_test])
            self._full_dataset_test['delta'] = np.concatenate([self._full_dataset_test['delta'], delta_test])
            self._full_dataset_test['info'] = np.concatenate([self._full_dataset_test['info'], info_test])

            self._full_dataset_train['obs'] = np.concatenate([self._full_dataset_train['obs'], obs_train])
            self._full_dataset_train['act'] = np.concatenate([self._full_dataset_train['act'], act_train])
            self._full_dataset_train['delta'] = np.concatenate([self._full_dataset_train['delta'], delta_train])
            self._full_dataset_train['info'] = np.concatenate([self._full_dataset_train['info'], info_train])

        if self.max_data_size is not None and len(self._full_dataset_train['obs']) > self.max_train_size:
            train_rand_ids = np.random.randint(0, len(self._full_dataset_train['obs']), self.max_train_size)
            self._dataset_train = dict(obs=self._full_dataset_train['obs'][train_rand_ids],
                                        act=self._full_dataset_train['act'][train_rand_ids],
                                        delta=self._full_dataset_train['delta'][train_rand_ids],
                                        info=self._full_dataset_train['info'][train_rand_ids])

            test_rand_ids = np.random.randint(0, len(self._full_dataset_test['obs']), self.max_test_size)
            self._dataset_test = dict(obs=self._full_dataset_test['obs'][test_rand_ids],
                                        act=self._full_dataset_test['act'][test_rand_ids],
                                        delta=self._full_dataset_test['delta'][test_rand_ids],
                                        info=self._full_dataset_test['info'][test_rand_ids])
        else:
            self._dataset_train = self._full_dataset_train
            self._dataset_test = self._full_dataset_test

        valid_loss_rolling_average = None
        epoch_times = []

        """ ------- Looping over training epochs ------- """
        num_steps_per_epoch = max(int(np.prod(self._dataset_train['obs'].shape[:2]) / self.batch_size), 1)
        num_steps_test = max(int(np.prod(self._dataset_test['obs'].shape[:2]) / self.batch_size), 1)

        for epoch in range(epochs):
            total_epoch = itr * epochs + epoch

            # preparations for recording training stats
            batch_losses = []
            batch_dec_losses = []
            t0 = time.time()

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""
            for _ in range(num_steps_per_epoch):
                hist_obs_batch, hist_act_batch, hist_delta_batch, hist_info_batch, obs_batch, act_batch, delta_batch, info_batch = self._get_batch(train=True)
                with tf.GradientTape() as tape:
                    if self.freeze_enc and self.freeze_enc_itr < itr:
                        terrain_emb, delta_pred = self.forward(hist_obs_batch, hist_act_batch, obs_batch, act_batch, training=True, freeze_enc=True)
                    else:
                        terrain_emb, delta_pred = self.forward(hist_obs_batch, hist_act_batch, obs_batch, act_batch, training=True)

                    batch_loss = tf.reduce_mean(tf.square(delta_batch - delta_pred))
                    if self.train_dec:
                        # assert tf.reduce_all(tf.equal(hist_info_batch, tf.expand_dims(info_batch, 1)))
                        batch_dec_loss = tf.reduce_mean(tf.square(terrain_emb - info_batch))
                    else:
                        batch_dec_loss = 0

                    batch_total_loss = batch_loss + batch_dec_loss

                gradients = tape.gradient(batch_total_loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                batch_losses.append(batch_loss)
                batch_dec_losses.append(batch_dec_loss)

            # valid
            valid_losses = []
            valid_dec_losses = []
            for _ in range(num_steps_test):
                hist_obs_test, hist_act_test, hist_delta_test, hist_info_test, obs_test, act_test, delta_test, info_test = self._get_batch(train=False)
                terrain_emb, delta_pred = self.forward(hist_obs_test, hist_act_test, obs_test, act_test, training=True)

                valid_loss = tf.reduce_mean(tf.square(delta_test - delta_pred))
                if self.train_dec:
                    # assert tf.reduce_all(tf.equal(hist_info_batch, tf.expand_dims(info_batch, 1)))
                    valid_dec_loss = tf.reduce_mean(tf.square(terrain_emb - info_test))
                else:
                    valid_dec_loss = 0
                valid_total_loss = valid_loss + valid_dec_loss

                valid_losses.append(valid_loss)
                valid_dec_losses.append(valid_dec_loss)

            if valid_loss_rolling_average is None:
                valid_loss_rolling_average = 1.5 * valid_total_loss  # set initial rolling to a higher value avoid too early stopping
                valid_loss_rolling_average_prev = 2 * valid_total_loss
                if valid_total_loss < 0:
                    valid_loss_rolling_average = valid_total_loss/1.5  # set initial rolling to a higher value avoid too early stopping
                    valid_loss_rolling_average_prev = valid_total_loss/2

            valid_loss_rolling_average = rolling_average_persitency*valid_loss_rolling_average \
                                         + (1.0-rolling_average_persitency)*valid_total_loss


            epoch_times.append(time.time() - t0)
            if verbose:
                logger.log("Training DynamicsModel - finished epoch %i --"
                           "train loss: %.4f  valid loss: %.4f  valid_loss_mov_avg: %.4f  epoch time: %.2f"
                           % (epoch, np.mean(batch_losses), valid_loss, valid_loss_rolling_average,
                              time.time() - t0))

            # write to tb
            with tb_writer.as_default():
                tf.summary.scalar('epoch/train_loss', np.mean(batch_losses), step=total_epoch)
                tf.summary.scalar('epoch/train_dec_loss', np.mean(batch_dec_losses), step=total_epoch)
                tf.summary.scalar('epoch/valid_loss', valid_loss, step=total_epoch)
                tf.summary.scalar('epoch/valid_dec_loss', valid_dec_loss, step=total_epoch)
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
            logger.logkv('Epochs', epoch)

    def adapt(self, obs, act):
        assert len(obs) == len(act)
        obs = np.stack(obs, axis=0)
        act = np.stack(act, axis=0)

        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 3 and obs.shape[2] == self.obs_space_dims
        assert act.ndim == 3 and act.shape[2] == self.action_space_dims

        if self.normalize_input:
            # Normalize data
            obs, act = self._normalize_data(obs, act)
            assert obs.ndim == act.ndim

        self.hist_obs_adapt = obs
        self.hist_act_adapt = act

    def forward(self, hist_obs, hist_act, obs, act, training, numpy=False, freeze_enc=False):
        if numpy:
            hist_obs = tf.convert_to_tensor(hist_obs, dtype=tf.float32)
            hist_act = tf.convert_to_tensor(hist_act, dtype=tf.float32)
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)
            act = tf.convert_to_tensor(act, dtype=tf.float32)

        # get terrain embedding
        B = hist_obs.shape[0]
        encoder_input = tf.reshape(tf.concat([hist_obs, hist_act], axis=-1), [B, -1])

        terrain_emb = self.encoder(encoder_input, training=training)
        if freeze_enc:
            terrain_emb = tf.stop_gradient(terrain_emb)
        params = self.hypernet(terrain_emb, training=training)

        if obs.shape[0] > B: # this is for data sampling
            # reshape a bit to work with targetnet
            obs = tf.reshape(obs, [B, -1, obs.shape[-1]])
            act = tf.reshape(act, [B, -1, act.shape[-1]])
            targetnet_input = tf.concat([obs, act], axis=-1)
            delta_pred, _ = self.targetnet(targetnet_input, params, training=training)
            delta_pred = tf.reshape(delta_pred, [-1, delta_pred.shape[-1]])
        else:
            targetnet_input = tf.concat([obs, act], axis=-1)
            delta_pred, _ = self.targetnet(targetnet_input, params, training=training)

        return terrain_emb, delta_pred
        
    def predict(self, obs, act, info):
        assert obs.shape[0] == act.shape[0] == info.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs
        if self.normalize_input:
            obs, act = self._normalize_data(obs, act)
            _, delta = self.forward(self.hist_obs_adapt, self.hist_act_adapt, obs, act, numpy=True, training=False)
            delta = delta.numpy()
            delta = denormalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
        else:
            _, delta = self.forward(self.hist_obs_adapt, self.hist_act_adapt, obs, act, numpy=True, training=False)
            delta = delta.numpy()

        assert delta.ndim == 2

        pred_obs = obs_original + delta

        return pred_obs

    def _get_batch(self, train=True):
        if train:
            dataset = self._dataset_train
        else:
            dataset = self._dataset_test

        num_paths, len_path = dataset['obs'].shape[:2]
        idx_path = np.random.randint(0, num_paths, size=self.batch_size)
        idx_batch = np.random.randint(self.adapt_batch_size, len_path - 1, size=self.batch_size)

        all_obs_batch = np.stack([dataset['obs'][ip, ib - self.adapt_batch_size:ib + 1, :]
                                    for ip, ib in zip(idx_path, idx_batch)], axis=0)
        all_act_batch = np.stack([dataset['act'][ip, ib - self.adapt_batch_size:ib + 1, :]
                                    for ip, ib in zip(idx_path, idx_batch)], axis=0)
        all_delta_batch = np.stack([dataset['delta'][ip, ib - self.adapt_batch_size:ib + 1, :]
                                    for ip, ib in zip(idx_path, idx_batch)], axis=0)
        all_info_batch = np.stack([dataset['info'][ip, ib - self.adapt_batch_size:ib + 1, :]
                                    for ip, ib in zip(idx_path, idx_batch)], axis=0)

        hist_obs_batch = tf.convert_to_tensor(all_obs_batch[:, :self.adapt_batch_size, :], dtype=tf.float32)
        hist_act_batch = tf.convert_to_tensor(all_act_batch[:, :self.adapt_batch_size, :], dtype=tf.float32)
        hist_delta_batch = tf.convert_to_tensor(all_delta_batch[:, :self.adapt_batch_size, :], dtype=tf.float32)
        hist_info_batch = tf.convert_to_tensor(all_info_batch[:, :self.adapt_batch_size, :], dtype=tf.float32)

        obs_batch = tf.convert_to_tensor(all_obs_batch[:, self.adapt_batch_size, :], dtype=tf.float32)
        act_batch = tf.convert_to_tensor(all_act_batch[:, self.adapt_batch_size, :], dtype=tf.float32)
        delta_batch = tf.convert_to_tensor(all_delta_batch[:, self.adapt_batch_size, :], dtype=tf.float32)
        info_batch = tf.convert_to_tensor(all_info_batch[:, self.adapt_batch_size, :], dtype=tf.float32)

        return hist_obs_batch, hist_act_batch, hist_delta_batch, hist_info_batch, obs_batch, act_batch, delta_batch, info_batch

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

def normalize(data_array, mean, std):
    return (data_array - mean) / (std + 1e-10)


def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean


def train_test_split(obs, act, delta, env_info, test_split_ratio=0.2):
    assert obs.shape[0] == act.shape[0] == delta.shape[0] == env_info['info'].shape[0]
    dataset_size = obs.shape[0]
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split_idx = int(dataset_size * (1-test_split_ratio))

    idx_train = indices[:split_idx]
    idx_test = indices[split_idx:]
    assert len(idx_train) + len(idx_test) == dataset_size

    return obs[idx_train, :], act[idx_train, :], delta[idx_train, :], env_info['info'][idx_train, :], \
           obs[idx_test, :], act[idx_test, :], delta[idx_test, :], env_info['info'][idx_test, :]
