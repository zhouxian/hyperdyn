from modules.utils.serializable import Serializable
from modules.utils.utils import remove_scope_from_name
import tensorflow as tf
import copy
from collections import OrderedDict
import modules.dynamics.core.keras as keras


class Layer(Serializable):
    """
    A container for storing the current pre and post update policies
    Also provides functions for executing and updating policy parameters

    Note:
        the preupdate policy is stored as tf.Variables, while the postupdate
        policy is stored in numpy arrays and executed through tf.placeholders

    Args:
        obs_dim (int): dimensionality of the observation space -> specifies the input size of the policy
        action_dim (int): dimensionality of the action space -> specifies the output size of the policy
        name (str) : Name used for scoping variables in policy
        hidden_sizes (tuple) : size of hidden layers of network
        learn_std (bool) : whether to learn variance of network output
        hidden_nonlinearity (Operation) : nonlinearity used between hidden layers of network
        output_nonlinearity (Operation) : nonlinearity used after the final layer of network
    """
    def __init__(self,
                 name,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 input_var=None,
                 params=None,
                 **kwargs
                 ):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.input_var = input_var

        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        self.batch_normalization = kwargs.get('batch_normalization', False)

        self._params = params
        self._assign_ops = None
        self._assign_phs = None

    def build_graph(self):
        """
        Builds computational graph for policy
        """
        raise NotImplementedError

    """ --- methods for serialization --- """

    def get_params(self):
        """
        Get the tf.Variables representing the trainable weights of the network (symbolic)

        Returns:
            (dict) : a dict of all trainable Variables
        """
        return self._params

    def get_param_values(self):
        """
        Gets a list of all the current weights in the network (in original code it is flattened, why?)

        Returns:
            (list) : list of values for parameters
        """
        param_values = tf.get_default_session().run(self._params)
        return param_values

    def set_params(self, policy_params):
        """
        Sets the parameters for the graph

        Args:
            policy_params (dict): of variable names and corresponding parameter values
        """
        assert all([k1 == k2 for k1, k2 in zip(self.get_params().keys(), policy_params.keys())]), \
            "parameter keys must match with variable"

        if self._assign_ops is None:
            assign_ops, assign_phs = [], []
            for var in self.get_params().values():
                assign_placeholder = tf.placeholder(dtype=var.dtype)
                assign_op = tf.assign(var, assign_placeholder)
                assign_ops.append(assign_op)
                assign_phs.append(assign_placeholder)
            self._assign_ops = assign_ops
            self._assign_phs = assign_phs
        feed_dict = dict(zip(self._assign_phs, policy_params.values()))
        tf.get_default_session().run(self._assign_ops, feed_dict=feed_dict)

    def __getstate__(self):
        state = {
            # 'init_args': Serializable.__getstate__(self),
            'network_params': self.get_param_values()
        }
        return state

    def __setstate__(self, state):
        # Serializable.__setstate__(self, state['init_args'])
        tf.get_default_session().run(tf.variables_initializer(self.get_params().values()))
        self.set_params(state['network_params'])

class MLP(keras.Net):
    def __init__(self, output_dim=None, hidden_sizes=[32, 32], momentum=0.9, is_normalize=False, p_name_scope="", name=None, 
                 hidden_nonlinearity=tf.nn.relu, output_nonlinearity=None):
        super(MLP, self).__init__(name=name)
        self.update_name_scope(p_name_scope)
        net = tf.keras.Sequential()
        for layer_id, layer_out_dim in enumerate(hidden_sizes):
            net.add(tf.keras.layers.Dense(layer_out_dim, activation=hidden_nonlinearity, name=f'fc_{layer_id}'))
            if is_normalize:
                net.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum, name=f'batch_normalization_{layer_id}'))
        net.add(tf.keras.layers.Dense(output_dim, activation=output_nonlinearity, name='fc_final'))
        self.net = net

    def call(self, input, training):
        out = self.net(input, training)
        return out

class HyperNet(keras.Net):
    def __init__(self, output_dim, init_std, momentum=0.9, is_normalize=False, p_name_scope="", name=None):
        super(HyperNet, self).__init__(name=name)
        self.update_name_scope(p_name_scope)

        net = tf.keras.Sequential()
        net.add(tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu, name='hypernet_fc_1', kernel_initializer=tf.random_normal_initializer(0., init_std)))
        net.add(tf.keras.layers.Dense(output_dim, name='hypernet_fc_2', kernel_initializer=tf.random_normal_initializer(0., init_std)))
        self.net = net

    def call(self, input, training):
        out = self.net(input, training)
        return out


class HyperMLP(keras.Net):
    def __init__(self, output_dim=None, hidden_sizes=[32, 32], momentum=0.9, is_normalize=False, p_name_scope="", name=None, 
                 hidden_nonlinearity=tf.nn.relu, output_nonlinearity=None):
        super(HyperMLP, self).__init__(name=name)
        self.update_name_scope(p_name_scope)

        self.layer_list = []
        for layer_id, layer_out_dim in enumerate(hidden_sizes):
            self.layer_list.append(HyperDense(layer_out_dim, activation=hidden_nonlinearity, name=f'fc_{layer_id}'))
            if is_normalize:
                self.layer_list.append(tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum, name=f'batch_normalization_{layer_id}'))
        self.layer_list.append(HyperDense(output_dim, activation=output_nonlinearity, name='fc_final'))

    def call(self, x, params, training):
        params_used = 0
        out = x
        for layer_id, layer in enumerate(self.layer_list):
            if layer.name.startswith("batch_normalization"):
                out = layer(out, training=training)
            else:
                outshape = layer.size
                inshape = out.shape[-1]
                w_param = tf.reshape(params[:, params_used: params_used + outshape*inshape], (-1, inshape, outshape))
                params_used += outshape*inshape
                b_param = params[:, params_used: params_used + outshape]
                params_used += outshape
                out = layer(out, w_param, b_param)
        return out, params_used

class HyperDense(tf.keras.layers.Layer):
    def __init__(self, size, activation=None, **kwargs):
        super(HyperDense, self).__init__(**kwargs)
        self.size = size
        self.activation = activation
        
    @tf.function
    def call(self, x, wt, b):
        rank = len(x.shape)
        if rank == 3:
            out = tf.matmul(x, wt) + tf.expand_dims(b, 1)
        elif rank ==2:
            out = tf.matmul(tf.expand_dims(x, 1), wt) + tf.expand_dims(b, 1)
            out = tf.squeeze(out, axis=1)
        else:
            raise Exception('unsupported rank')

        if self.activation is not None:
            return self.activation(out)
        return out
