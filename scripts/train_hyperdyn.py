from modules.dynamics.hyper_dynamics import HyperDynamicsModel 
from modules.trainers.mb_trainer import Trainer
from modules.policies.mpc_controller import MPCController
from modules.samplers.sampler import Sampler
from modules.logger import logger
from modules.envs.normalized_env import normalize
from modules.utils.utils import ClassEncoder
from modules.samplers.model_sample_processor import ModelSampleProcessor
from modules.envs import *
import argparse
import json
import os
import tensorflow as tf


EXP_NAME = 'hyperdyn'


def run_experiment(config, exp_name):
    exp_dir = os.getcwd() + '/log/' + EXP_NAME + '/' + exp_name
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    tb_writer = tf.summary.create_file_writer(exp_dir)

    env = normalize(config['env'](reset_every_episode=True))

    dynamics_model = config['dynmodel'](
        name="dyn_model",
        env=env,
        learning_rate=config['learning_rate'],
        hidden_sizes=config['hidden_sizes'],
        valid_split_ratio=config['valid_split_ratio'],
        rolling_average_persitency=config['rolling_average_persitency'],
        hidden_nonlinearity=config['hidden_nonlinearity'],
        batch_size=config['batch_size'],
        extra=config.get('dynmodel_extra', None),
        max_data_size=config.get('max_data_size', None),
    )

    policy = MPCController(
        name="policy",
        env=env,
        dynamics_model=dynamics_model,
        discount=config['discount'],
        n_candidates=config['n_candidates'],
        horizon=config['horizon'],
        use_cem=config['use_cem'],
        num_cem_iters=config['num_cem_iters'],
    )

    sampler = Sampler(
        env=env,
        policy=policy,
        num_rollouts=config['num_rollouts'],
        max_path_length=config['max_path_length'],
        n_parallel=config['n_parallel'],
        adapt_batch_size=config.get('adapt_batch_size', None),
        enc_hyper=config.get('enc_hyper', False),
    )

    sample_processor = ModelSampleProcessor(recurrent=config.get('recurrent_processor', False))

    algo = Trainer(
        env=env,
        policy=policy,
        dynamics_model=dynamics_model,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        tb_writer=tb_writer,
        initial_random_samples=config['initial_random_samples'],
        dynamics_model_max_epochs=config['dynamic_model_epochs'],
    )
    algo.train()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--en', dest='exp_name',
                        type=str)
    return parser.parse_args()


if __name__ == '__main__':
    configs = {}

    configs['basic'] = {
            # Environment
            'env': HalfCheetahHFieldEnv,
            'dynmodel': HyperDynamicsModel,
            'dynmodel_extra': {'init_std': 0.1,
                                'adapt_batch_size': 16,
                                'enc_out_dim': 1,
                                'train_dec': False,
                                'hidden_sizes': (128, 128)
                                },

            # Policy
            'n_candidates': 500,
            'horizon': 20,
            'use_cem': False,
            'num_cem_iters': 5,
            'discount': 1.,

            # Sampling
            'max_path_length': 500,
            'num_rollouts': 10,
            'initial_random_samples': True,
            'recurrent_processor': True,
            'enc_hyper': True,

            # Training
            'max_data_size': 100,
            'n_itr': 500,
            'learning_rate': 1e-3,
            'batch_size': 128,
            'dynamic_model_epochs': 100,
            'valid_split_ratio': 0.1,
            'rolling_average_persitency': 0.99,

            # Dynamics Model
            'hidden_sizes': (128, 128),
            'hidden_nonlinearity': 'relu',
            'adapt_batch_size': 16,

            #  Other
            'n_parallel': 5,
            }
    args = parse_arguments()
    run_experiment(configs[args.exp_name], args.exp_name)