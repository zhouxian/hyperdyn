from modules.dynamics.meta_mlp_dynamics import MetaMLPDynamicsModel
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


EXP_NAME = 'maml'


def run_experiment(config, exp_name):
    exp_dir = os.getcwd() + '/log/' + EXP_NAME + '/' + exp_name
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    tb_writer = tf.summary.create_file_writer(exp_dir)

    env = normalize(config['env'](reset_every_episode=True))

    dynamics_model = config['dynmodel'](
        name="dyn_model",
        env=env,
        meta_batch_size=config['meta_batch_size'],
        inner_learning_rate=config['inner_learning_rate'],
        learning_rate=config['learning_rate'],
        hidden_sizes=config['hidden_sizes'],
        valid_split_ratio=config['valid_split_ratio'],
        rolling_average_persitency=config['rolling_average_persitency'],
        hidden_nonlinearity=config['hidden_nonlinearity'],
        batch_size=config['adapt_batch_size'],
        max_data_size=config.get('max_data_size', None),
        num_steps_per_epoch_mult=config.get('num_steps_per_epoch_mult', None),
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
        n_parallel=config['n_parallel'],
        max_path_length=config['max_path_length'],
        num_rollouts=config['num_rollouts'],
        adapt_batch_size=config['adapt_batch_size'],  # Comment this out and it won't adapt during rollout
    )

    sample_processor = ModelSampleProcessor(recurrent=True)

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
            'dynmodel': MetaMLPDynamicsModel,

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

            # Training
            'max_data_size': 100,
            'n_itr': 500,
            'learning_rate': 1e-3,
            'meta_batch_size': 10,
            'dynamic_model_epochs': 100,
            'valid_split_ratio': 0.1,
            'rolling_average_persitency': 0.99,

            # Dynamics Model
            'hidden_sizes': (128, 128),
            'hidden_nonlinearity': 'relu',
            'inner_learning_rate': 0.001,
            'adapt_batch_size': 16,

            #  Other
            'n_parallel': 5,
    }
    args = parse_arguments()
    run_experiment(configs[args.exp_name], args.exp_name)