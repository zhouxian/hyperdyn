# hyperdyn
This repo contains code for the paper [HyperDynamics: Meta-Learning Object and Agent Dynamics with Hypernetworks](https://arxiv.org/abs/2103.09439) (ICLR 2021). This repo borrows code structure from [learning_to_adapt](https://github.com/iclavera/learning_to_adapt).

Project site: [zhouxian.github.io/hyperdynamics](https://zhouxian.github.io/hyperdynamics) 

If you use this code, please cite this paper:

    @inproceedings{xian2021hyperdynamics,
      author    = {Zhou Xian and
                   Shamit Lal and
                   Hsiao{-}Yu Tung and
                   Emmanouil Antonios Platanios and
                   Katerina Fragkiadaki},
      title     = {HyperDynamics: Meta-Learning Object and Agent Dynamics with Hypernetworks},
      booktitle = {9th International Conference on Learning Representations, {ICLR} 2021,
                   Virtual Event, Austria, May 3-7, 2021},
      year      = {2021},
    }

### Environment
Download and install [mujoco (version 131)](https://roboti.us/download.html).  
Set up conda env and install dependencies:
```
conda env create -f environment.yml
conda activate hyperdyn
. source_this.sh
```

### Training
Training on the task Cheetah-Slop with tensorboard logging:
```
# HyperDynamics
python scripts/train_hyperdyn.py --en basic
# MAML baseline
python scripts/train_maml.py --en basic
```
Tensorboard log files are saved under `log/{model}/{en}`.

