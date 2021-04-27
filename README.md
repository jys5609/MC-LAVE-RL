# Monte-Carlo Planning and Learning with Language Action Value Estimates

This repository is the implementation of ["Monte-Carlo Planning and Learning with Language Action Value Estimates"](https://openreview.net/pdf?id=7_G8JySGecm)

## Requirements

To install requirements:
```sh
conda env create -f environment.yml
conda activate mc-lave-rl
python -m spacy download en_core_web_lg
python -m spacy download en
```

## Monte-Carlo Tree Search (MCTS)
To run the data collection with MCTS for initial iteration, run this command with :
```sh
python run_planning.py --seed=$SEED --trial=0
```
(For each planning, use --seed option with different index)

All experiments in the paper were conducted with [Google Cloud Platform(GCP)](https://cloud.google.com/) and [HTCondor](https://research.cs.wisc.edu/htcondor/) to run in parallel. If you want to run the experiment in parallel, we reccomend to set up the HTCondor and environment on your GCP account.

## Merge planning trajectories and experience replays
To merge the data for learning the Q-network and policy, run this command: 
```sh
python merge_plans.py
```

## Train the Q-network and policy
To run the Q-network and policy with collected data, run this command:
```sh
(train Q-network) python run_learning.py --mode=0
(train policy) python run_learning.py --mode 1
```

## Monte-Carlo Planning with Language Action Value Estimates (MC-LAVE)
To run the MC-LAVE planning with trained Q-network and policy, run this command with :
```sh
python run_planning.py --seed=$SEED --trial=1
```
(For each iteration, use --trial option with iteration number)

## Citation
If this repository helps you in your academic research, you are encouraged to cite our paper. Here is an example bibtex:
```bibtex
@inproceedings{jang2021montecarlo,
  title={Monte-Carlo Planning and Learning with Language Action Value Estimates},
  author={Youngsoo Jang and Seokin Seo and Jongmin Lee and Kee-Eung Kim},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=7_G8JySGecm}
}
```
## Acknowledgement
This code is adapted and modified upon the code [github](https://github.com/microsoft/jericho) of AAAI 2020 paper ["Interactive Fiction Games: A Colossal Adventure"](https://arxiv.org/pdf/1909.05398.pdf). We appreciate their released dataset and code which are very helpful to our research.
