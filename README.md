# Constrained Policy Optimization (CPO) implementation in PyTorch

This is PyTorch implementation of Constrained Policy Optimization (CPO) [[ArXiv]](https://arxiv.org/abs/1705.10528).

If you use this code in your research project please cite us as:
```
@software{Sikchi_pytorch-CPO,
author = {Sikchi, Harshit},
doi = {10.5281/zenodo.5121027},
license = {MIT},
title = {{pytorch-CPO}},
url = {https://github.com/hari-sikchi/pytorch_CPO}
}
```

## Requirements

* pytorch
* safety_gym
* mpi4py


## Instructions
To train an CPO agent on the `pointgoal1` task run:
```
python cpo.py --env=Safexp-PointGoal1-v0 --cost_lim=<cost threshold> --exp_name=<exp path>
```
This will produce the exp_path folder, where all the outputs are going to be stored. 

