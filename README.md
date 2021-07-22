# Constrained Policy Optimization (CPO) implementation in PyTorch

This is PyTorch implementation of Constrained Policy Optimization (CPO) [[ArXiv]](https://arxiv.org/abs/1705.10528).

If you use this code in your research project please cite us as:
```
@article{harshit sikchi_2021, title={hari-sikchi/pytorch_CPO: CPO}, 
DOI={10.5281/zenodo.5121027}, 
abstractNote={<p>Pytorch implementation of CPO</p>}, publisher={Zenodo},
author={Harshit Sikchi}, year={2021}, month={Jul}}
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

