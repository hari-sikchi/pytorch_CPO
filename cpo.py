import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import  core
import sys
import safety_gym
import torch.nn.functional as F
from torch.autograd import Variable
from ppo_utils.logx import EpochLogger
from ppo_utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from ppo_utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import copy
from trust_region_utils import *
from replay_buffers import *
import os

def cpo(env_fn, env_name = '', actor_critic=core.MLPActorCriticCost, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, jf_lr=1e-3, penalty_init=1., penalty_lr=5e-2, cost_lim=25, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, target_l2=0.012, logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    # if not os.path.exists('safe_initial_policies/'+env_name+'.pt'):
    #     ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    # else:
    # Commented for LOOP experiments
    # if 'Grid' in env_name: 
    #     ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    # else:
    #     ac = torch.load('safe_initial_policies/'+env_name+'.pt')
    
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    # # print("Looking for safe policy at: {}".format())
    # if not os.path.exists('safe_initial_policies/'+env_name+'.pt'):
    #     raise NotImplementedError('A safe initial policy was not found at {}'.format('safe_initial_policies/'+env_name+'.pt'))
    # ac.load_state_dict(ac.state_dict(),'safe_initial_policies/'+env_name+'.pt')
    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.Qv1])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    # td3_buf = TD3Buffer(obs_dim[0], act_dim[0])

    # Set up penalty params
    soft_penalty = Variable(torch.exp(torch.Tensor([penalty_init]))-1, requires_grad=True)
    penalty_optimizer = torch.optim.Adam([soft_penalty],lr=penalty_lr)
    margin = 0
    margin_lr = 0.05
    learn_margin = False
    constraint_violations = [0]
    constraint_violations_count = [0]


    def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = Avp(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x


    def linesearch(model,
                f,
                get_kl,
                optim_case,
                c,
                x,
                fullstep,
                max_kl,
                max_backtracks=10,
                accept_ratio=.1):
        fval1, fval2 = f()[0].data,f()[1].data
        old_model = copy.deepcopy(model)
        # print("fval before", fval.item())
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            set_flat_params_to(model, xnew)
            newfval1, newfval2 = f()[0].data, f()[1].data
            kl_change = get_kl(old_model=old_model).mean()

            if (kl_change <= max_kl and
                (newfval1 <= fval1 if optim_case > 1 else True) and
                newfval2 - fval2 <= max(-c,0)):
                return True, xnew
        return False, x


    def trust_region_step(model, get_loss, get_kl, max_kl, damping):
        # global margin, margin_lr, learn_margin
        loss_pi, loss_j = get_loss()
        grads = torch.autograd.grad(loss_pi, model.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        grads_j = torch.autograd.grad(loss_j, model.parameters())
        loss_grad_j = torch.cat([grad.view(-1) for grad in grads_j]).data
        c = ac.epsilon/(gamma-1)
        rescale = 100
        # TODO correct this
        margin = 0
        margin_lr = 0.05
        learn_margin = False
        if learn_margin:
            margin+=margin_lr*c
            margin = max(0,margin)

        # TODO: mpi_Avg margin?

        c += margin
        c/= (rescale+1e-8)

        def Fvp(v):
            kl = get_kl()
            kl = kl.mean()

            grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, model.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + v * damping


        v = conjugate_gradients(Fvp, -loss_grad, 10)
        approx_g = Fvp(v)
        q = np.dot(v, approx_g)


        # Determine optim_case (switch condition for calculation,
        # based on geometry of constrained optimization problem)
        if np.dot(loss_grad_j,loss_grad_j) <= 1e-8 and c < 0:
            # feasible and cost grad is zero---shortcut to pure TRPO update!
            w, r, s, A, B = 0, 0, 0, 0, 0
            optim_case = 4
        else:
            # cost grad is nonzero: CPO update!
            w = conjugate_gradients(Fvp, -loss_grad_j, 10) 
            r = np.dot(w, approx_g)         # b^T H^{-1} g
            s = np.dot(w, Fvp(w))            # b^T H^{-1} b
            A = q - r**2 / s                # should be always positive (Cauchy-Shwarz)
            B = 2*max_kl - c**2 / s      # does safety boundary intersect trust region? (positive = yes)

            if c < 0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
            elif c < 0 and B >= 0:
                # x = 0 is feasible and safety boundary intersects
                # ==> most of trust region is feasible
                optim_case = 2
            elif c >= 0 and B >= 0:
                # x = 0 is infeasible and safety boundary intersects
                # ==> part of trust region is feasible, recovery possible
                optim_case = 1
                print("Alert! Attempting feasible recovery!")
                # self.logger.log('Alert! Attempting feasible recovery!', 'yellow')
            else:
                # x = 0 infeasible, and safety halfspace is outside trust region
                # ==> whole trust region is infeasible, try to fail gracefully
                optim_case = 0
                print("Alert! Attempting infeasible recovery!")
                # self.logger.log('Alert! Attempting infeasible recovery!', 'red')



        if optim_case in [3,4]:
            lam = np.sqrt(q / (2*max_kl))
            nu = 0
        elif optim_case in [1,2]:
            LA, LB = [0, r /c], [r/c, np.inf]
            LA, LB = (LA, LB) if c < 0 else (LB, LA)
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(np.sqrt(A/B), LA)
            lam_b = proj(np.sqrt(q/(2*max_kl)), LB)
            f_a = lambda lam : -0.5 * (A / (lam+1e-8) + B * lam) - r*c/(s+1e-8)
            f_b = lambda lam : -0.5 * (q / (lam+1e-8) + 2 * target_kl * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam * c - r) / (s + 1e-8)
        else:
            lam = 0
            nu = np.sqrt(2 * max_kl / (s+1e-8))

        # normal step if optim_case > 0, but for optim_case =0,
        # perform infeasible recovery: step to purely decrease cost
        x = (1./(lam+1e-8)) * (v + nu * w) if optim_case > 0 else nu * w

        if optim_case==0:
            logger.store(Backtrack=-1)
        else:
            logger.store(Backtrack=1)



        prev_params = get_flat_params_from(model)
        success, new_params = linesearch(model, get_loss, get_kl, optim_case, c, prev_params, x, max_kl)
        set_flat_params_to(model, new_params)

        return loss_pi        

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data, epoch_no=1):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        def get_kl(old_mean=None, new_mean=None, old_model=None):
            if old_mean is None:
                mean1 = ac.pi(obs)
            else:
                mean1 = old_mean
            if old_model is not None:
                mean1 = old_model(obs)


            log_std1, std1 = -2.99, 0.05
            if new_mean is None:
                mean0 = torch.autograd.Variable(mean1.data)
            else:
                mean0 = new_mean
            log_std0 = -2.99
            std0 = 0.05
            kl = log_std1 - log_std0 + (std0**2 + (mean0 - mean1).pow(2)) / (2.0 * std1**2) - 0.5
            return kl.sum(1, keepdim=True)

        def get_loss_pi():
            # if ac.epsilon<0:
            #     loss_pi =  (ac.Qj1(torch.cat((obs, ac.pi(obs)),dim=1))).mean()
            # else:
            #     loss_pi =  - (ac.Qv1(torch.cat((obs, ac.pi(obs)),dim=1))).mean()

            loss_pi = -(ac.Qv1(torch.cat((obs, ac.pi(obs)),dim=1))).mean()
            loss_j = (ac.Qj1(torch.cat((obs, ac.pi(obs)),dim=1))).mean()
            # loss_pi =  (ac.Qj1(torch.cat((obs, ac.pi(obs)),dim=1))).mean()
            return loss_pi, loss_j

        
        old_mean = ac.pi(obs).detach().data


        loss_pi = trust_region_step(ac.pi, get_loss_pi, get_kl, target_l2, 0.1)

        # Useful extra info
        approx_l2 = torch.sqrt(torch.mean((ac.pi(obs) - data['old_act'])**2)).item()
        
        approx_kl = get_kl(old_mean = old_mean, new_mean=ac.pi(obs).detach()).mean().item()

        # ent = pi.entropy().mean().item()
        ent = 0
        clipped = [0]
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl,l2=approx_l2, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, act, ret = data['obs'], data['act'], data['ret']
        return ((ac.Qv1(torch.cat((obs,act),dim=1)) - ret)**2).mean(), ((ac.Qv2(torch.cat((obs,act),dim=1)) - ret)**2).mean()


    # Set up function for computing value loss
    def compute_loss_j(data):
        obs, act, cost_ret = data['obs'], data['act'], data['cost_ret']
        return ((ac.Qj1(torch.cat((obs,act),dim=1)) - cost_ret)**2).mean(), ((ac.Qj2(torch.cat((obs,act),dim=1)) - cost_ret)**2).mean()



    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    pi_bc_optimizer = Adam(ac.pi.parameters(), lr=0.001)
    vf1_optimizer = Adam(ac.Qv1.parameters(), lr=vf_lr)
    vf2_optimizer = Adam(ac.Qv2.parameters(), lr=vf_lr)
    jf1_optimizer = Adam(ac.Qj1.parameters(), lr=jf_lr)
    jf2_optimizer = Adam(ac.Qj2.parameters(), lr=jf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)



    def update(epoch_no, constraint_violations, constraint_violations_count):
        # global soft_penalty, penalty_optimizer
        data = buf.get()

        # Update the penalty
        curr_cost = logger.get_stats('EpCostRet')[0]


        if curr_cost-cost_lim>0:
            # constraint_violations_count.append(1)
            # constraint_violations.append(curr_cost-cost_lim)
            logger.log('Warning! Safety constraint is already violated.', 'red')

        ac.epsilon = (1-gamma)*(cost_lim-curr_cost)
        if epoch_no==0 or ac.epsilon>=0:
            ac.baseline_pi = copy.deepcopy(ac.pi)
            ac.baseline_Qj = copy.deepcopy(ac.Qj1)
            
        
        # pi_l_old, pi_info_old = compute_loss_pi(data)
        # pi_l_old = pi_l_old.item()
        # v_l_old, _ = compute_loss_v(data)
        # v_l_old = v_l_old.item()
        # j_l_old, _ = compute_loss_j(data)
        # j_l_old = j_l_old.item()

        pi_l_old, v_l_old, j_l_old = 0, 0, 0
        pi_info_old = dict(kl=0,l2=0, ent=0, cf=0)

        if epoch_no==0:
           for i in range(train_v_iters):
            vf1_optimizer.zero_grad()
            vf2_optimizer.zero_grad()
            loss_v1, loss_v2 = compute_loss_v(data)
            loss_v1.backward()
            loss_v2.backward()
            mpi_avg_grads(ac.Qv1)    # average grads across MPI processes
            mpi_avg_grads(ac.Qv2)
            vf1_optimizer.step()
            vf2_optimizer.step()

            jf1_optimizer.zero_grad()
            jf2_optimizer.zero_grad()
            loss_j1, loss_j2 = compute_loss_j(data)
            loss_j1.backward()
            loss_j2.backward()
            mpi_avg_grads(ac.Qj1)    # average grads across MPI processes
            mpi_avg_grads(ac.Qj2)
            jf1_optimizer.step()
            jf2_optimizer.step() 


        # Trust region update for policy 
        loss_pi, pi_info = compute_loss_pi(data, epoch_no = epoch_no)


        logger.store(StopIter=0)

        # Value and Cost Value function learning
        for i in range(train_v_iters):
            vf1_optimizer.zero_grad()
            vf2_optimizer.zero_grad()
            loss_v1, loss_v2 = compute_loss_v(data)
            loss_v1.backward()
            loss_v2.backward()
            mpi_avg_grads(ac.Qv1)    # average grads across MPI processes
            mpi_avg_grads(ac.Qv2)
            vf1_optimizer.step()
            vf2_optimizer.step()

            jf1_optimizer.zero_grad()
            jf2_optimizer.zero_grad()
            loss_j1, loss_j2 = compute_loss_j(data)
            loss_j1.backward()
            loss_j2.backward()
            mpi_avg_grads(ac.Qj1)    # average grads across MPI processes
            mpi_avg_grads(ac.Qj2)
            jf1_optimizer.step()
            jf2_optimizer.step()



        # Log changes from update
        kl,l2, ent, cf = pi_info['kl'],pi_info['l2'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old, LossJ= j_l_old,
                     KL=kl, L2=l2, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v1.item() - v_l_old),
                     DeltaLossJ=(loss_j1.item() - j_l_old),
                     Penalty=torch.nn.functional.softplus(soft_penalty))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret,ep_cost_ret, ep_len = env.reset(), 0, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, j, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            noise = 0.05 * np.random.randn(*a.shape)
            # print("Action: {}".format(a+noise))
            a = a + noise
            next_o, r, d, info = env.step(a)
            ep_ret += r
            ep_cost_ret += info.get('cost', 0)
            ep_len += 1

            # save and log
            buf.store(o, a, r, info.get('cost', 0), v, j, logp, a)
            # td3_buf.add(o, a,next_o, r, info.get('cost', 0),d)
            logger.store(VVals=v, JVals = j)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, j, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v, j = 0, 0
                buf.finish_path(v, j)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpCostRet=ep_cost_ret, EpLen=ep_len)
                o, ep_ret , ep_cost_ret, ep_len = env.reset(), 0, 0, 0


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update(epoch, constraint_violations, constraint_violations_count)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCostRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('JVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('LossJ', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('DeltaLossJ', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        # logger.log_tabular('L2', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Penalty', average_only=True)
        logger.log_tabular('Backtrack', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
    print("Total constraint violations were: {} with total violation cost: {}".format(sum(constraint_violations_count), sum(constraint_violations)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_l2', type=float, default=0.012)
    parser.add_argument('--cost_lim', type=float, default=25)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=3)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--exp_name', type=str, default='td3_dump')
    
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from ppo_utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)


    env_fn = lambda : gym.make(args.env)
    cpo(env_fn, env_name= args.env, actor_critic=core.MLPActorCriticTD3trust,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, target_l2=args.target_l2,cost_lim=args.cost_lim,
        logger_kwargs=logger_kwargs)

