# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
##### Soft Actor-Critic Algorithm implementation can be found at: https://github.com/pranz24/pytorch-soft-actor-critic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
          self.action_scale = torch.tensor(1.)
          self.action_bias = torch.tensor(0.)
        #else:
        #    self.action_scale = torch.FloatTensor(
        #        (action_space.high - action_space.low) / 2.)
        #    self.action_bias = torch.FloatTensor(
        #        (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.03)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
    
import math

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam


class SAC(object):
    def __init__(self, gamma, tau, alpha,target_update_interval, hidden_size, lear, num_inputs, action_space, policy = "Deterministic", automatic_entropy_tuning = False):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        #print(torch.cuda.is_available())
        #print(torch.version.cuda)
        if torch.cuda.is_available():  
          dev = "cuda:0" 
          print("GPU will be used")
        else:   
          dev = "cpu" 
          print("CPU will be used")
        #self.device = torch.device("cpu")
        self.device = torch.device(dev)

        self.critic = QNetwork(num_inputs, path_num, hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lear)

        self.critic_target = QNetwork(num_inputs, path_num, hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(path_num).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lear)

            self.policy = GaussianPolicy(num_inputs, path_num, hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lear)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, path_num, hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr= lear)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)  
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        #print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        #print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
#Create link capacities and find path capacities
import numpy as np
from itertools import combinations
from itertools import permutations
from scipy.special import factorial
N = 15
p = 0
#all_paths = np.zeros((N+2,N+2,int(factorial(N+1))));
all_paths = np.zeros((N+2,N+2,500));
for ind in range(0,N+1):
  if ind == 0:
    Nset = np.arange(1,N+2)
  else:
    Nset = np.arange(2,N+2)
  comb = combinations(Nset,ind)
  for node_col in list(comb):
    node_col = np.array(node_col)
    perm = permutations(node_col)

    for subpath in reversed(list(perm)):
      subpath = np.array(subpath)-1
      mask = np.zeros(((N+2)**2))
      tmp = [0]
      for i in range(0,subpath.shape[0]):
        tmp.append(subpath[i])
      tmp.append(N+1)
      path_ = np.array(tmp)
      indic = path_.shape[0]
      arr = (path_[1:indic],path_[0:indic-1])
      mask[np.ravel_multi_index(arr,dims = (N+2,N+2), order = 'F')] = 1
      all_paths[:,:,p] = np.reshape(mask,(N+2,N+2)).T;
      p = p+1
      if p == all_paths.shape[2]:
        break
    if p == all_paths.shape[2]:
        break
  if p == all_paths.shape[2]:
        break
print(p)
all_paths = all_paths[:,:,0:p];
num_paths = all_paths.shape[2]

#'''
#Find the approximate capacity
import numpy as np
import itertools as it
from scipy.optimize import linprog
import cvxpy as cp

def CreateLPSolver(N):
    ### Zero out all flows that shouldn't exist
    F_constraints_A = []
    F_constraints_b = []

    A_diag_zero = np.eye(N)

    F_constraints_A.append(A_diag_zero)
    F_constraints_b.append(0)

    for i in range(1,N-1):
        T = np.zeros((N,N))
        T[i,:] = 1
        T[:,i] = -1
        F_constraints_A.append(T)
        F_constraints_A.append(-T)
        F_constraints_b.append(0)
        F_constraints_b.append(0)

        
    L_constraints_A = []
    L_constraints_b = []

    ### Zero out all lambda values that shouldn't exist
    A_diag_zero = np.eye(N)

    L_constraints_A.append(A_diag_zero)
    L_constraints_b.append(0)

    ### Single node constraints
    for i in range(N):
        T = np.zeros((N,N))
        T[i,:] = 1
        L_constraints_A.append(T)
        L_constraints_b.append(1)

    for i in range(N):
        T = np.zeros((N,N))
        T[:,i] = 1
        L_constraints_A.append(T)
        L_constraints_b.append(1)

    ### Add flow variables to lambda constraints and vice versa
    num_L_constraints = len(L_constraints_A)
    num_F_constraints = len(F_constraints_A)
    L_A = np.stack(L_constraints_A).reshape((num_L_constraints,-1))
    L_A = np.hstack([0*L_A, L_A])
    L_b = np.stack(L_constraints_b)

    F_A = np.stack(F_constraints_A).reshape((num_F_constraints,-1))
    F_A = np.hstack([F_A, 0*F_A])
    F_b = np.stack(F_constraints_b)

    A = np.vstack([F_A, L_A, -np.eye(2*(N**2))])
    b = np.hstack([F_b, L_b, np.zeros(2*(N**2))])

    ### Create objective function
    c = np.zeros(N**2)
    c[-N:] = 1
    c = np.hstack([c, np.zeros(N**2)])
    
    # Define the solver function that is going to be retuned (will need to get LinC)
    def LPSolver(LinC, zero_out_unsued=True):
        
        Cap_A = np.hstack([np.eye(N**2), -np.diag(LinC.flatten())])
        Cap_b = np.zeros(N**2)
        
        A_ = np.vstack([A,Cap_A, -Cap_A])
        b_ = np.hstack([b,Cap_b, Cap_b])

        if zero_out_unsued:
            unused_mat = 1*(LinC == 0)
            unused = np.hstack([np.zeros(N**2), unused_mat.flatten()])
            A_ = np.vstack([A_, unused])
            b_ = np.hstack([b_, np.zeros(1)])
        

        res = linprog(-1*c, A_ub=A_, b_ub=b_, method='revised simplex')

        cap = -res.fun
        Lambdas = res.x[N**2:].reshape((N,N))
        Flows = res.x[:N**2].reshape((N,N))
        return cap, Lambdas, Flows
    
    return LPSolver


#'''

#'''
episode_lim = 200
path_num = 15

# Imports the weights assigned to each link in the paper
import pickle
with open('dist_larger','rb') as f:
  dist = pickle.load(f)

'''
# Creates random weights for the each link
dist = 250*np.random.rand(N+2,N+2)
dist = np.tril(dist,-1)
dist = dist + dist.T
dist[0,:] = 0
dist[:,N+1] = 0
'''

def block_edge(H, dist, bern_block, change):
    if change == True:
        bern_block = np.zeros((N+2,N+2))
    for i in range(0,N+2):
        for j in range(i+1,N+2):  
            if i != j:        
                nonblock_p = np.exp(-dist[i,j]/500)
                if np.isnan(nonblock_p) == True:
                    print('Nan blockage prob')
                elif nonblock_p > 1:
                    print('Nonblock p is greater than 1: ', nonblock_p)
                    print('Distance: ',dist[i,j])
                if change == True:
                    bern_block[i,j] = np.random.binomial(1,1-nonblock_p) 
                    bern_block[j,i] = copy(bern_block[i,j])
                else:
                    pass
                if bern_block[i,j] == 1:
                    H[i,j] = 1e-7
                    H[j,i] = 1e-7
    return H,bern_block

            
#'''

# Import the channels used in the paper
import pickle
with open('channels_larger','rb') as f:
  all_channels = pickle.load(f)

# Creates a new time-varying network
'''
all_channels = []
H = 10*np.random.rand(N+2,N+2)
H = np.tril(H,-1)
H = H + H.T
H[0,:] = 0
H[:,N+1] = 0
all_channels.append(H)
for i in range(0,episode_lim-1):
    H = H + 2*np.random.rand(N+2,N+2)-1
    if np.sum(H < 0) > 0:
        H[H < 0] = np.random.rand()
    H = np.tril(H,-1)
    H = H + H.T
    H[0,:] = 0
    H[:,N+1] = 0
    all_channels.append(H)

'''


###################################
import matplotlib
matplotlib.use('Agg')
import argparse
import datetime
import numpy as np
import itertools
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt
    
path_num = 15
count = 0
instance_num = 5
eps_rate = np.zeros((instance_num,episode_lim+1))
val_rate = np.zeros((instance_num,episode_lim+1))
inv_count = 0
avg_des = np.zeros((episode_lim+1))
benchmark_rates = np.zeros((instance_num,episode_lim+1))
benchmark_rates2 = np.zeros((instance_num,episode_lim+1))
H_placeholder = []
num_episodes = 5
count_lim = 500

#selected_paths = np.random.randint(0,num_paths,path_num) #Select the paths randomly
selected_paths = np.array([478 ,228, 256, 408, 173 ,141 ,304, 370 ,402 ,262, 50 ,139, 245, 46 ,191]) #Paths used in the paper

for instance_ind in range(0,instance_num):
    print("Instance: ", instance_ind)
    count_channel = 0
    gamma = 1
    tau = 0.005
    lr = 0.0003
    alpha = 0.2
    hidden_size = 256
    updates_per_step = 1
    target_update_interval = 1
    replay_size = 1000000
    seed = 123456
    decay = 1.1
    batch_size = 32
    count = 0       
    step_lim = 500

    # Agent
    agent = SAC(gamma, tau, alpha, target_update_interval, hidden_size,lr, path_num, None, policy = "Gaussian",automatic_entropy_tuning = True)

    # Memory
    memory = ReplayMemory(replay_size,seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    rewards = []
    avg_rewards = []
    found = False
    
    count_channel = 0

    # Find the path capacities
    H = all_channels[count_channel]
    count_channel += 1 
    path_cap = np.zeros((N+2,N+2,num_paths))
    for ind in range(0,num_paths):
        path_cap[:,:,ind] = np.multiply(all_paths[:,:,ind],H)
    path_cap[path_cap == 0] = 1e3
    path_cap = np.ndarray.min(np.ndarray.min(path_cap,axis = 0),axis = 0)
    path_cap[path_cap == 1e3] = 1e-7
 
    comp_paths = all_paths[:,:,selected_paths]
    sel_cap = path_cap[selected_paths]

    # Find the shortest path
    path_length = np.zeros((path_num))
    for i in range(0,path_num):
        for j in range(0,N+2):
            for k in range(0,N+2):
                if comp_paths[j,k,i] == 1:
                    path_length[i] = path_length[i]+dist[j,k]

    indices = np.argsort(path_length)                
    shortest_path = indices[0:2]
    print('Shortest path ind: ',shortest_path)
    print('Shortest path capacities: ',sel_cap[shortest_path])
    print('Rate of the first benchmark: ',np.sum(sel_cap[shortest_path])/2)
    benchmark_rates[instance_ind,1] = np.sum(sel_cap[shortest_path])/2
    benchmark_rates2[instance_ind,1] = np.sum(sel_cap)/path_num
    blockage = True
    blockage2 = False
    des_rate_count = 0
    blockage_time = 0
    solve_FD_cap = CreateLPSolver(N+2) 
    H_py = np.zeros((N+2,N+2))
    Lambdas = np.zeros((N+2,N+2))
    Flows = np.zeros((N+2,N+2))
    capacity_app, Lambda_ji, Flows1 = solve_FD_cap(H)
    des_rate = capacity_app*(0.5)
    
    avg_des[0] = avg_des[0]+des_rate
    print('desired rate: ',des_rate)
    blockage_applied = True
    variation = False
    H_old = copy(H)
    bern_block = np.zeros((N+2,N+2))
    blocked_paths = np.zeros((episode_lim))
    for episode in range(episode_lim):
        print("Episode: ",episode)
        if episode > 0:
            if instance_ind == 0:
                H = all_channels[count_channel] # Change link capacities
                if episode %10 == 0:
                    change = True
                else:
                    change = False
                H, bern_block = block_edge(H,dist,bern_block,change) # There is blockage 
                H = np.tril(H,-1)
                H = H + H.T
                H[0,:] = 0
                H[:,N+1] = 0
                
                H_placeholder.append(H)
            else:
                H = H_placeholder[episode-1]
         
            count_channel += 1 
            path_cap = np.zeros((N+2,N+2,num_paths))
            for ind in range(0,num_paths):             
                path_cap[:,:,ind] = np.multiply(all_paths[:,:,ind],H)
            path_cap[path_cap == 0] = 1e3
            path_cap = np.ndarray.min(np.ndarray.min(path_cap,axis = 0),axis = 0)
            path_cap[path_cap == 1e3] = 1e-7
        
            comp_paths = all_paths[:,:,selected_paths]
            sel_cap = path_cap[selected_paths]
            benchmark_rates[instance_ind,episode+1] = np.sum(sel_cap[shortest_path])/2
            benchmark_rates2[instance_ind,episode+1] = np.sum(sel_cap)/path_num
            print('Number of blocked paths: ',np.sum(sel_cap < 1e-3))
            blocked_paths[episode] = np.sum(sel_cap < 1e-3)
            capacity_app, Lambda_ji, Flows1 = solve_FD_cap(H)
            des_rate = capacity_app*0.5
            if des_rate < 0:
                des_rate = 0

        avg_des[episode+1] = avg_des[episode+1] + des_rate
        print('desired rate: ',des_rate)
        episode_reward = 0
        episode_steps = 0
        done = False

        state = np.zeros((path_num)) # Initial state
        count_invalid_state = 0
        for step in range(step_lim):
            action = agent.select_action(state)
            if np.isnan(np.sum(action)):
                print('Invalid action')
       
            
            action[action < 1e-3] = 0 # Action clipping
            temp_new_state = state + action
            act_times = np.divide(temp_new_state,sel_cap)
            reward = 0
            counted = False

            # Environment checks if the next state is valid
            done = False
            finish = False
            node_time_pre = np.zeros((N+2))
            node_time_fol = np.zeros((N+2))
            for node in range(0,N+2):
                for i in range(0,path_num):
                    if np.sum(comp_paths[node,:,i]) > 0 or np.sum(comp_paths[:,node,i]) > 0:
                        if node != 0:
                            ind_pre = np.argwhere(comp_paths[node,:,i] == 1)
                            node_time_pre[node] = node_time_pre[node] + act_times[i]*(sel_cap[i]/H[node,ind_pre])
                        if node != N+1:
                            ind_fol = np.argwhere(comp_paths[:,node,i] == 1)
                            node_time_fol[node] = node_time_fol[node] + act_times[i]*(sel_cap[i]/H[ind_fol,node])
            if np.sum(node_time_pre[1:N+2] > 1) > 0 or np.sum(node_time_fol[0:N+1] > 1) > 0 or np.sum(act_times < 0) > 0 or np.any(temp_new_state > sel_cap):
                new_state = copy(state) 
            else:
                new_state = copy(temp_new_state)

            eps_rate[instance_ind,episode+1] = eps_rate[instance_ind,episode+1] + np.sum(new_state)  
            if np.sum(new_state) >= des_rate:
                finish = True
                reward = 1
                eps_rate[instance_ind,episode+1] = (eps_rate[instance_ind,episode+1] + np.sum(new_state)*(step_lim-step-1))/step_lim
    
            if step == step_lim-1 and finish == False:
                eps_rate[instance_ind,episode+1] = eps_rate[instance_ind,episode+1]/step_lim
                
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
    
            memory.push(state, action, reward, new_state, done) # Append transition to memory
            if len(memory) > batch_size:
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, batch_size, updates)
                updates += 1
            state = copy(new_state)
        
            if finish:
                break

        
        avgval = np.zeros((path_num))
        lim_val = 500
        epsilon_valid = 0
        for valid_ind in range(0,num_episodes):
            valstate = np.zeros((path_num))
            for valid_step in range(0,lim_val):
                action = agent.select_action(valstate)
                action[action < 1e-3] = 0
                if np.isnan(np.sum(action)):
                    print('Invalid action')

                temp_new_state = valstate + action 
                act_times = np.divide(temp_new_state,sel_cap)
            
                # Environment checks if the next state is valid
                node_time_pre = np.zeros((N+2))
                node_time_fol = np.zeros((N+2))
                for node in range(0,N+2):
                    for i in range(0,path_num):
                        if np.sum(comp_paths[node,:,i]) > 0 or np.sum(comp_paths[:,node,i]) > 0:
                            if node != 0:
                                ind_pre = np.argwhere(comp_paths[node,:,i] == 1)
                                node_time_pre[node] = node_time_pre[node] + act_times[i]*(sel_cap[i]/H[node,ind_pre])
                            if node != N+1:
                                ind_fol = np.argwhere(comp_paths[:,node,i] == 1)
                                node_time_fol[node] = node_time_fol[node] + act_times[i]*(sel_cap[i]/H[ind_fol,node])
                if np.sum(node_time_pre[1:N+2] > 1) > 0 or np.sum(node_time_fol[0:N+1] > 1) > 0 or np.sum(act_times < 0) > 0 or np.any(temp_new_state > sel_cap):       
                    new_state = copy(valstate) ###eski
                else:
                    new_state = copy(temp_new_state)
         
                if np.sum(new_state) >= des_rate:
                    val_rate[instance_ind,episode+1] = val_rate[instance_ind,episode+1] + np.sum(new_state)
                    break

                if valid_step == lim_val-1:
                    val_rate[instance_ind,episode+1] = val_rate[instance_ind,episode+1] + np.sum(new_state)
                valstate = copy(new_state)
   
        val_rate[instance_ind,episode+1] = val_rate[instance_ind,episode+1]/num_episodes
  
        print("Validation rate: ",val_rate[instance_ind,episode+1])
        print("Training rate: ",eps_rate[instance_ind,episode+1])
        print("Benchmark rate: ",benchmark_rates[instance_ind,episode+1])
        print("Benchmark2 rate: ",benchmark_rates2[instance_ind,episode+1])
   
eps_rate2 = np.sum(eps_rate,0)/instance_num
val_rate2 = np.sum(val_rate,0)/instance_num
avg_des = avg_des/instance_num
benchmark = np.sum(benchmark_rates,0)/instance_num
benchmark2 = np.sum(benchmark_rates2,0)/instance_num


episode = np.linspace(0,episode_lim,episode_lim+1)
fig = plt.figure()
plt.plot(episode,eps_rate2, label = 'Average Training Rate')
plt.plot(episode,val_rate2, label = 'Evaluation Rate')
plt.plot(episode,benchmark, label = 'SP')
plt.plot(episode,benchmark2, label = 'ES')
plt.plot(episode,avg_des,label='Desired Rate')
plt.xlabel('Episodes')
plt.ylabel('Rate')
plt.legend()
fig.savefig('test.png')
plt.show()

episode = np.linspace(0,episode_lim-1,episode_lim)
fig = plt.figure()
plt.plot(episode,blocked_paths)
plt.xlabel('Episodes')
plt.ylabel('Number of blocked paths')
fig.savefig('blocked_paths.png')
plt.show()

