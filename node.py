import torch
import numpy as np
import torch.distributions as tdist
import sys

N = tdist.Normal(torch.tensor([0.0]), torch.tensor([0.1]))

class GM_Node():
    
    def __init__(self, in_features, input_size, num_contexts, init_weights=None):
        self.in_features = in_features
        self.num_contexts = num_contexts
        self.context_dim = 2**num_contexts
        if init_weights:
            if not in_features == len(init_weights):
                raise Exception
            else:
                self.w = init_weights
        else:
            self.w = torch.zeros(self.context_dim, in_features)
        
        self.context_vectors = [N.sample([input_size]).view(-1) for i in range(num_contexts)]
        self.context_biases = [0.0 for i in range(num_contexts)]
        
    def get_context(self, x):
        ret = 0
        for i in range(self.num_contexts):
            if torch.dot(x, self.context_vectors[i]) >= self.context_biases[i]:
                ret = ret + 2**i
        return ret
    
    #Geo_wc(z)(x_t = 1; p_t)
    def forward(self, p, z):
        context = self.get_context(z)
        return torch.sigmoid(torch.dot(self.w[context], GM_Node.logit(p))), p, context
        
    def backward(self, forward, target, p, context, learning_rate, hyper_cube_bound = 200):
        epsilon = 1e-6
        if target == 0:
            loss = -1 * torch.log(min(1 - forward + epsilon, torch.as_tensor(1 - epsilon)))
        else:
            loss = -1 * torch.log(min(forward + epsilon, torch.as_tensor(1 - epsilon)))
            
        if torch.isnan(loss):
            print(target, p, (p / (1 - p + 1e-6)))
            sys.exit()
        
        self.w[context] = GM_Node.clip(self.w[context] - learning_rate * (forward - target) * GM_Node.logit(p), hyper_cube_bound)
        return loss
    
    #Inverse sigmoid function on torch tensor
    def logit(x):
        return torch.log(x / (1 - x + 1e-6) + 1e-6)
        
    #Project x onto the hyper_cube_bound
    def clip(x, hyper_cube_bound):
        x[x > hyper_cube_bound] = hyper_cube_bound
        x[x < -1 * hyper_cube_bound] = -1 * hyper_cube_bound
        return x
    
    
        