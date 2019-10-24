import torch
import node
import math

class GMN_layer():
    
    def __init__(self, in_features, num_nodes, side_info_size, num_contexts):
        self.in_features = in_features
        self.nodes = [node.GM_Node(in_features + 1, side_info_size, num_contexts) for i in range(num_nodes)]
        self.bias = math.e / (math.e + 1)
        
    def __call__(self, z, p):
        return self.forward(p, z)
    
    def forward(self, z, p):
        if not p is None:
            p_hat = torch.cat((torch.as_tensor([self.bias]), p))
        else:
            #Forward with random base probabilities
            p_hat = torch.cat((torch.as_tensor([self.bias]), 0.5 * torch.ones(self.in_features)))
        return [self.nodes[i].forward(p_hat, z) for i in range(len(self.nodes))]
    
    def backward(self, forward, target, learning_rate, hyper_cube_bound = 200):
        #forward is an array with each element being a tuple (output, p_hat, context)
        loss = []
        for i in range(len(self.nodes)):
            loss.append(self.nodes[i].backward(forward[i][0], target, forward[i][1], forward[i][2], learning_rate, hyper_cube_bound=hyper_cube_bound))
        return loss