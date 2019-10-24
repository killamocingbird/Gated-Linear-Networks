import torch
import layer as L

class GMN():
    def __init__(self, init, num_nodes, input_size, num_contexts):
        self.layers = [L.GMN_layer(init, num_nodes[0], input_size, num_contexts)]
        self.layers = self.layers + [L.GMN_layer(num_nodes[i - 1], num_nodes[i], input_size, num_contexts) for i in range(1, len(num_nodes))]
    
    def train_on_sample(self, z, target, learning_rate):
        z = z.view(-1)
        for i in range(len(self.layers)):
            if i == 0:
                forward = self.layers[i].forward(z, None)
                self.layers[i].backward(forward, target, learning_rate)
            else:
                p = torch.cat([forward[i][0].unsqueeze(0) for i in range(self.layers[i].in_features)])
                forward = self.layers[i].forward(z, p)
                loss = self.layers[i].backward(forward, target, learning_rate)
        return loss
    
    def infer(self, z):
        z = z.view(-1)
        for i in range(len(self.layers)):
            if i == 0:
                forward = self.layers[i].forward(z, None)
            else:
                p = torch.cat([forward[i][0].unsqueeze(0) for i in range(self.layers[i].in_features)])
                forward = self.layers[i].forward(z, p)
        return forward[0][0]
            
        
        