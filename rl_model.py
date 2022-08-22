import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Qnet(nn.Module):
    def __init__(self, input_len, hidden_layer_len, output_len):
        super().__init__()
        self.linear_layer1 = nn.Linear(input_len, hidden_layer_len) # for linear transformation from input to hidden layer
        self.linear_layer2 = nn.Linear(hidden_layer_len,output_len) # for linear transformation from hidden layer to output layer
        
    
    def forward(self,input):
        input = F.relu(self.linear_layer1(input))
        output = self.linear_layer2(input) # try softmax
        return output
    
    def save(self, file_name="model.pth"):
        folder_path = "./model"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        file_name = os.path.join(folder_path,file_name)
        
        torch.save(self.state_dict, file_name)
        
        
class Q_training:
    def __init__(self, model:Qnet,lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr) # SGD 
        self.loss = nn.MSELoss() # (Qt-Q)**2
        
        
        
    def train_step(self,state,action, reward, new_state, isOver):
        state = torch.tensor(state, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            isOver = (isOver, )
            
            
        prediction = self.model(state)
        