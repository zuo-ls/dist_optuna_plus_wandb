import torch.nn as nn

class LSTM_mnist(nn.Module):
    def __init__(self,input_size=28,hidden_size=64,num_layers=2):
        super(LSTM_mnist,self).__init__()
        self.lstm_layer = nn.LSTM(
            input_size = 28,
            hidden_size = 64,
            num_layers = 2,
            batch_first = True
        )
        self.out = nn.Linear(64,10)
    
    def forward(self,x):
        r_out,_ = self.lstm_layer(x,None)
        out = self.out(r_out[:,-1,:])
        return out