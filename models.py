import torch.nn as nn
import torch


class CNN1DClassifier(nn.Module):
    def __init__(self, n_way, indclude_head=True):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.include_head = include_head

        self.model_lst = [
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3),
            #             nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3),

            nn.Flatten(),
            nn.Linear(in_features=6144, out_features=128, bias=bias),
            # nn.ReLU(),
            
        ]

        if last_relu:
            self.model_lst.append(nn.ReLU())    
        
        self.model = nn.Sequential(*self.model_lst) 


        if indclude_head:
            self.head = nn.Linear(in_features=128, out_features=n_way, bias=True)
            

    def forward(self, x, params=None):
        x = self.model(x)
        
        if self.include_head:
            x = self.head(x)

        return x



class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))
    def forward(self, x):
        return self.alpha * x + self.beta
    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())




