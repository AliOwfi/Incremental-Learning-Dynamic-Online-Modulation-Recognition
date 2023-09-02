import torch.nn as nn


class CNN1DClassifier(nn.Module):
    def __init__(self, n_way, indclude_head=True):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.include_head = indclude_head   

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3),
            #             nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3),

            nn.Flatten(),
            nn.Linear(in_features=6144, out_features=128, bias=True),
            nn.ReLU(),
            
        )

        if indclude_head:
            self.head = nn.Linear(in_features=128, out_features=n_way, bias=True)
            

    def forward(self, x, params=None):
        x = self.model(x)

        if self.include_head:
            x = self.head(x)

        return x


