import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=3,
            padding=0,
            bias=True
        )
        torch.nn.init.normal_(self.conv_1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.conv_1.bias, mean=0, std=0.01)
        self.conv_2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 39),
            stride=3,
            padding=0,
            bias=True
        )
        torch.nn.init.normal_(self.conv_2.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.conv_2.bias, mean=0, std=0.01)
        self.conv_3 = nn.Conv2d( # Fully connected layer
            in_channels=64,
            out_channels=256,
            kernel_size=(8, 1),
            bias=True
        )
        torch.nn.init.normal_(self.conv_3.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.conv_3.bias, mean=0, std=0.01)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        """Encoder forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input data. Shape: (batch_size, 1, time_points, num_features).
        
        Returns
        -------
        torch.Tensor
            Output data. Shape: (batch_size, embedding_dim).
        """
        x = self.conv_1(x)
        x = self.tanh(x)
        x = self.conv_2(x)
        x = self.tanh(x)
        x = self.conv_3(x)
        x = x.squeeze(-1).squeeze(-1)
        return x
