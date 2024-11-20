import torch
from SOFTDTW.soft_dtw_cuda import SoftDTW

class SoftDTWBarycenter(torch.nn.Module):
    def __init__(self, X, weights, gamma):
        super(SoftDTWBarycenter, self).__init__()
        self.X = X  # List of time series tensors
        self.weights = weights  # Weights for each time series
        self.gamma = gamma
        # Initialize the barycenter as a torch parameter
        self.Z = torch.nn.Parameter(torch.mean(torch.stack(X), dim=0))
    
    def forward(self):
        loss = 0
        for x_i, w_i in zip(self.X, self.weights):
            dist_func = self._softdtw_distance(x_i, self.Z)
            loss += w_i * dist_func
        return loss

    def _softdtw_distance(self, x, y):
        # Use your SoftDTW implementation here
        sdtw = SoftDTW(use_cuda=True, gamma=self.gamma)
        return sdtw(x.unsqueeze(0), y.unsqueeze(0))