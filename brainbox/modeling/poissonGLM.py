import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class PoissonGLM(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(PoissonGLM, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.linear(x)
        return out

    def fit(self, x, y, epochs, optimizer):
        # Set the model into training mode
        self.train()
        num_cells = y.shape[1]
        device = y.device

        #Initialization
        min_loss = torch.ones(num_cells, dtype=torch.float32, device=device)*1e9
        best_weight = torch.ones_like(self.linear.weight.data)
        best_bias = torch.ones_like(self.linear.bias.data)


        for epoch in tqdm(range(1, epochs+1), 'Epoch: ', leave=False):
            # Clear cummulated gradients in every epoch
            optimizer.zero_grad()
            # Get output from the model
            outputs = self(x)
            # Compute poisson negative log likelihood loss for each cell
            loss_cells = F.poisson_nll_loss(outputs, y, reduce=False)
            loss_cells = torch.sum(loss_cells, dim=0)
            # Update min loss, best weight/bias for each cell
            mask = loss_cells < min_loss
            min_loss[mask] = loss_cells[mask].detach() # Don't grad on this tensor
            best_weight[mask] = self.linear.weight.data[mask]
            best_bias[mask] = self.linear.bias.data[mask]

            # Compute average loss for grad
            loss = torch.sum(loss_cells)/num_cells
            # Get gradients for parameters
            loss.backward()
            # Update parameters
            optimizer.step()

            # if epoch % 200 == 0:
            #     print(f'Epoch: {epoch} Loss: {loss.item()}')
        print(f'Training end, min loss: {torch.sum(min_loss)/num_cells}')

        return min_loss.cpu().numpy(), best_weight.cpu().numpy(), best_bias.cpu().numpy()