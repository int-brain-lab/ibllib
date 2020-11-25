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

    def fit(self, x, y, epochs, optim, lr):
        # Set the model into training mode
        self.train()
        num_cells = y.shape[1]
        device = y.device

        # Initialization
        min_loss = torch.ones(num_cells, dtype=torch.float32, device=device) * 1e9
        best_weight = torch.ones_like(self.linear.weight.data)
        best_bias = torch.ones_like(self.linear.bias.data)

        if optim == 'adam':
            # Use Adam to fit the model
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            # Use LBFGS to fit the model
            optimizer = torch.optim.LBFGS(self.parameters(), lr=lr)

        for epoch in tqdm(range(1, epochs + 1), 'Epoch: ', leave=False):
            if optim == 'adam':
                # Clear cummulated gradients in every epoch
                optimizer.zero_grad()
                # Get output from the model
                outputs = self(x)
                # Compute poisson negative log likelihood loss for each cell
                loss_cells = F.poisson_nll_loss(outputs, y, reduction='none')
                loss_cells = torch.sum(loss_cells, dim=0)

                # Compute average loss for grad
                loss = torch.sum(loss_cells) / num_cells
                # Get gradients for parameters
                loss.backward()
                # Update parameters
                optimizer.step()
            else:
                def closure():
                    if torch.is_grad_enabled():
                        # Clear cummulated gradients in every epoch
                        optimizer.zero_grad()
                    # Get output from the model
                    outputs = self(x)
                    # Compute average loss for grad
                    loss = F.poisson_nll_loss(outputs, y)
                    if loss.requires_grad:
                        # Get gradients for parameters
                        loss.backward()
                    return loss
                # Update parameters
                optimizer.step(closure)

                # Get output from the model
                outputs = self(x)
                # Compute poisson negative log likelihood loss for each cell
                loss_cells = F.poisson_nll_loss(outputs, y, reduction='none')
                loss_cells = torch.sum(loss_cells, dim=0)
                loss = torch.sum(loss_cells) / num_cells

            # Update min loss, best weight/bias for each cell
            mask = loss_cells < min_loss
            min_loss[mask] = loss_cells[mask].detach()  # Don't grad on this tensor
            best_weight[mask] = self.linear.weight.data[mask]
            best_bias[mask] = self.linear.bias.data[mask]

            # if epoch % 200 == 0:
            #     print(f'Epoch: {epoch} Loss: {loss.item()}')
        # print(f'Training end, min loss: {torch.sum(min_loss)/num_cells}')

        return min_loss.cpu().numpy(), best_weight.cpu().numpy(), best_bias.cpu().numpy()
