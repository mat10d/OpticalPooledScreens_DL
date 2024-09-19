import math
import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_shape, x_dim, h_dim1, h_dim2, z_dim):
        """
        Basic encoding model.

        Parameters
        ----------
        input_shape: tuple
            shape of the input data in spatial dimensions (not channels)
        x_dim: int
            input channels in the input data
        h_dim1: int
            number of features in the first hidden layer
        h_dim2: int
            number of features in the second hidden layer
        z_dim: int
            number of latent features
        """
        super().__init__()
        # encoder part
        self.conv1 = nn.Conv2d(x_dim, h_dim1, kernel_size=3, stride=1, padding=1)
        # o = [(i + 2*p - k) / s] + 1
        output_shape = [(s + 2 * 1 - 3) + 1 for s in input_shape]
        self.conv2 = nn.Conv2d(h_dim1, h_dim2, kernel_size=3, stride=1, padding=1)
        self.output_shape = [(s + 2 * 1 - 3) + 1 for s in output_shape]
        # Computing the shape of the data at this point
        linear_h_dim = h_dim2 * math.prod(output_shape)
        self.fc31 = nn.Linear(linear_h_dim, z_dim)
        self.fc32 = nn.Linear(linear_h_dim, z_dim)

    def forward(self, x):
        """
        x: torch.Tensor
            input tensor

        Returns
        -------
        mu: torch.Tensor
            mean tensor
        log_var: torch.Tensor
            log variance tensor
        """
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        batch_size = h.size(0)
        h = h.view(batch_size, -1)
        return self.fc31(h), self.fc32(h)  # mu, log_var


class Decoder(nn.Module):
    def __init__(self, z_dim, h_dim1, h_dim2, x_dim, output_shape):
        """
        Basic decoding model

        Parameters
        ----------
        z_dim: int
            number of latent features
        h_dim1: int
            number of features in the first hidden layer
        h_dim2: int
            number of features in the second hidden layer
        x_dim: int
            number of output channels
        output_shape: tuple
            shape of the output data in the spatial dimensions
        """
        super().__init__()
        # decoder part
        self.z_spatial_shape = (h_dim1, *output_shape)
        spatial_shape = math.prod(self.z_spatial_shape)
        # "Upsample" the data back to the amount we need for the output shape
        self.fc = nn.Linear(z_dim, spatial_shape)
        # Here there will be a reshape
        self.conv1 = nn.Conv2d(h_dim1, h_dim2, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(h_dim2, x_dim, kernel_size=3, padding="same")

    def forward(self, z):
        z = F.relu(self.fc(z))
        h = z.view(-1, *self.z_spatial_shape)
        h = F.relu(self.conv1(h))
        return F.sigmoid(self.conv2(h))


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def check_shapes(self, data_shape, z_dim):
        with torch.no_grad():
            try:
                output, mu, var = self.forward(torch.zeros(data_shape))
                input_shape = data_shape
                assert (
                    output.shape == input_shape
                ), f"Output shape {output.shape} is not the same as input shape {input_shape}"
                assert (
                    mu.shape[-1] == z_dim
                ), f"Mu shape {mu.shape} is not the same as latent shape {z_dim}"
                assert (
                    var.shape[-1] == z_dim
                ), f"Var shape {var.shape} is not the same as latent shape {z_dim}"
                print("Model shapes are correct")
            except AssertionError as e:
                raise (e)
            except Exception as e:
                print("Error in checking shapes")
                raise (e)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z  # return z sample

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        return self.decoder(z), mu, log_var
