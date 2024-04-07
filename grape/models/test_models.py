from typing import Callable
import torch
from torch import nn, Tensor
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool

__all__ = [
    'SimpleGNN'
]

class SimpleGNN(nn.Module):
    """A simple model to assist testing model development. It takes a defined message passing algorith such as MPNN
    and adds a simple global pooling layer as well as an output layer. Made for continuous output such as mpC or
    logP.

    Parameters
    ------------
    model_message: nn.Module
        The message passing layers that will be used to generate the node embedding.
    input_pool_size: int
        The output dimension of the defined model, usually just the hidden dimension of that model.
    output_size: int
        The prediction dimension of the overall model. Default: 1

    """

    def __init__(self, model_message: nn.Module or nn.Sequential, input_pool_size: int, pool_layer: Callable = None,
                 output_size: int = 1,
                 ):

        super().__init__()
        self.model = model_message
        self.lin_out = Linear(in_features=input_pool_size, out_features=output_size)
        self.pool = global_mean_pool
    def forward(self, data, return_latents = False) -> tuple[Tensor] or tuple[Tensor, Tensor]:
        """
        Parameters
        ------------
        data: Data or DataLoader
            A singular graph Data object or a batch of graphs in the form of a DataLoader object.
        return_latents: bool
            Decides if the latent representation of the nodes from the Message Passing Network should be returned.
            Default: False

        Returns
        ---------
        tuple[Tensor] or tuple[Tensor,Tensor]
            The continuous variable prediction of the shape (batch_size) and the latent representation of the nodes.

        """

        nodes_out = self.model(data)
        nodes_out = self.pool(nodes_out, batch=data.batch)

        if return_latents:
            return self.lin_out(nodes_out).view(-1), nodes_out
        else:
            return self.lin_out(nodes_out).view(-1)

