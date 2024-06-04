from typing import Callable, Union
from torch import nn, Tensor
from torch.nn import Linear, ReLU
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data, DataLoader

__all__ = [
    'SimpleGNN'
]

class SimpleGNN(nn.Module):
    """A simple model to assist testing model development. It takes a defined message passing algorith such as MPNN
    and adds a simple global pooling layer as well as an output layer. Made for continuous output such as mpC or
    logP unless an output model is specified.

    # TODO: Extend to categorical target.

    Notes
    ------
    Either the output model ``out_model`` or the output feature size of the message passing
    network ``out_message_size`` has to be specified.

    Parameters
    ------------
    model_message: nn.Module or nn.Sequential
        The message passing layers that will be used to generate the node embedding.
    pool_layer: Callable
        A pooling layer from torch_geometric
        (https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#pooling-layers) such as global_max_pool.
        Defaults to global_mean_pool if not specified.
    out_model: nn.Module or nn.Sequential
        A model that specifies the output model of the GNN. Could be any model as long as it generates the correct
         output size. If not specified, it will default to a simple three layered MLP.
    out_message_size: int
        The feature dimension of the output from the message passing network. It is usually just the hidden dimension
        of that model.
    output_size: int
        The prediction dimension of the overall model. Default: 1
    out_hidden_feats: int
        If no output model is given, this parameter specifies the number of hidden features in the default three layered
        MLP. Default: 64

    """

    def __init__(self, model_message: Union[nn.Module, nn.Sequential], pool_layer: Callable = None,
                out_model: Union[nn.Module, nn.Sequential] = None, out_message_size: int = None,
                 output_size: int = 1, out_hidden_feats: int = 64):

        super().__init__()
        self.model_message = model_message

        assert out_model is not None or out_message_size is not None, ('Either the output model or '
                                                                            'the number of output features from the'
                                                                            ' message passing model has to be given.')
        if out_model is None:
            self.out = nn.Sequential(
                Linear(out_message_size, out_hidden_feats),
                ReLU(),
                Linear(out_hidden_feats, output_size)
            )
        else:
            self.out = out_model

        self.pool = pool_layer if pool_layer is not None else global_mean_pool

    def forward(self, data: Union[Data, DataLoader], return_latents = False) \
            -> Union[tuple[Tensor],tuple[Tensor, Tensor]]:
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

        nodes_out = self.model_message(data)
        nodes_out = self.pool(nodes_out, batch=data.batch)

        if return_latents:
            return self.out(nodes_out).view(-1), nodes_out
        else:
            return self.out(nodes_out).view(-1)