from dataclasses import dataclass, fields
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn 
from torchsummary import summary

class PositiveLinear(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int):
        super(PositiveLinear, self).__init__()
        self.input_dimension = input_dimension 
        self.output_dimension = output_dimension
        self.log_weights = nn.Parameter(torch.Tensor(output_dimension, input_dimension))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input, self.log_weights.exp())

class InputConvexLayer(nn.Module): 
    def __init__(self, input_dimension: int, output_dimension: int): 
        super(InputConvexLayer, self).__init__()

        self.input_dimension = input_dimension 
        self.output_dimension = output_dimension
        self.skip_linear: nn.Module = nn.Linear(output_dimension, input_dimension) 
        self.constrained_linear: nn.Module = PositiveLinear(input_dimension, output_dimension)

    def __repr__(self) -> str: 
        return f"{self.__class__.__name__}(input_dimension={self.input_dimension}, output_dimension={self.output_dimension})"

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor: 
        skip_out: torch.Tensor = self.skip_linear(x) 
        constrained_out: torch.Tensor = self.constrained_linear(z) 
        return skip_out + constrained_out

@dataclass 
class InputConvexNetConfig: 
    layer_sizes: Sequence[int] 
    activation: callable
    activate_last: Optional[bool]=False

class InputConvexNet(nn.Module): 
    def __init__(self, config: InputConvexNetConfig): 
        super(InputConvexNet, self).__init__()
        self.config = config 

        self.layer_sizes: Sequence[Tuple[int]] = list(zip(self.config.layer_sizes[1:-2], self.config.layer_sizes[1:-1]))
        self.input_layer: nn.Module = nn.Linear(self.config.layer_sizes[1], self.config.layer_sizes[0])
        self.internal_layers: Sequence[nn.Module] = [] 

        if self.layer_sizes: 
            for (input_dimension, output_dimension) in self.layer_sizes: 
                self.internal_layers.append(InputConvexLayer(input_dimension, output_dimension))

        self.output_layer: nn.Module = InputConvexLayer(self.config.layer_sizes[-2], self.config.layer_sizes[-1])

    def forward(self, x: torch.Tensor, z_init: Optional[torch.tensor]=None) -> torch.Tensor: 
        z: torch.Tensor = self.input_layer(x) 

        if self.internal_layers: 
            for layer in self.internal_layers: 
                z = self.config.activation(layer(z, x))

        z = self.output_layer(z, x)
        
        if self.config.activate_last: 
            z = self.config.activation(z)

        return z 

@dataclass 
class PartiallyConvexNetConfig: 
    convex_layer_sizes: Sequence[int] 
    nonconvex_layer_sizes: Sequence[int] 
    convex_input_size: int 
    convex_activation: Optional[callable]=nn.functional.relu
    nonconvex_activation: Optional[callable]=nn.functional.relu
    activate_last: Optional[bool]=False

class PartiallyConvexLayer(nn.Module): 
    def __init__(self, convex_dimensions: Tuple[int], nonconvex_dimensions: Tuple[int], convex_skip_dimension: int, **kwargs): 
        super(PartiallyConvexLayer, self).__init__()

        self.convex_dimensions = convex_dimensions
        self.nonconvex_dimensions = nonconvex_dimensions
        self.nonconvex_activation = kwargs.get("nonconvex_activation", nn.functional.relu) 

        # nonconvex signal path 
        self.nonconvex_output_dimension, self.nonconvex_input_dimension = nonconvex_dimensions
        self.nonconvex_weights = nn.Parameter(torch.Tensor(*nonconvex_dimensions))
        self.nonconvex_bias = nn.Parameter(torch.Tensor(self.nonconvex_output_dimension))

        # convex signal path 
        self.skip_dimension = convex_skip_dimension
        self.convex_output_dimension, self.convex_input_dimension = convex_dimensions
        self.log_weights = nn.Parameter(torch.Tensor(*convex_dimensions)) # W^{(z)}
        self.nonconvex_to_latent_weights = nn.Parameter(torch.Tensor(self.convex_input_dimension, self.nonconvex_output_dimension)) # W^{(zu)}
        self.convex_inner_bias = nn.Parameter(torch.Tensor(self.convex_input_dimension)) # b^{(z)}
        self.convex_weights = nn.Parameter(torch.Tensor(self.convex_output_dimension, convex_skip_dimension)) # W^{(y)}
        self.nonconvex_to_convex_weights = nn.Parameter(torch.Tensor(convex_skip_dimension, self.nonconvex_output_dimension)) # W^{(yu)}
        self.skip_bias = nn.Parameter(torch.Tensor(convex_skip_dimension)) # b^{(y)}
        self.nonconvex_latent_weights = nn.Parameter(torch.Tensor(self.convex_output_dimension, self.nonconvex_output_dimension)) # W^{(u)}
        self.convex_outer_bias = nn.Parameter(torch.Tensor(self.convex_output_dimension)) # b
        self.reset_parameters()

    def __repr__(self) -> str: 
        return f"{self.__class__.__name__}(convex_dimensions={self.convex_dimensions}, nonconvex_dimensions={self.nonconvex_dimensions})"

    def reset_parameters(self):
        for parameter in self.parameters(): 
            if parameter.dim() > 1: 
                nn.init.xavier_uniform(parameter)

    def forward(self, nonconvex_inputs: torch.Tensor, latent: torch.Tensor, convex_inputs: torch.Tensor) -> Tuple[torch.Tensor]: 
        u: torch.Tensor = self.nonconvex_activation(self.nonconvex_weights @ nonconvex_inputs + self.nonconvex_bias) 
        z: torch.Tensor = self.log_weights.exp() @ (latent * nn.functional.relu(self.nonconvex_to_latent_weights @ u + self.convex_inner_bias)) + \
                            self.convex_weights @ (convex_inputs * (self.nonconvex_to_convex_weights @ u + self.skip_bias)) + \
                            self.nonconvex_latent_weights @ u + self.convex_outer_bias
        return u, z 

class PartiallyConvexNet(nn.Module): 
    def __init__(self, config: PartiallyConvexNetConfig): 
        super(PartiallyConvexNet, self).__init__()
        self.config = config
        self.initialize()

    def initialize(self): 
        self.nonconvex_layer_spec = list(zip(self.config.nonconvex_layer_sizes[1:], self.config.nonconvex_layer_sizes[:-1]))
        self.config.convex_layer_sizes.insert(0, self.config.nonconvex_layer_sizes[0])
        self.convex_layer_spec = list(zip(self.config.convex_layer_sizes[1:], self.config.convex_layer_sizes[:-1]))
        self.layers = [PartiallyConvexLayer(convex_dimensions, nonconvex_dimensions, self.config.convex_input_size) for convex_dimensions, nonconvex_dimensions in zip(self.convex_layer_spec, self.nonconvex_layer_spec)]

    def forward(self, nonconvex_inputs: torch.Tensor, convex_inputs: torch.Tensor) -> torch.Tensor: 
        u: torch.Tensor = nonconvex_inputs
        z: torch.Tensor = torch.zeros_like(nonconvex_inputs)

        for layer in self.layers[:-1]: 
            u, z = layer(u, z, convex_inputs)
            z = self.config.convex_activation(z)

        u, z = self.layers[-1](u, z, convex_inputs)
        if self.config.activate_last: 
            z = self.config.convex_activation(z)
        return z 

convex_layer_sizes = [5, 6, 2] 
nonconvex_layer_sizes = [3, 5, 6, 2]
convex_input_size = 1 
config = PartiallyConvexNetConfig(convex_layer_sizes, nonconvex_layer_sizes, convex_input_size)
net = PartiallyConvexNet(config)
nonconvex_input = torch.ones(3) 
convex_input = torch.ones(1) 
out = net(nonconvex_input, convex_input)
breakpoint()
dummy: int = 5 
