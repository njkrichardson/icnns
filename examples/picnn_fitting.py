import argparse 
from pathlib import Path 

import torch 

from custom_tensorboard import SummaryWriter
from nnets import PartiallyConvexNetConfig, PartiallyConvexNet
from utils import LOG_DIRECTORY, get_now_str

parser = argparse.ArgumentParser()
parser.add_argument("--enable-tb", action="store_true", help="Enable Tensorboard logging.")


def main(args): 
    # configure logging 
    if args.enable_tb: 
        writer: SummaryWriter = SummaryWriter(Path(LOG_DIRECTORY) / get_now_str())

    # setup the network 
    convex_input_size: int = 1 
    nonconvex_input_size: int = 1
    output_size: int = 1
    convex_layer_sizes = [5, 6, output_size] 
    nonconvex_layer_sizes = [nonconvex_input_size, 5, 6, output_size]

    config = PartiallyConvexNetConfig(
            convex_layer_sizes, 
            nonconvex_layer_sizes, 
            convex_input_size, 
            )
    net = PartiallyConvexNet(config)
    nonconvex_input = torch.ones(nonconvex_input_size) 
    convex_input = torch.ones(convex_input_size) 
    out = net(nonconvex_input, convex_input)

    if args.enable_tb: 
        writer.close()

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
