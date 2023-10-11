import argparse 
import logging 
from pathlib import Path 
import random 
from time import perf_counter

import numpy.random as npr
import torch 

from custom_tensorboard import SummaryWriter
from nnets import PartiallyConvexNetConfig, PartiallyConvexNet
from utils import LOG_DIRECTORY, get_now_str

Tensor: type = torch.Tensor

parser = argparse.ArgumentParser()
parser.add_argument("--enable-tb", action="store_true", help="Enable Tensorboard logging.")
parser.add_argument("--deterministic", action="store_true", help="Run in deterministic mode.")
parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs to execute during optimization.")
parser.add_argument("--step-size", type=float, default=1e-05, help="Gradient step size.")
parser.add_argument("--report-every", type=int, default=10, help="Interval with which to print the current value of the objective.")

def objective(predicted: Tensor, target: Tensor) -> Tensor: 
    residual: Tensor = predicted - target
    return torch.pow(residual, 2).sum()

def main(args): 
    # configure logging 
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    log.info(f"Deterministic mode: {'ENABLED' if args.deterministic else 'DISABLED'}")
    log.info(f"Tensorboard logging: {'ENABLED' if args.enable_tb else 'DISABLED'}")
    log.info(f"CUDA Available: {'TRUE' if torch.cuda.is_available() else 'FALSE'}")

    if torch.cuda.is_available(): 
        log.info(f"GPU Count: {torch.cuda.device_count()}")
        log.info(f"Current Device: {torch.cuda.get_device_name(0)}")
        device: torch.cuda.device = torch.device("cuda:0")
    else: 
        device = torch.device("cpu")

    if args.enable_tb: 
        writer: SummaryWriter = SummaryWriter(Path(LOG_DIRECTORY) / get_now_str())

    # PRNG 
    if args.deterministic: 
        torch.manual_seed(0)
        random.seed(0)
        npr.seed(0)
        torch.use_deterministic_algorithms(True)

    # sample uniform random variates in the unit cube 
    num_inputs: int = 100 
    inputs: Tensor = torch.rand(*(num_inputs, 2)).to(device) 
    targets: Tensor = torch.full((num_inputs, 1), 0.5).to(device)
    log.info(f"Generated {num_inputs} inputs")

    # setup the network 
    convex_input_size: int = 1 
    nonconvex_input_size: int = 1
    output_size: int = 1

    convex_layer_sizes = [3, 3, output_size] 
    nonconvex_layer_sizes = [nonconvex_input_size, 3, 3, output_size]

    config = PartiallyConvexNetConfig(
            convex_layer_sizes, 
            nonconvex_layer_sizes, 
            convex_input_size, 
            )
    net = PartiallyConvexNet(config).to(device)
    vnet = torch.func.vmap(lambda x, y: net(x, y))
    log.info(f"Configured network")

    # optimization 
    optimizer = torch.optim.SGD(net.parameters(), lr=args.step_size, momentum=0.9) 

    for i in range(args.num_epochs): 
        optimizer.zero_grad()
        outputs: Tensor = vnet(inputs[:, 0][:, None], inputs[:, 1][:, None]) 
        objective_value: Tensor = objective(outputs, targets)
        objective_value.backward()

        # telemetry 
        if args.enable_tb: 
            writer.scalar("Objective", objective_value.item(), step=i)

        optimizer.step()

        if i % args.report_every == 0: 
            log.info(f"Epoch [{i:03d}/{args.num_epochs:03d}] Objective: {objective_value.item():0.4f}")

    if args.enable_tb: 
        writer.close()

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
