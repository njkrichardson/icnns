import argparse 
import logging 
from pathlib import Path 
import random 
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np 
import numpy.random as npr
import torch 

from custom_tensorboard import SummaryWriter, image_from_figure
from plotting import surface_plot
from nnets import PartiallyConvexNetConfig, PartiallyConvexNet
from utils import LOG_DIRECTORY, get_now_str

Tensor: type = torch.Tensor
torch.set_default_dtype(torch.float32) 

parser = argparse.ArgumentParser()
parser.add_argument("--enable-tb", action="store_true", help="Enable Tensorboard logging.")
parser.add_argument("--deterministic", action="store_true", help="Run in deterministic mode.")
parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs to execute during optimization.")
parser.add_argument("--step-size", type=float, default=1e-05, help="Gradient step size.")
parser.add_argument("--report-every", type=int, default=10, help="Interval with which to print the current value of the objective.")
parser.add_argument("--plot-every", type=int, default=25, help="Interval with which to plot the current network surface.")

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
    targets: Tensor = torch.pow(inputs - Tensor([0.5, 0.5]).to(device), 2).sum(1)
    log.info(f"Generated {num_inputs} inputs")

    # TODO plot the target data and objective 
    target_figure = surface_plot(lambda x, y: objective(torch.hstack((x, y)), Tensor([0.5, 0.5]).to(device)), (0, 1), device=targets.device, add_scatter=True, inputs=(inputs[:, 0][:, None], inputs[:, 1][:, None]))
    target_image = image_from_figure(target_figure)
    writer.image("Objective surface", target_image)

    # setup the network 
    convex_input_size: int = 1 
    nonconvex_input_size: int = 1
    output_size: int = 1

    convex_layer_sizes = [3, 3, 8, 8, output_size] 
    nonconvex_layer_sizes = [nonconvex_input_size, 3, 3, 8, 8, output_size]

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

        if args.enable_tb: 
            writer.scalar("Objective", objective_value.item(), step=i)

            if i % args.plot_every == 0: 
                network_figure = surface_plot(lambda x, y: net(x, y), (0, 1), device=targets.device, dtype=inputs.dtype)
                network_image = image_from_figure(network_figure)
                writer.image("Network surface", network_image, step=i)
                plt.close()

                for name, param in net.named_parameters(): 
                    if param.requires_grad: 
                        data = param.data.detach().cpu().numpy()
                        writer.histogram(name, data, 10, step=i)
                        grad_data = param.grad.data.detach().cpu().numpy()
                        writer.histogram(f"{name}_grad", grad_data, 10, step=i)
                        rms: callable = lambda x: np.linalg.norm(x.ravel()) / np.sqrt(x.size)
                        writer.scalar(f"rms({name}_grad)", rms(grad_data), step=i)




            # TODO derivatives 
            #   rms(grad) 
            #   proportional rms(grad) (pie-chart) 
            # TODO parameters  
            # TODO network surface 
            # TODO network transfer characteristic Jacobian norm 
            # TODO network transfer chatacteristic Hessian norm 

        optimizer.step()

        if i % args.report_every == 0: 
            log.info(f"Epoch [{i:03d}/{args.num_epochs:03d}] Objective: {objective_value.item():0.4f}")

    if args.enable_tb: 
        writer.close()

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
