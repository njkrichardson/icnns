from typing import Tuple 

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt 
import numpy as np
import torch 
from torch.func import vmap

Tensor: type = torch.Tensor

def contour_plot(f: callable, range: Tuple[float, float], **kwargs): 
    num_evaluations: int = kwargs.get("num_fn_evaluations", 100)
    x: np.ndarray = np.linspace(*range, num=num_evaluations)
    y_range: Tuple[float, float] = range if kwargs.get("y_range", None) is None else kwargs["y_range"]
    y: np.ndarray = np.linspace(*y_range, num=num_evaluations)
    X, Y = np.meshgrid(x, y)

    inputs: np.ndarray = np.dstack((X, Y)).reshape(-1, 2)
    inputs: Tensor = torch.from_numpy(inputs) 
    f_inputs: np.ndarray = np.reshape((vmap(f)(inputs[:, 0][:, None], inputs[:, 1][:, None]), (num_evaluations, num_evaluations)).numpy())

    if not kwargs.get("ax", False): 
        figure, ax = plt.subplots(nrows=1, ncols=1, dpi=128) 
    else: 
        ax = kwargs["ax"] 

    ax.contour(x, y, f_inputs, cmap="Greys_r")

    if kwargs.get("ax", False): 
        return ax 

    if kwargs.get("save_path", None) is not None: 
        plt.savefig(kwargs.get("save_path"))
        plt.close()
    else: 
        return figure 

def surface_plot(f: callable, range: Tuple[float, float], **kwargs): 
    num_evaluations: int = kwargs.get("num_fn_evaluations", 100)
    device = kwargs.get("device", torch.device("cpu"))
    figure = plt.figure(dpi=128)
    ax = figure.add_subplot(111, projection="3d")

    x: np.ndarray = np.linspace(*range, num=num_evaluations)
    y_range: Tuple[float, float] = range if kwargs.get("y_range", None) is None else kwargs["y_range"]
    y: np.ndarray = np.linspace(*y_range, num=num_evaluations)
    X, Y = np.meshgrid(x, y)
    inputs: np.ndarray = np.dstack((X, Y)).reshape(-1, 2)
    inputs: Tensor = torch.from_numpy(inputs).to(device)
    inputs = inputs.type(kwargs.get("dtype", torch.float64))
    with torch.no_grad():
        f_inputs: np.ndarray = np.reshape((vmap(f)(inputs[:, 0][:, None], inputs[:, 1][:, None])).cpu().numpy(), (num_evaluations, num_evaluations))

    if kwargs.get("add_scatter", False): 
        x, y = kwargs["inputs"]
        with torch.no_grad():
            f_scatter = (vmap(f)(x, y)).cpu().numpy()
        ax.scatter(x.cpu().numpy(), y.cpu().numpy(), f_scatter)

    ax.plot_surface(X, Y, f_inputs, cmap="autumn_r", lw=0.5, rstride=1, cstride=1)
    ax.contour(X, Y, f_inputs, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=-1)
    ax.contour(X, Y, f_inputs, 10, lw=3, colors="k", linestyles="solid")

    if kwargs.get("save_path", None) is not None: 
        plt.savefig(kwargs.get("save_path"))
        plt.close()
    else: 
        return figure 
