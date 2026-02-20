import numpy
import matplotlib.pyplot as plt
import sklearn
import torch
import logging

import time
import os
import sys
from pathlib import Path
from network import FCNN

class HarmonicPINN:
    def __init__(self, config_dict: dict):
        """
        Class to learn the damped harmonic oscillator
        """
        self.config = config_dict
        self.logger = self._setup_logger()
        self.logger.info("Initializing Trainer...")
        self.logger.info(f"Configs: {self.config}")

        ## sampling configs ##
        self.num_collocation = self.config.get('n_collocation', 5000)
        self.num_initial = self.config.get('n_initial', 1000)
        self.num_test = self.config.get('n_test', 1000)

        self.t_min = self.config.get('t_min', 0.)
        self.t_max = self.config.get('t_max', 1.)
        self.xi_min = self.config.get('xi_min', 0.1)
        self.xi_max = self.config.get('xi_max', 0.4)

        ## network ##
        self.input_dim = self.config.get('input_dim', 2)
        self.hidden_dim = self.config.get('hidden_dim', 64)
        self.hidden_layers = self.config.get('hidden_layers', 3)
        self.activation = self.config.get('activation', 'tanh')

        ## optimizer and training ##
        self.opt_name = self.config.get('optimizer', 'adam')
        self.lr = self.config.get('learning_rate', 1e-3)

        ## device ##
        device_cfg = self.config.get('device', 'auto')
        if device_cfg == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_cfg)


    def sample_collocation_points(self, num_samples):
        t = torch.rand(num_samples, device=self.device, requires_grad=True)
        xi = torch.rand(num_samples, device=self.device)

        return t, xi
    
    def setup(self):
        ## ODE residual calculation ##
        def ode_residual(input, model, x_0, v_0, t_max, xi_min, xi_max):

            t = input[:, 0:1]
            xi_norm = input[:, 1:2]
            xi = xi_min + (xi_max - xi_min) * xi_norm
            output = model(input)

            ## hard-coding initial values ##
            x = x_0 + v_0 * t + t**2 * output

            dx_t = torch.autograd.grad(x,
                                       t,
                                       grad_outputs=torch.ones_like(output),
                                       create_graph=True)[0]
            
            dx_tt = torch.autograd.grad(dx_t,
                                       t,
                                       grad_outputs=torch.ones_like(dx_t),
                                       create_graph=True)[0]
            
            dx_z = dx_t / t_max
            dx_zz = dx_tt / (t_max**2)

            residual = dx_zz + 2.0 * xi * dx_z + x

            return residual
        
        def compute_loss(residual):
            return torch.mean(residual**2)
        
        self.ode_residual = ode_residual
        self.loss_fn = compute_loss

        self.network = FCNN(self.input_dim, self.hidden_dim, self.hidden_layers, self.activation)

        if self.opt_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr = self.lr,
            )

    def train(self):
        pass




    def _setup_logger(self):
        log_dir = Path("outputs")
        log_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = self.config.get("run_name", "pinn_run")
        run_dir = log_dir / f"{run_name}_{timestamp}"
        run_dir.mkdir()

        inference_dir = run_dir / "inference_results"
        inference_dir.mkdir()

        # setup logger
        logger = logging.getLogger(str(self.run_dir))
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        log_file = self.run_dir / "train.log"

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        logger.info(f"Logging initialized. Log file: {log_file}")
        return logger