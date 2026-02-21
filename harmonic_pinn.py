import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sklearn
import torch
import logging

import time
import os
import sys
from pathlib import Path
from network import FCNN, FourierFCNN

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
        self.t_max = self.config.get('t_max', 20.)
        self.xi_min = self.config.get('xi_min', 0.1)
        self.xi_max = self.config.get('xi_max', 0.4)

        self.x_0 = self.config.get('x_0', 0.7)
        self.v_0 = self.config.get('v_0', 1.2)

        ## network ##
        self.net_type = self.config.get('network', 'fcnn')
        self.input_dim = self.config.get('input_dim', 2)
        self.hidden_dim = self.config.get('hidden_dim', 64)
        self.hidden_layers = self.config.get('hidden_layers', 3)
        self.activation = self.config.get('activation', 'tanh')
        if self.net_type == "fourier":
            self.num_frequencies = self.config.get('frequencies', 10)
            self.fourier_scale = self.config.get('scale', 5.0)

        ## optimizer and training ##
        self.optim = self.config.get('optimizer', {})
        self.lr = self.config.get('learning_rate', 1e-3)
        self.epochs = self.config.get('epochs', 20000)
        self.batch_size = self.config.get('batch_size', 1024)
        self.resample_period = self.config.get('resample_period', 1000)

        ## device ##
        device_cfg = self.config.get('device', 'auto')
        if device_cfg == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_cfg)

        self.setup()


    def sample_collocation_points(self, num_samples):
        t = torch.rand(num_samples, 1, device=self.device)
        xi = torch.rand(num_samples, 1, device=self.device)

        inputs = torch.cat([t, xi], dim=1)
        inputs.requires_grad_(True)

        return inputs
    
    def setup(self):
        ## ODE residual calculation ##
        def ode_residual(input, net, xi_min, xi_max, t_max):

            xi_norm = input[:, 1:2]
            xi = xi_min + (xi_max - xi_min) * xi_norm
            x = net(input)
            # print("t.requires_grad:", t.requires_grad)
            # print("x.requires_grad:", x.requires_grad)


            grads = torch.autograd.grad(x,
                                       input,
                                       grad_outputs=torch.ones_like(x),
                                       create_graph=True)[0]
            
            dx_t = grads[:, 0:1]
            
            grads2 = torch.autograd.grad(dx_t,
                                       input,
                                       grad_outputs=torch.ones_like(dx_t),
                                       create_graph=True)[0]
            
            dx_tt = grads2[:, 0:1]
            
            dx_z = dx_t / t_max
            dx_zz = dx_tt / (t_max**2)

            residual = (dx_zz + 2.0 * xi * dx_z + x)

            return residual
        
        def compute_loss(residual):
            return torch.mean(torch.pow(residual, 2))
        
        self.ode_residual = ode_residual
        self.loss_fn = compute_loss
        
        ## Setup networks ##

        if self.net_type == "fcnn":
            self.network = FCNN(self.input_dim, self.hidden_dim, self.hidden_layers, self.x_0, self.v_0, self.t_max, self.activation)
            self.network.to(self.device)
        elif self.net_type == "fourier":
            self.network = FourierFCNN(self.hidden_dim, self.hidden_layers, self.x_0, self.v_0, self.t_max, self.num_frequencies, self.fourier_scale, self.activation)
            self.netwrok.to(self.device)

        ## Setup optimizers ##

        self.use_adam = self.optim.get("adam", {}).get("use_adam", True)
        if self.use_adam:
            self.adam_lr = self.optim["adam"].get("lr", 1e-4)
            self.adam_epochs = self.optim["adam"].get("epochs", 5000)

            self.adam_optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.adam_lr
            )
        
        self.use_lbfgs = self.optim.get("lbfgs", {}).get("use_lbfgs", False)
        if self.use_lbfgs:
            self.lbfgs_lr = self.optim["lbfgs"].get("lr", 1.0)
            self.lbfgs_max_iter = self.optim["lbfgs"].get("max_iter", 1000)
            self.lbfgs_history = self.optim["lbfgs"].get("history_size", 100)

            self.lbfgs_optimizer = torch.optim.LBFGS(
                self.network.parameters(),
                lr=self.lbfgs_lr,
                max_iter=self.lbfgs_max_iter,
                history_size=self.lbfgs_history,
                line_search_fn="strong_wolfe"
            )


    def train(self):

        # beginning to train #
        # sample collocation points
        self.loss_history = []
        self.epoch = []
        epoch = 0
        x_0 = self.x_0
        v_0 = self.v_0
        t_max = self.t_max
        xi_min = self.xi_min
        xi_max = self.xi_max
        self.network.train()

        iter = 0
        
        # ---------------------------------
        # Stage 1: Adam
        # ---------------------------------
        if self.use_adam:
            self.logger.info("Starting Adam training")

            for epoch in range(self.adam_epochs):

                if epoch == 0 or epoch % self.resample_period == 0:
                    inputs = self.sample_collocation_points(self.num_collocation)

                residual = self.ode_residual(
                    input=inputs,
                    net=self.network,
                    t_max=t_max,
                    xi_min=xi_min,
                    xi_max=xi_max
                )

                loss = self.loss_fn(residual)

                self.adam_optimizer.zero_grad()
                loss.backward()
                self.adam_optimizer.step()

                self.loss_history.append(loss.item())
                self.epoch.append(iter)

                if epoch % 1000 == 0:
                    self.logger.info(f"[Adam] Epoch {epoch} | Loss: {loss.item():.6e}")
                
                iter += 1
            
        # ---------------------------------
        # Stage 2: LBFGS
        # ---------------------------------
        if self.use_lbfgs:
            self.logger.info("Starting LBFGS refinement")

            inputs = self.sample_collocation_points(self.num_collocation)
            lbfgs_iter = 0

            def closure():
                nonlocal iter, lbfgs_iter
                self.lbfgs_optimizer.zero_grad()

                residual = self.ode_residual(
                    input=inputs,
                    net=self.network,
                    t_max=self.t_max,
                    xi_min=self.xi_min,
                    xi_max=self.xi_max
                )

                loss = self.loss_fn(residual)
                loss.backward()
                # Store loss
                self.loss_history.append(loss.item())
                self.epoch.append(iter)

                # Logging every 1000 LBFGS steps
                if lbfgs_iter % 1000 == 0:
                    self.logger.info(
                        f"[LBFGS] Step {lbfgs_iter} | Loss: {loss.item():.6e}"
                    )

                iter += 1
                lbfgs_iter += 1
                return loss

            self.lbfgs_optimizer.step(closure)
            self.logger.info(f"[LBFGS] Final Loss: {self.loss_history[-1]:.6e}")

        self.logger.info("Training complete.")

        # plot losses
        self._save_loss_plot()
        

    def inference(self):
        self.network.eval()

        # Choose damping values
        xi_values = np.linspace(self.xi_min, self.xi_max, 5)

        z = np.linspace(0, self.t_max, self.num_test)
        t = z / self.t_max  # normalized time

        for xi_test in xi_values:

            # -----------------------------------
            # Analytical solution
            # -----------------------------------
            omega_d = np.sqrt(1 - xi_test**2)

            C1 = self.x_0
            C2 = (self.v_0 + xi_test * self.x_0) / omega_d

            x_exact = np.exp(-xi_test * z) * (
                C1 * np.cos(omega_d * z) +
                C2 * np.sin(omega_d * z)
            )

            # -----------------------------------
            # Numerical solution
            # -----------------------------------
            def ode_system(z, y):
                return [
                    y[1],
                    -2 * xi_test * y[1] - y[0]
                ]

            sol = solve_ivp(
                ode_system,
                [0, self.t_max],
                [self.x_0, self.v_0],
                t_eval=z
            )

            x_numeric = sol.y[0]

            # -----------------------------------
            # PINN prediction
            # -----------------------------------
            t_tensor = torch.tensor(
                t,
                dtype=torch.float32,
                device=self.device
            ).view(-1, 1)

            xi_norm = (xi_test - self.xi_min) / (self.xi_max - self.xi_min)
            xi_tensor = torch.full_like(t_tensor, xi_norm)

            inputs = torch.cat([t_tensor, xi_tensor], dim=1)

            with torch.no_grad():
                x_pred = self.network(inputs)

            x_pred = x_pred.cpu().numpy().flatten()

            # -----------------------------------
            # L2 error
            # -----------------------------------
            l2_error = np.sqrt(np.mean((x_pred - x_exact)**2))

            # -----------------------------------
            # Plot
            # -----------------------------------
            plt.figure(figsize=(8,5))
            plt.plot(z, x_exact, label="Analytical")
            plt.plot(z, x_numeric, '--', label="Numerical")
            plt.plot(z, x_pred, ':', label="PINN")
            plt.legend()
            plt.xlabel("z")
            plt.ylabel("x(z)")
            plt.title(f"Damped Oscillator (xi={xi_test:.2f}) | L2={l2_error:.2e}")
            plt.grid(True)

            # Save plot
            save_path = self.inference_dir / f"inference_xi_{xi_test:.2f}.png"
            plt.savefig(save_path, dpi=300)
            plt.close()

            self.logger.info(
                f"Inference saved for xi={xi_test:.2f} | L2 error={l2_error:.4e}"
            )




    def _save_loss_plot(self):
        plt.figure(figsize=(8,5))
        plt.plot(self.epoch, self.loss_history)
        plt.yscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.title("Training Loss (Adam + LBFGS)")
        plt.grid(True)

        save_path = self.run_dir / "train_loss.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        self.logger.info(f"Loss curve saved to {save_path}")


    def _setup_logger(self):
        self.log_dir = Path("outputs")
        self.log_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = self.config.get("run_name", "pinn_run")
        self.run_dir = self.log_dir / f"{run_name}_{timestamp}"
        self.run_dir.mkdir()

        self.inference_dir = self.run_dir / "inference_results"
        self.inference_dir.mkdir()

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