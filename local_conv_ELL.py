import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from linearFA import LinearFA
from surrogate_gradient import ExponentialSurroGrad

class LocalLossBlockConv(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 kernel_size,
                 stride,
                 padding,
                 dim_out,
                 dim_decoder,
                 num_classes,
                 threshold=None,
                 decay=0.2,
                 print_stats=False,
                 first_layer=False,
                 bias=False,
                 random_matrix=False,
                 fa=False,
                 last_layer=False):
        super(LocalLossBlockConv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.bias = bias
        self.thresh = threshold
        self.decay = decay
        self.dim_in_decoder = dim_decoder
        self.random_matrix = random_matrix
        self.fa = fa
        self.is_print_stats = print_stats
        self.encoder = nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride, padding=padding, bias=self.bias)

        if True:
            # Resolve average-pooling kernel size in order for flattened dim to match self.dim_in_decoder
            ks_h, ks_w = 1, 1
            dim_out_h, dim_out_w = dim_out, dim_out
            dim_in_decoder = ch_out * dim_out_h * dim_out_w
            while dim_in_decoder > self.dim_in_decoder and ks_h < dim_out:
                ks_h *= 2
                dim_out_h = math.ceil(dim_out / ks_h)
                dim_in_decoder = ch_out * dim_out_h * dim_out_w
                if dim_in_decoder > self.dim_in_decoder:
                    ks_w *= 2
                    dim_out_w = math.ceil(dim_out / ks_w)
                    dim_in_decoder = ch_out * dim_out_h * dim_out_w
            if ks_h > 1 or ks_w > 1:
                pad_h = (ks_h * (dim_out_h - dim_out // ks_h)) // 2
                pad_w = (ks_w * (dim_out_w - dim_out // ks_w)) // 2
                self.avg_pool = nn.AvgPool2d((ks_h, ks_w), padding=(pad_h, pad_w))
                print(self.avg_pool)
            else:
                self.avg_pool = None

        # Auxiliary classifier
        if self.fa:
            self.decoder_y = LinearFA(dim_in_decoder, num_classes, bias=self.bias)
        else:
            self.decoder_y = nn.Linear(dim_in_decoder, num_classes, bias=self.bias)

        if self.random_matrix:
            self.decoder_y.weight.requires_grad = False
            if self.decoder_y.bias is not None:
                self.decoder_y.bias.requires_grad = False

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0, weight_decay=0.0)

        self.clear_stats()

    def clear_stats(self):
        self.loss_pred = 0.0
        self.correct = 0
        self.examples = 0

    def print_stats(self):
        err = 100.0 * float(self.examples - self.correct) / self.examples
        stats = '{},loss_pred={:.4f}, error={:.3f}%, num_examples={}\n'.format(
            self.encoder,
            self.loss_pred / self.examples,
            err,
            self.examples)
        return stats, err

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        self.optimizer.step()

    def forward(self, x, y, y_onehot, time_window=10):
        # Loop over time window T
        for step in range(time_window):
            # Compute the input current
            if self.first_layer:
                h = self.encoder(x)
            else:
                h = self.encoder(x[step])

            # Membrane integration and spike firing
            if step == 0:
                mem = h
                spike = ExponentialSurroGrad.apply(mem, self.thresh)
                spike_return = spike.clone()  # Output spikes
                if self.training or self.is_print_stats:
                    if self.avg_pool is not None:
                        mem_pool = self.avg_pool(
                            spike)  # Average pooling operation is applied first in order to reduce the number of auxilinary weights
                    else:
                        mem_pool = spike
                    y_hat_mem = self.decoder_y(mem_pool.view(mem_pool.size(0), -1))
                    y_hat_spike = ExponentialSurroGrad.apply(y_hat_mem, self.thresh)
                    spike_sum = y_hat_spike  # Accumulate spikes for decision
            else:
                # Detach the membrane and spike from the computation graph because ELL doesn't consider temporal dependent gradients.
                mem = mem.detach() * self.decay + h - spike.detach() * self.thresh * self.decay  # Membrane integration
                spike = ExponentialSurroGrad.apply(mem, self.thresh)
                spike_return = torch.cat((spike_return, spike), dim=0)
                if self.training or self.is_print_stats:
                    if self.avg_pool is not None:
                        mem_pool = self.avg_pool(spike)
                    else:
                        mem_pool = spike
                    y_hat_mem = y_hat_mem.detach() * self.decay + self.decoder_y(
                        mem_pool.view(mem_pool.size(0), -1)) - y_hat_spike.detach() * self.thresh * self.decay
                    y_hat_spike = ExponentialSurroGrad.apply(y_hat_mem, self.thresh)
                    spike_sum = spike_sum + y_hat_spike

            # Compute loss and update parameters during training
            if self.training or self.is_print_stats:
                loss_sup = F.mse_loss(y_hat_spike, y_onehot.detach())
                if step == 0:
                    loss = loss_sup
                else:
                    loss = loss + loss_sup

                if self.training:
                    self.optimizer.zero_grad()
                    loss_sup.backward(retain_graph=False)
                    self.optimizer.step()  # Update all parameters

        spike_return = spike_return.view(-1, *spike.size())

        if self.is_print_stats:
            self.loss_pred += loss.item() * y_onehot.size(0)
            self.correct += spike_sum.max(1)[1].eq(y).cpu().sum()
            self.examples += y_onehot.size(0)

        if self.last_layer:
            return (100.0 * float(self.examples - self.correct) / self.examples), loss.item()
        return spike_return.detach(), loss.item()
