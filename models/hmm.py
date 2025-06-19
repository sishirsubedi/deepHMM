import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class HMM(nn.Module):
    """
    Variational Hidden Markov Model (V-HMM)
    """

    def __init__(self, input_dim, z_dim, config):
        super().__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.clip_norm = config['clip_norm']

        # p(z_t | z_{t-1}): transition logits [z_dim x z_dim]
        self.transition_logits = nn.Parameter(torch.randn(z_dim, z_dim))

        # p(x_t | z_t): emission parameters [z_dim x input_dim], assume Bernoulli
        self.emission_logits = nn.Parameter(torch.randn(z_dim, input_dim))

        # p(z_1): initial state logits [z_dim]
        self.initial_logits = nn.Parameter(torch.randn(z_dim))

        # Variational q(z_t | x_1:T) - approximate posterior
        # Simple param: one per time step, shared across batch
        self.q_logits = nn.Parameter(torch.randn(1, z_dim))

        # Optimizer
        self.optimizer = Adam(self.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']))

    def kl_div(self, q_logits, p_logits):
        """
        KL divergence between two categorical distributions.
        """
        q_probs = torch.softmax(q_logits, dim=-1)
        p_probs = torch.softmax(p_logits, dim=-1)
        kl = torch.sum(q_probs * (torch.log(q_probs + 1e-10) - torch.log(p_probs + 1e-10)), dim=-1)
        return kl

    def infer(self, x, x_lens):
        """
        Infer q(z_{1:T} | x_{1:T}) using simple variational q(z_t)
        """
        batch_size, T_max, _ = x.size()

        rec_losses = torch.zeros(batch_size, T_max, device=x.device)
        kl_states = torch.zeros(batch_size, T_max, device=x.device)

        # Shared q(z_t) across batch (naive, not RNN)
        q_z_t = self.q_logits.expand(batch_size, T_max, self.z_dim)

        for t in range(T_max):
            if t == 0:
                # p(z_1)
                p_z_t = self.initial_logits.expand(batch_size, -1)
            else:
                # p(z_t | z_{t-1}) ≈ prev q(z_{t-1}) * transition matrix
                prev_q_probs = torch.softmax(q_z_t[:, t - 1, :], dim=-1)
                p_z_t_probs = torch.matmul(prev_q_probs, torch.softmax(self.transition_logits, dim=-1))
                p_z_t = torch.log(p_z_t_probs + 1e-10)

            # KL(q(z_t) || p(z_t))
            kl_states[:, t] = self.kl_div(q_z_t[:, t, :], p_z_t)

            # p(x_t | z_t)
            emission_probs = torch.sigmoid(self.emission_logits)  # [z_dim x input_dim]
            q_probs = torch.softmax(q_z_t[:, t, :], dim=-1)
            expected_emission = torch.matmul(q_probs, emission_probs)  # [batch_size x input_dim]

            # Negative log likelihood (Bernoulli)
            rec_loss = -torch.sum(
                x[:, t, :] * torch.log(expected_emission + 1e-10)
                + (1.0 - x[:, t, :]) * torch.log(1.0 - expected_emission + 1e-10),
                dim=-1
            )
            rec_losses[:, t] = rec_loss

        # Mask for variable-length sequences
        x_mask = sequence_mask(x_lens, T_max)  # force T_max = 160
        x_mask = x_mask.gt(0).view(-1)      
        
        rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
        kl_loss = kl_states.view(-1).masked_select(x_mask).mean()

        return rec_loss, kl_loss

    def train_AE(self, x, x_lens, kl_anneal):
        """
        One training step.
        """
        self.train()
        rec_loss, kl_loss = self.infer(x, x_lens)
        loss = rec_loss + kl_anneal * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
        self.optimizer.step()

        return {
            'train_loss': loss.item(),
            'rec_loss': rec_loss.item(),
            'kl_loss': kl_loss.item()
        }

    def valid_step(self, x, x_lens):
        """
        Validation step.
        """
        self.eval()
        with torch.no_grad():
            rec_loss, kl_loss = self.infer(x, x_lens)
            loss = rec_loss + kl_loss
        return loss.item()

    def generate(self, x, x_lens):
        """
        Sample x_{1:T} ~ p(x_{1:T}, z_{1:T})
        """
        batch_size, _, _ = x.size()
        T_max = x_lens.max()

        x_gen = torch.zeros(batch_size, T_max, self.input_dim, device=x.device)

        # p(z_1)
        z_probs = torch.softmax(self.initial_logits, dim=-1).expand(batch_size, -1)

        for t in range(T_max):
            # Sample z_t
            z_t = torch.multinomial(z_probs, 1).squeeze(-1)  # [batch_size]

            # Sample x_t
            emission_probs = torch.sigmoid(self.emission_logits)  # [z_dim x input_dim]
            p_x_t = emission_probs[z_t]  # [batch_size x input_dim]
            x_t = torch.bernoulli(p_x_t)

            x_gen[:, t, :] = x_t

            # Next p(z_{t+1} | z_t)
            z_probs = torch.matmul(torch.nn.functional.one_hot(z_t, self.z_dim).float(),
                                   torch.softmax(self.transition_logits, dim=-1))

        return x_gen



def sequence_mask(lengths, max_len=None):
    """
    Create a boolean mask from sequence lengths.
    Args:
        lengths: [batch_size]
        max_len: int or None
    Returns:
        mask: [batch_size, max_len]
    """
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = lengths.max().item()

    range_row = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = range_row < lengths.unsqueeze(1)
    return mask