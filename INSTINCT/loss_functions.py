import torch
import torch.nn.functional as F


def classification_loss(cls_output, cls_target):
    loss = F.cross_entropy(cls_output, cls_target)
    return loss


def adversarial_loss_g(Y_src):
    loss = (torch.log1p(torch.exp(-Y_src))).mean()
    return loss


def adversarial_loss_d(X_src, Y_src):
    loss = (torch.log1p(torch.exp(-X_src)) + torch.log1p(torch.exp(Y_src))).mean()
    return loss


def latent_loss(D_concat, alpha, margin=None):
    if margin:
        loss = torch.mean(torch.relu(torch.clamp_max(D_concat, margin) + alpha))
    else:
        loss = torch.mean(torch.relu(D_concat + alpha))
    return loss


def mse_loss(X, X_rec):
    loss = torch.mean((X_rec - X)**2)
    return loss

