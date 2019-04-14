import torch

n_hidden=10

net_dropped = torch.nn.Sequential(
    torch.nn.Linear(1, n_hidden),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden, n_hidden),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden, 1),
)