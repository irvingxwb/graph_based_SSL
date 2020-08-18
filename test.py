import torch
import torch.nn.functional as f


def pairwise_distances(x, y):
    x = f.normalize(x).view(1, -1)
    y = f.normalize(y).view(1, -1)

    return torch.mm(x, y.transpose(1, 0))


if __name__ == "__main__":
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float).view(1, -1)
    y = torch.tensor([6, 7, 8, 9, 1], dtype=torch.float).view(1, -1)
    x = x.cuda()
    y = y.cuda()
    dist = pairwise_distances(x, y)
    print(round(dist))
