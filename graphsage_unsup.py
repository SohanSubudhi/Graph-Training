import torch
import torch.nn.functional as F
import tqdm

def train(model, optimizer, loader, device):
    retval = iterate(model, optimizer, loader, device, train = True)
    return retval

def evaluate(model, optimizer, loader, device):
    with torch.no_grad():
        retval = iterate(model, optimizer, loader, device, train = False)
    return retval

def iterate(model, optimizer, loader, device, train = True):
    if train:
        model.train()
    else: 
        model.eval() 
    
    total_loss = total_examples = 0
    for data in tqdm.tqdm(loader):
        data = data.to(device)
        
        if train:
            optimizer.zero_grad()
        
        h = model(data.x, data.edge_index)

        h_src = h[data.edge_label_index[0]]
        h_dst = h[data.edge_label_index[1]]
        link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.

        loss = F.binary_cross_entropy_with_logits(link_pred, data.edge_label)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss) * link_pred.numel()
        total_examples += link_pred.numel()

    return total_loss / total_examples