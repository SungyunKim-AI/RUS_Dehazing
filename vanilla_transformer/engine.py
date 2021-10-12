import torch
from tqdm import tqdm

def train_one_epoch(dataloader, model, loss_fn, optimizer, device, epoch):
    model.train()
    # loss_fn.train()
    
    total_loss, running_loss = 0.0, 0.0
    for batch in tqdm(dataloader):
        hazy, clear = batch
        inputs, targets = hazy.to(device), clear.to(device)
        
        outputs = model(inputs)
        print(outputs.shape)
        print(targets.shape)
        
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss


@torch.no_grad()
def evaluate(dataloader, model, loss_fn, device):
    model.eval()
    # loss_fn.eval()
    
    test_loss = [], 0.0
    for i, batch in enumerate(dataloader):
        hazy, clear = batch
        inputs, targets = hazy.to(device), clear.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        test_loss += loss.item()
        
    return test_loss
        
        
    