import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from rewritePT.forge_dataset import ForgeDataset
from rewritePT.pt_network import DigitNet

train_ds = ForgeDataset("/Users/danielglover/Python/Forge/output/train")
val_ds = ForgeDataset("/Users/danielglover/Python/Forge/output/val")


train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

# model, loss function and optimizer for training
model = DigitNet()
loss_fn = CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# training loop
for epoch in range(5):
    for batch_idx, (images, labels) in enumerate(train_loader):
        # clears old gradients, passes images forward, computes loss and alters
        # weights accordingly
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"epoch {epoch + 1}, batch {batch_idx}, loss {loss.item()}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            correct += (logits.argmax(dim=1) == labels).sum()

    print(f"Accuracy: {correct.item() / len(val_ds)}, Epoch: {epoch + 1}")

    model.train()

torch.save(model.state_dict(), "digitnet.pt")
