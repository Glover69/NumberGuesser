import numpy as np
import torch as pt
from PIL import Image
from torch.utils.data import DataLoader

from rewritePT.forge_dataset import ForgeDataset
from rewritePT.pt_network import DigitNet

model = DigitNet()
model.load_state_dict(pt.load("digitnet.pt"))
model.eval()

val_ds = ForgeDataset("/Users/danielglover/Python/Forge/output/val")
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)


# correct = 0
# with pt.no_grad():
#     for images, labels in val_loader:
#         logits = model(images)
#         correct += (logits.argmax(dim=1) == labels).sum()
#
# print(f"Accuracy: {correct.item() / len(val_ds)}")


# preprocess image
image_dir = "/Users/danielglover/PycharmProjects/NumberGuesser/data/hand-drawn.jpg"

_img = Image.open(image_dir).convert('L')
pixels = np.array(_img)

normalized = pixels / 255
tensor = pt.from_numpy(normalized).to(pt.float32)

tensor = tensor.unsqueeze(0)

with pt.no_grad():
    logits = model(tensor)

prediction = logits.argmax(dim=1).item()
print("Prediction: ", prediction)
