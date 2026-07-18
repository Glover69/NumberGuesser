from rewritePT.forge_dataset import ForgeDataset

ds = ForgeDataset("/Users/danielglover/Python/Forge/output/train")

print(f"Length of dataset: ", len(ds))

t, label = ds[0]
print("Tensor shape: ", t.shape)
print("Tensor dType: ", t.dtype)
print("Tensor min and max: ", t.min(), t.max())
print("Label for this particular data: ", label)