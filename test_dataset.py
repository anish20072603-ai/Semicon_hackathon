from dataset import NPYImageDataset
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\anish\Downloads\Test_NoisyLR\NoisyLR"

dataset = NPYImageDataset(DATA_DIR)

image, filename = dataset[0]

# Remove channel dimension
image = image.squeeze().numpy()

plt.imshow(image, cmap="gray")
plt.title(filename)
plt.axis("off")
plt.show()