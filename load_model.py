import numpy as np
import matplotlib.pyplot as plt
from lenet5_model import Model
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor


model = Model()
model = torch.load("mnist_0.98.pkl")
print(model)
batch_size = 128
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

model.eval()
(data, label) = test_data[0]
plt.imshow(data.reshape(28, 28), cmap='gray')
plt.title('label is :{}'.format(label))
plt.show()

data = data[None]
with torch.no_grad():
    pred = model(data.float()).detach()
    
    predicted, actual = classes[pred[0].argmax(0)], classes[label]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

