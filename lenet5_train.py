from lenet5_model import Model
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchvision import datasets

import torch

if __name__ == '__main__':

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 128

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = Model()
    sgd = SGD(model.parameters(), lr=1e-1)
    cost = CrossEntropyLoss()
    epoch = 6

    for _epoch in range(epoch):
        print('epoch: {}'.format(_epoch))
        model.train()
        for idx, (train_x, train_label) in enumerate(train_dataloader):
            label_np = np.zeros((train_label.shape[0], 10))
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = cost(predict_y, train_label.long())
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            sgd.step()

        correct = 0
        _sum = 0
        model.eval()
        for idx, (test_x, test_label) in enumerate(test_dataloader):
            predict_y = model(test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            label_np = test_label.numpy()
            _ = predict_ys == test_label
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        print('accuracy: {:.2f}'.format(correct / _sum))
    #at the end
    torch.save(model, 'mnist_{:.2f}.pkl'.format(correct / _sum))
