from tabnanny import verbose

import torch
import torch.nn as nn
from torch import optim

from ceg4n.constants import device


class Trainer:
    def __init__(
        self, model, loaders, num_epochs, lr=0.01, momentum=0.5, verbose=False
    ):
        self.model = model
        self.loaders = loaders
        self.num_epochs = num_epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        self.verbose = verbose

    def train(self):
        test_loss, test_accuracy = self._test()
        print(
            "Before train: Loss={:.4}, Accuracy={:.4}".format(test_loss, test_accuracy)
        )
        # self._report(epoch, train_loss, train_accuracy, test_loss, test_accuracy)
        for epoch in range(self.num_epochs):
            train_loss, train_accuracy = self._train()
            test_loss, test_accuracy = self._test()
            self._epoch_progress(
                epoch, train_loss, train_accuracy, test_loss, test_accuracy
            )
        print(
            "After train: Loss={:.4}, Accuracy={:.4}".format(test_loss, test_accuracy)
        )

    def _train(self):
        self.model.train()
        self.model.to(device)
        epoch_loss = 0
        epoch_accuracy = 0

        for batch_idx, (x, y) in enumerate(self.loaders["train"]):
            x = x.to(device)
            y = y.to(device)

            output = self.model(x)
            loss = self.loss_fn(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            epoch_accuracy += pred.eq(y.data.view_as(pred)).sum().item()
        return epoch_loss / len(self.loaders["train"]), epoch_accuracy / len(
            self.loaders["train"]
        )

    @torch.no_grad()
    def _test(self):
        self.model.eval()
        self.model.to(device)
        epoch_loss = 0
        epoch_accuracy = 0
        for batch_idx, (x, y) in enumerate(self.loaders["test"]):
            x = x.to(device)
            y = y.to(device)
            output = self.model(x)
            loss = self.loss_fn(output, y)
            epoch_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            epoch_accuracy += pred.eq(y.data.view_as(pred)).sum().item()
        return epoch_loss / len(self.loaders["test"]), epoch_accuracy / len(
            self.loaders["test"]
        )

    def _epoch_progress(
        self, epoch, train_loss, train_accuracy, test_loss, test_accuracy
    ):
        if not self.verbose:
            return
        self._report(epoch, train_loss, train_accuracy, test_loss, test_accuracy)

    def _report(self, epoch, train_loss, train_accuracy, test_loss, test_accuracy):
        epoch_part = f"Epoch [{epoch+1}/{self.num_epochs}]"
        loss_part = "Train/Test Loss: {:.4f}/{:.4f}".format(train_loss, test_loss)
        accuracy_part = "Train/Test Accuracy: {:.4f}/{:.4f}".format(
            train_accuracy, test_accuracy
        )
        print(f"{epoch_part}, {loss_part}, {accuracy_part}")
