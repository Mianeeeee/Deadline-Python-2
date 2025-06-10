import torch    # Image classification using both neural networks
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from Problem_1 import transform, trainloader, testloader, device

# Download full_trainset data
full_trainset = torchvision.datasets.CIFAR10(root='./SourceCode/data', train=True, download=True, transform=transform)

# Split train/validation
train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, valset = random_split(full_trainset, [train_size, val_size])

# DataLoader
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 512),

            nn.ReLU(),
            nn.Linear(512, 256),

            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)

# CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 → 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 → 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 8x8 → 4x4
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Train the model
def train_model(model, trainloader, valloader, num_epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Record the loss, accuracy and validation for each epoch
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_loss = running_loss / len(trainloader)
        train_acc = 100 * correct_train / total_train
        val_acc = evaluate_model(model, valloader)

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    return train_losses, train_accuracies, val_accuracies

# Evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

if __name__ == "__main__":
    # Run training the MLP/CNN model
    print("\nTraining MLP...")
    mlp_model = MLP()
    train_model(mlp_model, trainloader, valloader)

    print("\nTraining CNN...")
    cnn_model = CNN()
    train_model(cnn_model, trainloader, valloader)

    # Run evaluating the model
    print("Final Evaluation on Test Set")
    mlp_test_acc = evaluate_model(mlp_model, testloader)
    cnn_test_acc = evaluate_model(cnn_model, testloader)
    print(f"MLP Test Accuracy: {mlp_test_acc:.2f}%")
    print(f"CNN Test Accuracy: {cnn_test_acc:.2f}%")

# Save history
def save_history():    # Used to create data for plotting the learning curve
    mlp_model = MLP()
    cnn_model = CNN()

    print("\nTraining MLP...")
    mlp_train_loss, mlp_train_acc, mlp_val_acc = train_model(mlp_model, trainloader, valloader)

    print("\nTraining CNN...")
    cnn_train_loss, cnn_train_acc, cnn_val_acc = train_model(cnn_model, trainloader, valloader)

    return mlp_train_loss, mlp_train_acc, mlp_val_acc, cnn_train_loss, cnn_train_acc, cnn_val_acc