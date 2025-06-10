import torch    # Draw the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from Problem_3 import MLP, CNN, testloader, device

# Declare model
mlp_model = MLP()
cnn_model = CNN()

def plot_confusion_matrix(model, dataloader, classes, title='Model'):
    model.eval()  # Switch to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"SourceCode/Confusion-matrix/{title}.png")
    plt.show()
    plt.close()

# CIFAR-10 class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

plot_confusion_matrix(mlp_model, testloader, classes, title='MLP')
plot_confusion_matrix(cnn_model, testloader, classes, title='CNN')