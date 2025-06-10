import matplotlib.pyplot as plt    # Draw the learning curve
from Problem_3 import save_history

# Save the result
mlp_train_loss, mlp_train_acc, mlp_val_acc, cnn_train_loss, cnn_train_acc, cnn_val_acc = save_history()

def plot_learning_curves(train_loss, train_acc, val_acc, title='Model'):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} - Training Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b-', label='Train Acc')
    plt.plot(epochs, val_acc, 'g--', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.tight_layout()
    
    # Save plt
    filename = f'SourceCode/Draw-the-learning-curve/{title}.png'
    plt.savefig(filename)
    print(f"Successfully saved the {title} learning model chart!")
    plt.show()
    plt.close()

# Draw the curve
plot_learning_curves(mlp_train_loss, mlp_train_acc, mlp_val_acc, title='MLP')
plot_learning_curves(cnn_train_loss, cnn_train_acc, cnn_val_acc, title='CNN')