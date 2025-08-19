import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import os
from google.colab import drive

# Configuraciones
DATASET_PATH = "/content/drive/MyDrive/UTEC/CD & IA/Ciclo III/Machine Learning/Entrenamientos/Clasificación/Redes Neuronales/Análisis de Imágenes/Dataset/"
IMG_SIZE = 64
EPOCHS = 10
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Semilla reproducible
torch.manual_seed(42)
np.random.seed(42)

# Montar Google Drive
drive.mount('/content/drive')

# Transformaciones
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Clase wrapper para transformaciones
class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.subset)

# Modelo MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, activation='ReLU', dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden))
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'Tanh':
                layers.append(nn.Tanh())
            elif activation == 'Sigmoid':
                layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

# Modelo CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (IMG_SIZE // 4) ** 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

# Función de entrenamiento
def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs, model_name):
    train_losses, val_losses, train_accs, val_accs, epoch_times = [], [], [], [], []
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        print(f'{model_name} Epoch {epoch+1}/{epochs}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Acc {val_acc:.4f} | Time {epoch_time:.2f}s')
    return train_losses, val_losses, train_accs, val_accs, np.mean(epoch_times)

# Función de evaluación
def evaluate_model(model, loader, classes, model_name, split_name='Test'):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for inputs, true_labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(true_labels.numpy())
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(labels, preds)
    print(f'\n--- {model_name} {split_name} Evaluation ---')
    print(f'Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix ({split_name})')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_cm_{split_name.lower()}.png'))
    plt.close()
    return acc, macro_f1, weighted_f1, cm

# Función para graficar curvas
def plot_curves(train_losses, val_losses, train_accs, val_accs, model_name):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title(f'{model_name} Loss Curves')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.title(f'{model_name} Accuracy Curves')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_curves.png'))
    plt.close()

# Cargar dataset
full_dataset = datasets.ImageFolder(DATASET_PATH)
classes = full_dataset.classes
num_classes = len(classes)
total_samples = len(full_dataset)
train_size = int(0.8 * total_samples)
val_size = int(0.1 * total_samples)
test_size = total_samples - train_size - val_size
train_subset, val_subset, test_subset = random_split(full_dataset, [train_size, val_size, test_size])
train_dataset = TransformedDataset(train_subset, train_transform)
val_dataset = TransformedDataset(val_subset, val_test_transform)
test_dataset = TransformedDataset(test_subset, val_test_transform)

# Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuraciones MLP
mlp_configs = [
    {'hidden_sizes': [256], 'activation': 'ReLU', 'batch_size': 32, 'optimizer': 'Adam', 'lr': 0.001},
    {'hidden_sizes': [512, 256], 'activation': 'Tanh', 'batch_size': 64, 'optimizer': 'SGD', 'lr': 0.01},
    {'hidden_sizes': [128, 128], 'activation': 'Sigmoid', 'batch_size': 32, 'optimizer': 'Adam', 'lr': 0.001},
    {'hidden_sizes': [512], 'activation': 'ReLU', 'batch_size': 64, 'optimizer': 'SGD', 'lr': 0.01}
]

# Entrenamiento y evaluación MLP
mlp_results = []
input_size = IMG_SIZE * IMG_SIZE * 3
loss_fn = nn.CrossEntropyLoss()
for idx, config in enumerate(mlp_configs):
    print(f'\n--- MLP Config {idx+1}: {config} ---')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    model_mlp = MLP(input_size, config['hidden_sizes'], num_classes, config['activation']).to(device)
    optimizer = optim.Adam(model_mlp.parameters(), lr=config['lr']) if config['optimizer'] == 'Adam' else optim.SGD(model_mlp.parameters(), lr=config['lr'])
    train_losses, val_losses, train_accs, val_accs, avg_time = train_model(model_mlp, train_loader, val_loader, optimizer, loss_fn, EPOCHS, f'MLP_Config_{idx+1}')
    acc, macro_f1, weighted_f1, cm = evaluate_model(model_mlp, test_loader, classes, f'MLP_Config_{idx+1}')
    plot_curves(train_losses, val_losses, train_accs, val_accs, f'MLP_Config_{idx+1}')
    mlp_results.append({
        'config': config,
        'test_acc': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'avg_epoch_time': avg_time,
        'cm': cm,
        'curves': (train_losses, val_losses, train_accs, val_accs)
    })

# Seleccionar mejor MLP
best_mlp_idx = np.argmax([res['macro_f1'] for res in mlp_results])
best_mlp = mlp_results[best_mlp_idx]
print(f'\nMejor MLP: Config {best_mlp_idx+1}, Macro F1 {best_mlp["macro_f1"]:.4f}')

# Entrenamiento y evaluación CNN
batch_size_cnn = best_mlp['config']['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size_cnn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size_cnn)
test_loader = DataLoader(test_dataset, batch_size=batch_size_cnn)
model_cnn = SimpleCNN(num_classes).to(device)
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)
print('\n--- Entrenando CNN ---')
train_losses_cnn, val_losses_cnn, train_accs_cnn, val_accs_cnn, avg_time_cnn = train_model(model_cnn, train_loader, val_loader, optimizer_cnn, loss_fn, EPOCHS, 'CNN')
acc_cnn, macro_f1_cnn, weighted_f1_cnn, cm_cnn = evaluate_model(model_cnn, test_loader, classes, 'CNN')
plot_curves(train_losses_cnn, val_losses_cnn, train_accs_cnn, val_accs_cnn, 'CNN')

# Comparación MLP vs CNN
with open(os.path.join(OUTPUT_DIR, 'comparison.txt'), 'w') as f:
    f.write('Comparación entre MLP y CNN\n\n')
    f.write('| Metric                  | MLP             | CNN            |\n')
    f.write('|-------------------------|-----------------|----------------|\n')
    f.write(f'| Accuracy                | {best_mlp["test_acc"]:.4f}       | {acc_cnn:.4f}   |\n')
    f.write(f'| Macro F1                | {best_mlp["macro_f1"]:.4f}       | {macro_f1_cnn:.4f}   |\n')
    f.write(f'| Weighted F1             | {best_mlp["weighted_f1"]:.4f}       | {weighted_f1_cnn:.4f}   |\n')
    f.write(f'| Avg Time per Epoch (s)  | {best_mlp["avg_epoch_time"]:.2f}        | {avg_time_cnn:.2f}     |\n')

# Curvas comparativas
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(best_mlp['curves'][0], label='MLP Train Loss')
axs[0, 0].plot(best_mlp['curves'][1], label='MLP Val Loss')
axs[0, 0].legend()
axs[0, 1].plot(best_mlp['curves'][2], label='MLP Train Acc')
axs[0, 1].plot(best_mlp['curves'][3], label='MLP Val Acc')
axs[0, 1].legend()
axs[1, 0].plot(train_losses_cnn, label='CNN Train Loss')
axs[1, 0].plot(train_losses_cnn, label='CNN Val Loss')
axs[1, 1].plot(train_accs_cnn, label='CNN Train Acc')
axs[1, 1].plot(val_accs_cnn, label='CNN Val Acc')
axs[1, 1].legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_vs_cnn_curves.png'))
plt.close()

# Guardar matrices de confusión
with open(os.path.join(OUTPUT_DIR, 'confusion_matrices.txt'), 'w') as f:
    f.write('MLP Confusion Matrix:\n')
    f.write(str(best_mlp['cm']) + '\n\n')
    f.write('CNN Confusion Matrix:\n')
    f.write(str(cm_cnn))