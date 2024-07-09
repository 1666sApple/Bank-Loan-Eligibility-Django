import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle

# Load the dataset
df = pd.read_csv('data/bankloan.csv')
df = df.dropna()
df = df.drop('Loan_ID', axis=1)
df['LoanAmount'] = (df['LoanAmount'] * 1000).astype(int)

# Label encoding
label_encoder = LabelEncoder()
features_to_encode = ['Gender', 'Married', 'Education', 'Dependents', 'Self_Employed', 'Property_Area', 'Loan_Status']
for feature in features_to_encode:
    df[feature] = label_encoder.fit_transform(df[feature])

X, y = df.iloc[:, :-1], df.iloc[:, -1]

# SMOTE for handling class imbalance
smote = SMOTE(sampling_strategy='minority')
X, y = smote.fit_resample(X, y)

# Scale the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# Custom Dataset class
class LoanEligibilityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Neural network model
class AdaptiveSBCODNFN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AdaptiveSBCODNFN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Training function
def train(model, train_loader, val_loader, epochs, lr, save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path + ".pth")
            with open(save_path + ".pkl", "wb") as f:
                pickle.dump(model, f)

    return model

# Create datasets and dataloaders
train_dataset = LoanEligibilityDataset(X_train, y_train)
val_dataset = LoanEligibilityDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# Initialize and train the model
model = AdaptiveSBCODNFN(input_size=X_train.shape[1], num_classes=2)
trained_model = train(model, train_loader, val_loader, epochs=100, lr=0.001, save_path="best_model")
