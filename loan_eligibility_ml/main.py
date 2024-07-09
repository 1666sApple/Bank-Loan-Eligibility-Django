import sys
import os

# Add the project directory to the sys.path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from loan_eligibility.data_processing import load_and_process_data
from loan_eligibility.dataset import LoanEligibilityDataset
from loan_eligibility.model import AdaptiveSBCODNFN, CNN1D
from loan_eligibility.train import train

# Load and process data
X, y = load_and_process_data('data/bankloan.csv')

# Train-validation split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# Create datasets and dataloaders
from torch.utils.data import DataLoader
train_dataset = LoanEligibilityDataset(X_train, y_train)
val_dataset = LoanEligibilityDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# Initialize and train the model
model_type = input(f"Enter the model type(fnn/cnn): ").lower()
if model_type == 'fnn':
    model = AdaptiveSBCODNFN(input_size=X_train.shape[1], num_classes=2)
elif model_type == 'cnn':
    model = CNN1D(input_size=X_train.shape[1], num_classes=2)

trained_model = train(model, train_loader, val_loader, epochs=500, lr=0.001, save_path="best_model")
