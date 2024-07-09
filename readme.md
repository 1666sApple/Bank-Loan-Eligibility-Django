# Loan Eligibility Prediction

This repository contains a machine learning project for predicting loan eligibility using neural networks, including an Adaptive Sequentially Bi-Connected Output Deep Neural Fully Connected Network (AdaptiveSBCODNFN) and a 1D Convolutional Neural Network (CNN1D).

## Directory Structure

```bash
loan_eligibility/
├── init.py
├── data_processing.py
├── dataset.py
├── models.py
├── train.py
└── main.py
data/
└── bankloan.csv
```


## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn

### Installation

1. Clone the repository:

```sh
git clone https://github.com/your-username/loan-eligibility.git
cd loan-eligibility
```

2. Create a virtual environment and activate it:
```sh
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```
3. Install the required packages:
```sh
pip install -r requirements.txt
```
## Usage
### Data Processing
The data processing script loads and preprocesses the dataset, including handling missing values, encoding categorical features, and scaling numerical features.

```python
from loan_eligibility.data_processing import load_and_process_data

X_train, X_val, y_train, y_val = load_and_process_data('data/bankloan.csv')
```
### Dataset Preparation
The dataset script creates a custom PyTorch dataset for loan eligibility.

```python
from loan_eligibility.dataset import LoanEligibilityDataset

train_dataset = LoanEligibilityDataset(X_train, y_train)
val_dataset = LoanEligibilityDataset(X_val, y_val)
```
### Model Training
The training script handles the training and validation of the model, saving the best model based on validation accuracy.

```python
from loan_eligibility.models import AdaptiveSBCODNFN, CNN1D
from loan_eligibility.train import train

model = AdaptiveSBCODNFN(input_size=X_train.shape[1], num_classes=2)
trained_model = train(model, train_loader, val_loader, epochs=50, lr=0.001, save_path="best_model")
```

### Running the Main Script
To run the complete workflow, use the main.py script:

```sh
python loan_eligibility/main.py
```

### Models
#### AdaptiveSBCODNFN
A fully connected neural network with multiple hidden layers and ReLU activations.

#### CNN1D
A 1D convolutional neural network for sequential data.

### Contributing
- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes.
- Commit your changes (git commit -m 'Add some feature').
- Push to the branch (git push origin feature-branch).
- Create a new Pull Request.

### Acknowledgements
- PyTorch
- Pandas
- Scikit-learn
- Imbalanced-learn