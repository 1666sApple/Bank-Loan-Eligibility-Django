import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

def load_and_process_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    df = df.drop('Loan_ID', axis=1)
    df['LoanAmount'] = (df['LoanAmount'] * 1000).astype(int)

    label_encoder = LabelEncoder()
    features_to_encode = ['Gender', 'Married', 'Education', 'Dependents', 'Self_Employed', 'Property_Area', 'Loan_Status']
    for feature in features_to_encode:
        df[feature] = label_encoder.fit_transform(df[feature])

    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    smote = SMOTE(sampling_strategy='minority')
    X, y = smote.fit_resample(X, y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y
