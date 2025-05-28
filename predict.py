# read model from models/best.pth
import torch
import pandas as pd
from model import TitanicMLP, encode
import config

def predict():
    """
    Give predictions for the given data using the trained model.
    """
    #1.load model
    model = TitanicMLP(input_dim=config.input_dim, hidden_dim=config.hidden_dim, output_dim=1)
    model.load_state_dict(torch.load(config.model_read_path))

    #2.load test data
    test_data = pd.read_csv(config.test_data_path)
    test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)  # Fill missing Age with mean

    #3.encode test data
    encoded_test_data = encode(test_data)
    encoded_test_data = encoded_test_data.to(config.device)

    model.to(config.device)

    #4.predict
    model.eval()
    with torch.no_grad():
        predictions = model(encoded_test_data).squeeze()
        predictions = (predictions > config.threshold).int()  # Convert probabilities to binary predictions

        output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions.to('cpu').numpy()})
        output.to_csv('submission.csv', index=False)
        print("Your submission was successfully saved!")

if __name__ == "__main__":
    predict()