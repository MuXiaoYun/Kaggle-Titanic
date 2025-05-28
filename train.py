from create_folds import create_folds
from model import encode, TitanicMLP
import config
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score

def load_data(file_path):
    """
    Load the Titanic dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    # Fill missing values
    data['Age'].fillna(data['Age'].mean(), inplace=True)  # Fill missing Age with mean
    # turn to tensor
    data = create_folds(data)  # Create folds for cross-validation
    return data

def process_data(data, n_splits=5, validation_fold=0):
    # Split the data into training and validation sets based on folds
    train_data = data[data['fold'] != validation_fold].reset_index(drop=True)
    validation_data = data[data['fold'] == validation_fold].reset_index(drop=True)
    return train_data, validation_data

def train():
    """
    Train the model on the Titanic dataset.
    """
    # Define hyperparameters
    input_dim = config.input_dim  # Number of features after encoding
    hidden_dim = config.hidden_dim  # Size of hidden layer
    output_dim = 1  # Binary classification (0 or 1)
        
    # Initialize the model
    model = TitanicMLP(input_dim, hidden_dim, output_dim)
    losses = []
    for fold in range(5):
        print(f"Training on fold {fold}...")
        train_data, validation_data = process_data(load_data(config.train_data_path), n_splits=5, validation_fold=fold)
        
        # Print model summary
        print(model)

        batch_size = config.batch_size
        num_epochs = config.num_epochs
        learning_rate = config.learning_rate
        device = config.device

        # Move model to the appropriate device
        model.to(device)
        print(f"Using device: {device}")

        # Define loss function and optimizer
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            
            
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data.iloc[i:i + batch_size]
                if len(batch_data) == 0:
                    continue
                
                # Encode the batch data
                encoded_data = encode(batch_data)
                encoded_data = encoded_data.to(device)
                
                # Get labels
                labels = batch_data['Survived'].values.astype(float)
                labels = torch.tensor(labels).unsqueeze(1).to(device)  # Reshape to (batch_size, 1)
                
                # Forward pass
                outputs = model(encoded_data)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            losses.append(total_loss / len(train_data))
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_data):.4f}')
        print("Training complete.")

        # validate
        model.eval()
        total_val_loss = 0.0
        auc = 0.0
        count = 0
        with torch.no_grad():
            for i in range(0, len(validation_data), batch_size):
                batch_data = validation_data.iloc[i:i + batch_size]
                if len(batch_data) == 0:
                    continue
                
                # Encode the batch data
                encoded_data = encode(batch_data)
                encoded_data = encoded_data.to(device)
                
                # Get labels
                labels = batch_data['Survived'].values.astype(float)
                labels = torch.tensor(labels).unsqueeze(1).to(device)  # Reshape to (batch_size, 1)
                
                # Forward pass
                outputs = model(encoded_data)
                loss = criterion(outputs, labels)
                
                total_val_loss += loss.item()
                # Calculate AUC
                outputs = torch.sigmoid(outputs).cpu().numpy()
                labels = labels.cpu().numpy()
                
                auc += roc_auc_score(labels, outputs)
                count += 1

            
            print(f'Validation Loss: {total_val_loss / len(validation_data):.4f}')
            print(f'Validation AUC: {auc / count:.4f}')
        print("Validation complete.")

    
    if config.draw_loss_curve:
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
    # get time 
    import time
    current_time = time.strftime("%Y-%m-%d_%H_%M_%S_", time.localtime())
    # Save the model and name it with the current time
    torch.save(model.state_dict(), './models/titanic_mlp_' + str(current_time) + str(auc) + '.pth')

if __name__ == "__main__":
    train()
