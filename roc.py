# read model from models/best.pth
import torch
import pandas as pd
from model import TitanicMLP, encode
import config

def draw_roc(find_best_thres=config.find_best_thres):
    """
    Draw ROC curve for the Titanic dataset predictions using the trained model.
    """
    #1.load model
    model = TitanicMLP(input_dim=config.input_dim, hidden_dim=config.hidden_dim, output_dim=1)
    model.load_state_dict(torch.load(config.model_read_path))

    #2.load train data for test
    data = pd.read_csv(config.train_data_path)
    data['Age'].fillna(data['Age'].mean(), inplace=True)  # Fill missing Age with mean

    #3.encode data
    encoded_data = encode(data)
    encoded_data = encoded_data.to(config.device)

    model.to(config.device)

    #4.predict
    model.eval()
    with torch.no_grad():
        predictions = model(encoded_data).squeeze()
        true_ys = data['Survived'].values
        # Draw ROC curve
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        fpr, tpr, thres = roc_curve(true_ys, predictions.cpu().numpy())

        if find_best_thres:
            # Find the threshold that gives the best FPR
            min_dis_to_leftup = 3
            for i in range(len(fpr)):
                dis_to_leftup = (fpr[i] - 0) ** 2 + (tpr[i] - 1) ** 2
                if dis_to_leftup < min_dis_to_leftup:
                    min_dis_to_leftup = dis_to_leftup
                    best_index = i
            print("Best threshold: {:.4f}, FPR: {:.4f}, TPR: {:.4f}".format(thres[best_index], fpr[best_index], tpr[best_index]))

        roc_auc = auc(fpr, tpr)
        print("ROC AUC: {:.4f}".format(roc_auc))
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.show()

if __name__ == "__main__":
    draw_roc()