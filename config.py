import torch

train_data_path = './kaggle/input/titanic/train.csv'
test_data_path = './kaggle/input/titanic/test.csv'
model_read_path = './models/best_model_with_carbinembedding.pth'

input_dim = 46
hidden_dim = 64

find_best_thres = True

threshold = -0.9491

embed_carbin = False

draw_loss_curve = True

batch_size = 64
num_epochs = 15
learning_rate = 0.0025
device = 'cuda' if torch.cuda.is_available() else 'cpu'
