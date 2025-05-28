import torch

train_data_path = './kaggle/input/titanic/train.csv'
test_data_path = './kaggle/input/titanic/test.csv'
model_read_path = './models/hid32.pth'
input_dim = 13
hidden_dim = 64
threshold = 0.5

draw_loss_curve = True
draw_roc_curve = True

batch_size = 64
num_epochs = 15
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
