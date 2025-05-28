import pandas as pd
from sklearn.calibration import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def encode(data): #data is a pandas DataFrame
    batch_size = len(data['PassengerId'])
    
    # 初始化一个全零的张量，形状为 (batch_size, 13)
    encoded_vector = torch.zeros((batch_size, 13), dtype=torch.float32)
    
    # Pclass 的独热编码 (前 3 个分量)
    pclass = torch.tensor(data['Pclass'].values, dtype=torch.long)
    encoded_vector[:, :3] = F.one_hot(pclass - 1, num_classes=3).float()  # Pclass 从 1 开始，需要减 1
    
    # Sex 的独热编码 (第 4 和 5 个分量)
    sex = LabelEncoder().fit_transform(data['Sex'].values)
    encoded_vector[:, 3:5] = F.one_hot(torch.tensor(sex, dtype=torch.long), num_classes=2).float()
    
    # Age, SibSp, Parch 的具体数值 (第 6, 7, 8 个分量)
    encoded_vector[:, 5] = torch.tensor(data['Age'].values, dtype=torch.float32)
    encoded_vector[:, 6] = torch.tensor(data['SibSp'].values, dtype=torch.float32)
    encoded_vector[:, 7] = torch.tensor(data['Parch'].values, dtype=torch.float32)
    
    # Fare 的具体数值 (第 9 个分量)
    encoded_vector[:, 8] = torch.tensor(data['Fare'].values, dtype=torch.float32)
    
    # Embarked 的独热编码 (第 10-12 个分量)
    embarked = LabelEncoder().fit_transform(data['Embarked'].fillna('?'))
    encoded_vector[:, 9:13] = F.one_hot(torch.tensor(embarked, dtype=torch.long), num_classes=4).float()
    
    return encoded_vector

class TitanicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TitanicMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 第二层
        self.fc3 = nn.Linear(hidden_dim, output_dim) # 输出层
        self.relu = nn.ReLU()  # 激活函数
        self.dropout = nn.Dropout(0.5)  # Dropout 防止过拟合

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
