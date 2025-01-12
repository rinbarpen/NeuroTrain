import torch
from torch import nn

class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 测试LSTMModule
if __name__ == "__main__":
    input_tensor = torch.randn(32, 10, 8)  # 假设输入是一个32批次，10时间步，8特征的张量
    lstm_module = LSTMModule(input_size=8, hidden_size=16, num_layers=2, output_size=4)
    output_tensor = lstm_module(input_tensor)
    print(output_tensor.size())  # 输出的尺寸应为 (32, 4)
