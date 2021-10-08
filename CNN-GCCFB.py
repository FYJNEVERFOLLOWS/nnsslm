import torch.nn as nn
import torch

from torch import optim

import prepare_data_at_frame_level


# design model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # first conv and second conv
        self.conv_1 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=(5, 5), stride=2, padding=2),
            nn.BatchNorm2d(12), nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=(5, 5), stride=2, padding=2),
            nn.BatchNorm2d(24), nn.ReLU(inplace=True)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=(5, 5), stride=2, padding=2),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=(5, 5), stride=2, padding=2),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=96*3*4, out_features=360),
            # nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_1(x)
        print("after conv_1 x.shape:{}".format(x.shape))
        x = self.conv_2(x)
        print("after conv_2 x.shape:{}".format(x.shape))
        x = self.conv_3(x)
        print("after conv_3 x.shape:{}".format(x.shape))
        x = self.conv_4(x)
        print("after conv_4 x.shape:{}".format(x.shape))
        # flatten data from (n, 96, 3, 4) to (n, 96*3*4)
        x = x.view(batch_size, -1)
        print('before fc x.shape:{}'.format(x.shape))
        x = self.fc(x)

        return x


model = CNN()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update
def train(epoch):
    running_loss = 0.0

    data_loader = prepare_data_at_frame_level.DataLoader()
    # Train
    iter = 0
    for (batch_x, batch_y) in data_loader.generate():
        # 获得一个批次的数据和标签(inputs, labels)
        print("batch_x {}".format(batch_x.shape))
        print("batch_y {}".format(batch_y.shape))
        # 获得模型预测结果
        output = model(batch_x)
        print("output {}".format(output.shape))
        # 交叉熵代价函数
        loss = criterion(output, batch_y)

        running_loss += loss.item()
        if iter % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, iter + 1, running_loss / 100))
            running_loss = 0.0

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter += 1
    print('iter: {}'.format(iter))
        # Evaluate


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
