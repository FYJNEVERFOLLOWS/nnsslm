import torch.nn as nn
import torch
from tqdm import tqdm

from torch import optim

import prepare_data_at_frame_level_dataloader
from torch.utils.data import DataLoader

train_data_path = "/Work20/2021/fuyanjie/exp_data/exp_nnsslm/train_data_dir/train_data_frame_level"
test_data_path = "/Work20/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/test_data_frame_level"
device = torch.device('cuda:0')
# device = torch.device('cpu')

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
            nn.Linear(in_features=96 * 3 * 4, out_features=360),
            # nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_1(x)
        # print("after conv_1 x.shape:{}".format(x.shape))
        x = self.conv_2(x)
        # print("after conv_2 x.shape:{}".format(x.shape))
        x = self.conv_3(x)
        # print("after conv_3 x.shape:{}".format(x.shape))
        x = self.conv_4(x)
        # print("after conv_4 x.shape:{}".format(x.shape))
        # flatten data from (n, 96, 3, 4) to (n, 96*3*4)
        x = x.view(batch_size, -1)
        # print('before fc x.shape:{}'.format(x.shape))
        x = self.fc(x)

        return x


model = CNN()
model.to(device)

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_data = DataLoader(prepare_data_at_frame_level_dataloader.SSLR_Dataset(train_data_path), batch_size=256,
                        shuffle=True, num_workers=4)  # train_data is a tuple: (batch_x, batch_y)
test_data = DataLoader(prepare_data_at_frame_level_dataloader.SSLR_Dataset(test_data_path), batch_size=256,
                       shuffle=True, num_workers=4)  # test_data is a tuple: (batch_x, batch_y)


# training cycle forward, backward, update
def train(epoch):
    running_loss = 0.0

    # Train
    iter = 0
    total_loss = 0.
    sam_size = 0.
    # for (batch_x, batch_y) in tqdm(train_data, desc=f'train epoch{epoch}'):
    for (batch_x, batch_y) in train_data:
        # 获得一个批次的数据和标签(inputs, labels)
        # print("batch_x {}".format(batch_x.shape))
        # print("batch_y {}".format(batch_y.shape))
        # 获得模型预测结果
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        output = model(batch_x)
        # print("output {}".format(output.shape))

        # 交叉熵代价函数
        loss = criterion(output, batch_y)

        running_loss += loss.item()
        if iter % 1000 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, iter + 1, running_loss / 1000))
            running_loss = 0.0
        with torch.no_grad():
            total_loss += loss.clone().detach().item()
            sam_size += batch_y.shape[0]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter += 1
    # print the MSE and the sample size
    print(f'epoch {epoch} loss {total_loss / sam_size} sam_size {sam_size}')

# Evaluate
def test():
    correct = 0
    correct_coarse = 0
    total = 0

    with torch.no_grad():
        # for (batch_x, batch_y) in tqdm(test_data, desc=f"test epoch{epoch}"):
        for (batch_x, batch_y) in test_data:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # 获得模型预测结果
            output = model(batch_x)
            # dim=1是class所在维度，0是batch_size所在维度。等价于 predicted = output.argmax(dim=-1)
            _, predicted = torch.max(output.data, dim=1)
            # batch_y.shape[0] = batch_size, predicted.shape: [batch_size] / torch.Size([batch_size])
            total += batch_y.size(0)
            # 张量之间的比较运算，等价于 torch.sum(predicted == batch_y).item()
            correct += ((predicted - batch_y).ge(-2) & (predicted - batch_y).le(2)).sum().item()
            correct_coarse += ((predicted - batch_y).ge(-5) & (predicted - batch_y).le(5)).sum().item()
    print('accuracy on test set: %.2f %% ' % (100.0 * correct / total))
    print('accuracy_coarse on test set: %.2f %% ' % (100.0 * correct_coarse / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
