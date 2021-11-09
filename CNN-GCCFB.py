import numpy as np
import torch.nn as nn
import torch

from torch import optim

import prepare_multi_sources_data
import func
from torch.utils.data import DataLoader

train_data_path = "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/train_data_dir/train_data_frame_level_gcc"
test_data_path = "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/test_data_frame_level_gcc"
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
            nn.Sigmoid()
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
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_data = DataLoader(prepare_multi_sources_data.SSLR_Dataset(train_data_path), batch_size=128,
                        shuffle=True, num_workers=0)  # train_data is a tuple: (batch_x, batch_y)
test_data = DataLoader(prepare_multi_sources_data.SSLR_Dataset(test_data_path), batch_size=128,
                       shuffle=True, num_workers=0)  # test_data is a tuple: (batch_x, batch_y)


# training cycle forward, backward, update
def train(epoch):
    running_loss = 0.0

    # Train
    iter = 0
    total_loss = 0.
    sam_size = 0.

    model.train()
    for (batch_x, batch_y, batch_z) in train_data:
        # 获得一个批次的数据和标签(inputs, labels)
        batch_x = batch_x.to(device) # batch_x.shape [B, 6, 40, 51]
        batch_y = batch_y.to(device) # batch_y.shape [B, 360]

        # 获得模型预测结果
        output = model(batch_x) # output.shape [B, 360]

        # 代价函数
        loss = criterion(output, batch_y) # averaged loss on batch_y

        running_loss += loss.item()
        if iter % 1000 == 0:
            print('[%d, %5d] loss: %.5f' % (epoch + 1, iter + 1, running_loss / 1000), flush=True)
            running_loss = 0.0
        with torch.no_grad():
            total_loss += loss.clone().detach().item() * batch_y.shape[0]
            sam_size += batch_y.shape[0]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 一个iter以一个batch为单位
        iter += 1
    # print the MSE and the sample size
    print(f'epoch {epoch + 1} loss {total_loss / sam_size} sam_size {sam_size}', flush=True)

# Evaluate
def test():
    cnt_acc_single = 0
    cnt_acc_multi = 0
    sum_err_single = 0
    sum_err_multi = 0
    total = 0
    total_single = 0
    total_multi = 0

    with torch.no_grad():
        model.eval()
        for (batch_x, batch_y, batch_z) in test_data:
            batch_x = batch_x.to(device) # batch_x.shape [B, 6, 40, 51]
            # batch_y = batch_y.to(device) # batch_y.shape [B, 360]

            # batch_y.shape[0] = batch_size
            total += batch_z.size(0)

            # 获得模型预测结果
            output = model(batch_x) # output.shape [B, 360]

            for batch in range(batch_z.size(0)):
                # test for known number of sources
                num_sources = batch_z[batch]

                if num_sources == 0:
                    total -= 1
                
                if num_sources == 1:
                    pred = torch.max(batch_y[batch], 0)[1].item()
                    label = torch.max(output[batch], 0)[1].item()
                    abs_err = func.angular_distance(pred, label)

                    if abs_err <= 5:
                        cnt_acc_single += 1
                    sum_err_single += abs_err
                    total_single += 1
                if num_sources == 2:
                    pred = func.get_top2_doa(output[batch])
                    label = np.where(batch_y[batch].numpy() == 1)[0]
                    error = func.angular_distance(pred.reshape([2, 1]), label.reshape([1, 2]))
                    if error[0, 0]+error[1, 1] <= error[1, 0]+error[0, 1]:
                        abs_err = np.array([error[0, 0], error[1, 1]])
                    else:
                        abs_err = np.array([error[0, 1], error[1, 0]])
                    cnt_acc_multi += np.sum(abs_err <= 5)
                    sum_err_multi += abs_err.sum()
                    total += 1
                    total_multi += 2
        cnt_acc = cnt_acc_single + cnt_acc_multi
        sum_err = sum_err_single + sum_err_multi
    print(f'total_single {total_single} total_multi {total_multi} total {total}')
    print('Single-source accuracy on test set: %.2f %% ' % (100.0 * cnt_acc_single / total_single), flush=True)
    print('Single-source MAE on test set: %.3f ' % (sum_err_single / total_single), flush=True)
    print('Two-sources accuracy on test set: %.2f %% ' % (100.0 * cnt_acc_multi / total_multi), flush=True)
    print('Two-sources MAE on test set: %.3f ' % (sum_err_multi / total_multi), flush=True)             
    print('Overall accuracy on test set: %.2f %% ' % (100.0 * cnt_acc / total), flush=True)
    print('Overall MAE on test set: %.3f ' % (sum_err / total), flush=True)


if __name__ == '__main__':
    for epoch in range(30):
        train(epoch)
        test()
