
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_FNN(nn.Module):
    def __init__(self,J_num,O_max_len):
        super(CNN_FNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, ceil_mode=False)
        )
        self.fc1 = nn.Linear(6 * int(J_num / 2) * int(O_max_len / 2), 258)
        self.fc2 = nn.Linear(258, 258)
        self.out = nn.Linear(258, 17)
        # self.fc1 = nn.Linear(6 * int(J_num / 2) * int(O_max_len / 2), 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.out = nn.Linear(128, 17)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = F.relu(self.fc3(x))
        # x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class CNN_dueling(nn.Module):
    def __init__(self,J_num,O_max_len):
        super(CNN_dueling, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=6,
                      kernel_size=3,
                      stride=1,
                      padding=1,),
            nn.ReLU(),nn.MaxPool2d(kernel_size=2,ceil_mode=False)
        )

        self.val_hidden=nn.Linear(6*int(J_num/2)*int(O_max_len/2),258)
        self.adv_hidden=nn.Linear(6*int(J_num/2)*int(O_max_len/2),258)
        self.val=nn.Linear(258,1)
        self.adv=nn.Linear(258,17)

    def forward(self,x):
        x=self.conv1(x)
        x = x.view(x.size(0), -1)

        val_hidden=self.val_hidden(x)
        val_hidden=F.relu(val_hidden)

        adv_hidden=self.adv_hidden(x)
        adv_hidden=F.relu(adv_hidden)

        val=self.val(val_hidden)
        adv=self.adv(adv_hidden)

        adv_ave=torch.mean(adv,dim=1,keepdim=True)
        x=adv+val-adv_ave

        return x

    # def __init__(self, J_num, O_max_len):
    #     super(CNN_dueling, self).__init__()
    #     self.conv1 = nn.Sequential(
    #                 nn.Conv2d(in_channels=3,
    #                           out_channels=6,
    #                           kernel_size=3,
    #                           stride=1,
    #                           padding=1,),
    #                 nn.ReLU(),nn.MaxPool2d(kernel_size=2,ceil_mode=False)
    #             )
    #
    #     self.val_hidden=nn.Linear(6*int(J_num/2)*int(O_max_len/2),258)
    #     self.adv_hidden=nn.Linear(6*int(J_num/2)*int(O_max_len/2),258)
    #     self.val=nn.Linear(258,1)
    #     self.adv=nn.Linear(258,17)
    #
    #     self.val_hidden = nn.Linear(6 * int(J_num / 2) * int(O_max_len / 2), 512)
    #     self.adv_hidden = nn.Linear(6 * int(J_num / 2) * int(O_max_len / 2), 512)
    #
    #     self.val_hidden2 = nn.Linear(512, 256)
    #     self.adv_hidden2 = nn.Linear(512, 256)
    #
    #     self.val = nn.Linear(256, 1)
    #     self.adv = nn.Linear(256, 17)
    #
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = x.view(x.size(0), -1)
    #
    #     val_hidden = F.relu(self.val_hidden(x))
    #     adv_hidden = F.relu(self.adv_hidden(x))
    #
    #     val_hidden = F.relu(self.val_hidden2(val_hidden))
    #     adv_hidden = F.relu(self.adv_hidden2(adv_hidden))
    #
    #     val = self.val(val_hidden)
    #     adv = self.adv(adv_hidden)
    #
    #     adv_ave = torch.mean(adv, dim=1, keepdim=True)
    #     x = adv + val - adv_ave
    #
    #     return x







