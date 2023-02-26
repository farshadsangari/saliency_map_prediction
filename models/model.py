

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,48,(7,7),padding=3)        
        self.LayerNorm = nn.LocalResponseNorm(5,0.001,0.75)
        self.pool = nn.MaxPool2d((3,3) , stride = 2)
        self.conv2 = nn.Conv2d(48,128,(5,5),padding=2)
        self.conv3 = nn.Conv2d(128,256,(3,3),padding=1)
        self.conv4 = nn.Conv2d(256,256,(5,5),padding=2)
        self.conv5 = nn.Conv2d(256,256,(5,5),padding=2)
        self.conv6 = nn.Conv2d(256,128,(7,7),padding=3)
        self.conv7 = nn.Conv2d(128,64,(11,11),padding=5)
        self.conv8 = nn.Conv2d(64,16,(11,11),padding=5)
        self.conv9 = nn.Conv2d(16,1,(13,13),padding=6)
        self.deconv1 = nn.ConvTranspose2d(1,1,(8,8),stride=4,padding=2)

        ####################   Weight Initialization  #####################
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv6.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv7.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv8.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv9.weight, nonlinearity='relu')
        nn.init.normal_(self.deconv1.weight,0,0.00001)
        
        
    def forward(self, x):
        #print(x.shape)
        x = self.pool(self.LayerNorm(F.relu(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.deconv1(x)
        return x