import torch
import torch.nn as nn

class Se(nn.Module):
    def __init__(self,in_channel,reduction=16):
        super(Se, self).__init__()
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=self.fc(out.view(out.size(0),-1))
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class MDR(nn.Module):
    def __init__(self, channel):
        super(MDR, self).__init__()

        self.s3 = Conv2D(channel, channel, kernel_size=1, padding=0, act=False)

        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True)
        )
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True)
        )
        self.conv7_1 = nn.Sequential(
            nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=3, dilation=3, bias=True),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=4, dilation=4, bias=True),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True)
        )

        self.conv3_3 = nn.Sequential(
            nn.Conv2d(channel , channel , kernel_size=3, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        self.se = Se(channel)

    def forward(self, x):
        s = x

        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        # print(x0.size())
        x1 = self.conv5_1(xc[1] + x0 + xc[2])
        # print(x1.size())
        x2 = self.conv7_1(xc[2] + x1 + xc[3])
        # print(x2.size())
        x3 = self.conv(xc[3] + x2)
        # print(x3.size())
        x = torch.cat((x0, x1, x2, x3), dim=1)

        x = self.relu(x + s)

        x = self.conv3_3(x)

        return self.se(x)


if __name__ == "__main__":
    net = MDR(128)
    input = torch.rand(3,128,128,128)
    output = net(input)
    print(output.size())