import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from inception import Inception, InceptionBlock

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROPOUT_RATE = 0.5

print(f"Using {DEVICE} device")

# for attention
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# for time series data (TCN)
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=DROPOUT_RATE):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# Inception time
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=20, dropout=DROPOUT_RATE):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y

# Attention structure
class ATT_inception_time(nn.Module):
    def __init__(self, input_size, output_size):
        super(ATT_inception_time, self).__init__()
        self.causal_conv = weight_norm(nn.Conv1d(3, 9, kernel_size=812, stride=1, padding=811, dilation=1))
        self.chomp_att = Chomp1d(811)
        self.downsample_att = nn.Conv1d(9, 3, 1)
        self.local_att = nn.Sequential(self.causal_conv, self.chomp_att, self.downsample_att)
        self.conv1d = nn.Conv1d(3, 32, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.InceptionTime = nn.Sequential(
            InceptionBlock(in_channels=32, n_filters=8, kernel_sizes=[5,11,23], bottleneck_channels=8, use_residual=True, activation=nn.ReLU()),
            InceptionBlock(in_channels=32, n_filters=16, kernel_sizes=[5,11,23], bottleneck_channels=16, use_residual=True, activation=nn.ReLU()),
            nn.AdaptiveAvgPool1d(output_size=812)
        )
        self.conv1 = nn.Conv1d(64, 32, kernel_size=1)
        self.tdd = TimeDistributed(nn.Linear(32,16), batch_first=True)
        self.tcn = TemporalConvNet(16, [128,128,128,128,128,128], kernel_size=4, dropout=0.0)
        self.downsample = nn.Conv1d(128, 2, 1)
        self.output_linear = nn.Linear(144, 72)
        self.output_linear2 = nn.Linear(72, 36)
        self.output_linear3 = nn.Linear(36, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, feature):
        att = self.local_att(inputs)
        att = F.sigmoid(att)
        inputs = inputs * att
        inputs = self.relu(self.conv1d(att))
        inception_output = self.InceptionTime(inputs)
        inception_output = self.conv1(inception_output)
        tdd_output = self.tdd(inception_output.transpose(2, 1)).transpose(2, 1)
        tcn_output = self.tcn(tdd_output)
        output = self.downsample(tcn_output)

        feature = feature.to(torch.float32).reshape(-1, 16, 812)
        pre_output = torch.cat((tcn_output, feature), dim=1)
        self.intermediate_output = pre_output

        output = self.output_linear(pre_output.transpose(2, 1))
        output = self.output_linear2(output)
        output = self.output_linear3(output)
        output = self.sigmoid(output.transpose(2, 1))
        return output, att