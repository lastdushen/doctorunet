from unet_model import UNet
from dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
import time
import collections
import sys
import numpy as np


class Progbar(object):
    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)



def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    isbi_dataset = ISBI_Loader(data_path)
    #print(data_path)
    #print(isbi_dataset)
    #print(isbi_dataset[39])

    #isbi_dataset只是一个地址，比如0x00000295D2D5BF60，告诉图片存在哪
    #有了这个地址，之后就可以用isbi_dataset[0]等等，从中取出数据
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=False)
    #print(train_loader)比如0x00000295D2D5BF98


    #优化器带入的是unet中的参数net.parameters,之后会用optimizer.zero_grad()和optimizer.step()
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    for epoch in range(epochs):
        n = len(train_loader)
        pbar = Progbar(target=n)

        #train模式（net.train())和eval模式（net.eval())。一般的神经网络中，这两种模式是一样的，只有当模型中存在dropout和batchnorm的时候才有区别。
        net.train()

        #关于enumerate用法：
        #seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        #list(enumerate(seasons))
        #输出：[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
        for j, (image, label) in enumerate(train_loader):
            #如果不将梯度清零的话，梯度会与上一个batch的数据相关，因此该函数要写在反向传播和梯度下降之前
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            #print("##################################")
            pred = net(image)
            loss = criterion(pred, label)
            loss_val = round(float(loss.item()), 4)
            pbar.update(j + 1, values=[('epoch is ', epoch + 1), ('loss val', loss_val)])
            #print('epoch is ',epoch+1,' loss val',loss_val)
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), './model/best_model.pth')
            #使用loss.backward()后，会一层层的反向传播计算每个w的梯度值，并保存到该w的.grad属性中。
            loss.backward()
            #optimizer.step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值
            #因为梯度下降是基于梯度的，所以在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度
            optimizer.step()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    data_path = "./data/train/"
    train_net(net, device, data_path)