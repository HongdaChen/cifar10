from __future__ import division, print_function

import argparse

import torch
import torch.nn.functional as F
from torch import distributed, nn
from torch.utils import data
from torchvision import datasets, transforms

from utils.radam import RAdam, RAdam_4step, AdamW     # 用于添加算法的外部模块

from tensorboardX import SummaryWriter
writer=SummaryWriter('./summary/cifar10')

import models                                         # 用于选择网络结构的外部模块
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# 分布式初始化工具函数
# from torch import distributed
def distributed_is_initialized():       
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False

# 求一个epoch错误率的工具类
class Average(object):
	# 给对象赋值
    def __init__(self):
        self.accumulated_batch_loss = 0
        self.count = 0
    # 打印对象，返回的就是这个函数的返回值
    def __str__(self):
        return '{:.6f}'.format(self.average)
    # 方法属性化，可以通过对象.average直接得到方法的返回值，不需.average()
    @property
    def average(self):
        return self.accumulated_batch_loss / self.count
    # 每个batch结束后更新计数，使用该方法
    def update(self, value, number):
        self.accumulated_batch_loss += value * number
        self.count += number

# 求一个epoch总准确率的工具类
class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        return self.correct / self.count

    def update(self, output, target):
    	# 不需要梯度信息
        with torch.no_grad():
        	# 网络的输出是对应每个类的概率，概率最大的认为是该类
            pred = output.argmax(dim=1)
            # 概率最大的索引和标签相等的计数就是准确率的计数
            correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)

# 训练类
class Trainer(object):

    def __init__(self, model, optimizer, train_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def fit(self, epochs, optimizer):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()

            # 打印在终端
            print(
                'Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {}.'.format(test_loss, test_acc),
            )
            # 使用tensorboard工具，终端输入命令：tensorboard --logdir={SummaryWriter指定的路径} --host={ip,此项不写默认本地回环地址127.0.0.1}
            writer.add_scalars('train_loss', {optimizer:train_loss.average}, epoch)
            writer.add_scalars('test_loss', {optimizer:test_loss.average}, epoch)
            writer.add_scalars('train_acc', {optimizer:train_acc.accuracy}, epoch)
            writer.add_scalars('test_acc', {optimizer:test_acc.accuracy}, epoch)




    def train(self):
    	# 训练模式
        self.model.train()
        # 错误率类和准确率类实例化
        train_loss = Average()
        train_acc = Accuracy()
        # 按batch加载数据
        for data, target in self.train_loader:
        	# 将数据转移到对应平台
            data = data.to(self.device)
            target = target.to(self.device)
            # 输入数据到模型中
            output = self.model(data)
            # 模型输出与标签对比，损失函数指定cross_entropy()
            loss = F.cross_entropy(output, target)
            # 梯度清零
            self.optimizer.zero_grad()
            # 反向传播计算每层参数的梯度
            loss.backward()
            # 对每层参数更新
            self.optimizer.step()
            # 将一个batch的错误率和准确率累加在之前的基础上
            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, target)
        # 注意这个函数返回的是对象，并不是具体的数值，需要.accuracy()或.average()调出对应数值
        return train_loss, train_acc

    def evaluate(self):
    	# 验证模式
        self.model.eval()

        test_loss = Average()
        test_acc = Accuracy()
        # 不需要梯度信息
        with torch.no_grad():
        	# 按batch加载数据
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = F.cross_entropy(output, target)

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, target)

        return test_loss, test_acc

# 加载图象数据
# 继承自 from torch.utils import data
class CIFAR10DataLoader(data.DataLoader):
    def __init__(self, root, batch_size, train=True):
    	# from torchvision import transforms
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # from torchvision import datasets     
        dataset = datasets.CIFAR10(root, train=train, transform=transform,download=True)
        # 分布式
        sampler = None
        if train and distributed_is_initialized():
            sampler = data.DistributedSampler(dataset) 
        super(CIFAR10DataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
        )

# run
def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # 模型选择
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=10,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=10,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=10,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=10,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=10)

    # 转移数据 
    # from torch import nn
    if distributed_is_initialized():
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    else:
        model = nn.DataParallel(model)
        model.to(device)

    # 选择算法    
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'radam':
        optimizer = RAdam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # 加载数据
    train_loader = CIFAR10DataLoader(args.root, args.batch_size, train=True)
    test_loader = CIFAR10DataLoader(args.root, args.batch_size, train=False)

    # 训练类实例化
    trainer = Trainer(model, optimizer, train_loader, test_loader, device)
    trainer.fit(args.epochs,args.optimizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
    parser.add_argument(
                        '-i',
                        '--init-method',
                        type=str,
                        default='tcp://127.0.0.1:23456',
                        help='URL specifying how to initialize the package.')
    parser.add_argument('-s', '--world-size', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', type=int, default=0, help='Rank of the current process.')
    #optim
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['adamw', 'radam', 'radam4s', 'sgd','adam'])
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='beta2 for adam')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup steps for adam')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    #architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--block-name', type=str, default='BasicBlock',
                        help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
    parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
    
    args = parser.parse_args()

    print(args)

    if args.world_size > 1:
        distributed.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.rank,
        )

    run(args)


if __name__ == '__main__':
    main()
    writer.close()
