import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os
import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='',
                        help='path to latest checkpoint')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run')
    parser.add_argument('--save-freq', type=int, default=20,
                        help='save checkpoint every n epochs')
    return parser.parse_args()

class AddGaussianNoise:
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class AddSaltPepperNoise:
    def __init__(self, probability=0.05):
        self.probability = probability
        
    def __call__(self, tensor):
        noise_tensor = torch.rand(tensor.size())
        salt = (noise_tensor < self.probability/2).float()
        pepper = (noise_tensor > (1 - self.probability/2)).float()

        
        noisy_tensor = tensor.clone()
        noisy_tensor[salt > 0] = 1
        noisy_tensor[pepper > 0] = 0
        return noisy_tensor

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """保存检查点"""
    torch.save(state, filename)
    if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'resnet18_model_best.pth')
            torch.save(state, best_path)
            print(f"Saved best model to {best_path}")

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.best_acc = 0
        self.train_step = 0
        
        # 创建保存目录
        self.checkpoint_dir = 'cifar100/checkpoints_only_colorjitter'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 设置tensorboard
        run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(f'runs/cifar100_only_colorjitter{run_id}')
        
        self._setup_model()
        self._setup_data()
        self._setup_training()
        
        if args.resume:
            self._load_checkpoint(args.resume)

    def _setup_model(self):
        """设置模型"""
        self.model = resnet18(pretrained=False)
        # 修改第一层卷积层以匹配CIFAR的图像大小
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        # 修改最后的全连接层以输出100个类别
        self.model.fc = nn.Linear(self.model.fc.in_features, 100)
        self.model = self.model.to(self.device)

    def _setup_data(self):
        """设置数据加载器"""
        
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        self.testloader = DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2)

    def _setup_training(self):
        """设置训练相关组件"""
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1,
                                 momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs)

    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
        self.train_step = checkpoint['train_step']
        
        print(f"Resumed from epoch {self.start_epoch}")
        print(f"Best accuracy so far: {self.best_acc:.2f}%")

    def _save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'train_step': self.train_step
        }
        
        save_checkpoint(state, is_best, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.trainloader, desc=f'Epoch {epoch}/{self.args.epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            current_loss = train_loss / (total / targets.size(0))
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{current_loss:.3f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # 记录到tensorboard
            self.writer.add_scalar('Train/Loss', current_loss, self.train_step)
            self.writer.add_scalar('Train/Accuracy', current_acc, self.train_step)
            self.writer.add_scalar('Train/Learning_Rate', 
                                 self.scheduler.get_last_lr()[0], self.train_step)
            
            self.train_step += 1

        return current_loss, current_acc

    def test(self, epoch):
        """测试模型"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.testloader, desc='Testing')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                current_loss = test_loss / (total / targets.size(0))
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{current_loss:.3f}',
                    'Acc': f'{current_acc:.2f}%'
                })

        # 记录到tensorboard
        self.writer.add_scalar('Test/Loss', current_loss, epoch)
        self.writer.add_scalar('Test/Accuracy', current_acc, epoch)
        
        return current_loss, current_acc

    def train(self):
        """完整训练流程"""
        start_time = time.time()
        
        try:
            for epoch in range(self.start_epoch, self.args.epochs):
                # 训练和测试
                train_loss, train_acc = self.train_epoch(epoch)
                test_loss, test_acc = self.test(epoch)
                self.scheduler.step()
                
                # 打印epoch总结
                print(f'\nEpoch {epoch}:')
                print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
                print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
                
                # 检查是否是最佳模型
                is_best = test_acc > self.best_acc
                if is_best:
                    self.best_acc = test_acc
                    self._save_checkpoint(epoch, is_best=True)
                
                # 每N个epoch保存一次
                if (epoch + 1) % self.args.save_freq == 0:
                    self._save_checkpoint(epoch)
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            self._save_checkpoint(epoch)
        
        print(f'\nTraining completed in {(time.time()-start_time)/60:.2f} minutes')
        print(f'Best accuracy: {self.best_acc:.2f}%')
        self.writer.close()

def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()