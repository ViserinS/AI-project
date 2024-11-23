import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os
import argparse
import datetime
import math
import numpy as np
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup

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

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):  # 增加embed_dim
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4  # 增加MLP比率
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio),
                     drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dim=384, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0.1):  # 增加默认dropout
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 添加一个额外的MLP头
        self.pre_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.pre_head(x[:, 0])  # 使用额外的MLP头
        x = self.head(x)
        return x

class CIFAR100WithTransform(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        if isinstance(img, torch.Tensor):
            to_pil = transforms.ToPILImage()
            img = to_pil(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.base_dataset)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.best_acc = 0
        self.train_step = 0
        
        self.checkpoint_dir = 'cifar100/checkpoints_CIFAR100_flip'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(f'runs/cifar100_vit_flip{run_id}')
        
        self._setup_model()
        self._setup_data()
        self._setup_training()
        
        if args.resume:
            self._load_checkpoint(args.resume)

    def _setup_model(self):
        self.model = VisionTransformer(
            img_size=32,
            patch_size=4,
            embed_dim=192,  # 增加嵌入维度 384
            depth=7,       # 增加深度 12
            num_heads=8,   # 增加注意力头数 12
            mlp_ratio=4.,
            num_classes=100,
            drop_rate=0.1,
            attn_drop_rate=0.1
        ).to(self.device)
        
        n_parameters = sum(p.numel() for p in self.model.parameters())
        print(f'Number of parameters: {n_parameters:,}')

    def _setup_data(self):
        # 增强的数据增强策略
        # base: crop,flip,randaugment,colorjitter,erasing
        #

        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandAugment(num_ops=2, magnitude=9),  # 添加RandAugment
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761)),
            # transforms.RandomErasing(p=0.25)  # 添加随机擦除
            # AddGaussianNoise(std=0.05),
            # AddSaltPepperNoise(probability=0.05)
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761))
        ])
        
        # 设置Mixup
        self.mixup = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=0.5,
            switch_prob=0.5,
            mode='batch',
            num_classes=100
        )
        
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        
        self.trainloader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        self.testloader = DataLoader(
            testset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.steps_per_epoch = len(self.trainloader)

    def _setup_training(self):
        # 使用带有warmup的AdamW优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=0.05,  # 增加权重衰减
            betas=(0.9, 0.999)
        )
        
        # 设置学习率调度器
        total_steps = self.steps_per_epoch * self.args.epochs
        warmup_steps = int(total_steps * 0.1)  # 10% warmup
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * float(step - warmup_steps) / float(total_steps - warmup_steps)))
            
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # 使用标签平滑的交叉熵损失
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.trainloader, desc=f'Epoch {epoch}/{self.args.epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 应用mixup
            if self.args.mixup:
                inputs, targets_a, targets_b, lam = self.mixup(inputs, targets)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            if self.args.mixup:
                loss = self.criterion(outputs, targets_a) * lam + \
                       self.criterion(outputs, targets_b) * (1 - lam)
            else:
                loss = self.criterion(outputs, targets)
                
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()

            train_loss += loss.item()
            if not self.args.mixup:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            current_lr = self.scheduler.get_last_lr()[0]
            
            pbar.set_postfix({
                'Loss': f'{train_loss/total:.3f}',
                'Acc': f'{100.*correct/total:.2f}%' if not self.args.mixup else 'N/A',
                'LR': f'{current_lr:.6f}'
            })
            
            self.writer.add_scalar('Train/Loss', loss.item(), self.train_step)
            if not self.args.mixup:
                self.writer.add_scalar('Train/Accuracy', 100.*correct/total, self.train_step)
            self.writer.add_scalar('Train/Learning_Rate', current_lr, self.train_step)
            
            self.train_step += 1

        return train_loss/total, 100.*correct/total if not self.args.mixup else 0

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        class_correct = [0] * 100
        class_total = [0] * 100
        
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
                
                # 计算每个类别的准确率
                c = predicted.eq(targets).cpu()
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{test_loss/total:.3f}',
                    'Acc': f'{current_acc:.2f}%'
                })

        # 记录总体指标
        self.writer.add_scalar('Test/Loss', test_loss/total, epoch)
        self.writer.add_scalar('Test/Accuracy', current_acc, epoch)
        
        # 记录每个类别的准确率
        for i in range(100):
            class_acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            self.writer.add_scalar(f'Test/Class_{i}_Accuracy', class_acc, epoch)
        
        return test_loss/total, current_acc

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f'vit_cifar100_checkpoint_epoch_{epoch}.pth')
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'train_step': self.train_step
        }
        
        torch.save(state, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'vit_cifar100_model_best.pth')
            torch.save(state, best_path)
            print(f"Saved best model to {best_path}")

    def _load_checkpoint(self, checkpoint_path):
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

    def train(self):
        start_time = time.time()
        
        try:
            for epoch in range(self.start_epoch, self.args.epochs):
                # 训练和测试
                train_loss, train_acc = self.train_epoch(epoch)
                test_loss, test_acc = self.test(epoch)
                
                # 打印epoch总结
                print(f'\nEpoch {epoch}:')
                print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
                print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
                
                # 保存最佳模型
                is_best = test_acc > self.best_acc
                if is_best:
                    self.best_acc = test_acc
                    print(f'New best accuracy: {self.best_acc:.2f}%')
                    self._save_checkpoint(epoch, is_best=True)
                
                # 定期保存检查点
                if (epoch + 1) % self.args.save_freq == 0:
                    self._save_checkpoint(epoch)
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            self._save_checkpoint(epoch)
        
        training_time = (time.time() - start_time) / 3600  # 转换为小时
        print(f'\nTraining completed in {training_time:.2f} hours')
        print(f'Best accuracy: {self.best_acc:.2f}%')
        self.writer.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='',
                        help='path to latest checkpoint')
    parser.add_argument('--epochs', type=int, default=200,  # 增加训练轮数
                        help='number of total epochs to run')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='save checkpoint every n epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--mixup', action='store_true',
                        help='use mixup and cutmix for training')
    return parser.parse_args()

def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()