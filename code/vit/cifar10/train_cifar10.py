import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os
import argparse
import datetime
import math

class PatchEmbed(nn.Module):
    """将图像分割成patch并进行线性投影"""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class Attention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
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
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
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
    """Transformer块"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
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
    """Vision Transformer主模型"""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 embed_dim=192, depth=9, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.):
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
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化
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
        x = self.head(x[:, 0])
        return x

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.best_acc = 0
        self.train_step = 0
        
        # 创建保存目录
        self.checkpoint_dir = 'checkpoints_only_HorizontalFlip'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 设置tensorboard
        run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(f'runs/cifar10_vit_only_HorizontalFlip{run_id}')
        
        self._setup_model()
        self._setup_data()
        self._setup_training()
        
        if args.resume:
            self._load_checkpoint(args.resume)

    def _setup_model(self):
        """设置模型"""
        self.model = VisionTransformer(
            img_size=32,
            patch_size=4,
            embed_dim=192,
            depth=7,
            num_heads=8,
            mlp_ratio=2.,
            num_classes=10,
            drop_rate=0.1,
            attn_drop_rate=0.1
        ).to(self.device)
        
        # 计算参数量
        n_parameters = sum(p.numel() for p in self.model.parameters())
        print(f'Number of parameters: {n_parameters:,}')

    def _setup_data(self):
        """设置数据加载器"""
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        self.testloader = DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2)

    def _setup_training(self):
        """设置训练相关组件"""
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=0.05
        )
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
            self.checkpoint_dir, f'vit_checkpoint_epoch_{epoch}.pth')
        
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
            best_path = os.path.join(self.checkpoint_dir, 'vit_model_best.pth')
            torch.save(state, best_path)
            print(f"Saved best model to {best_path}")

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='',
                        help='path to latest checkpoint')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run')
    parser.add_argument('--save-freq', type=int, default=20,
                        help='save checkpoint every n epochs')
    return parser.parse_args()

def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()