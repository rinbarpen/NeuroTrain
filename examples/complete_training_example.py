"""
NeuroTrain 完整训练示例

本示例展示了使用NeuroTrain进行完整的模型训练流程，包括：
1. 数据集准备
2. 模型创建
3. 训练配置
4. 模型训练
5. 模型测试
6. 结果分析
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import get_train_valid_test_dataloader
from src.models import get_model
from src.engine import Trainer, Tester
from src.utils.criterion import get_criterion
from src.utils import EarlyStopping
from src.metrics import dice, iou_seg, accuracy
import logging


def setup_logging(output_dir: Path):
    """设置日志"""
    log_file = output_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('training')


def main():
    """主训练流程"""
    
    # ===== 1. 配置 =====
    print("=" * 80)
    print("Step 1: Configuration")
    print("=" * 80)
    
    config = {
        'basic': {
            'task_name': 'CIFAR10_Classification',
            'run_id': 'example_training'
        },
        'dataset': {
            'name': 'cifar10',
            'root_dir': 'data/cifar10',
            'train': True,
            'download': True
        },
        'model': {
            'arch': 'resnet18',
            'pretrained': True,
            'n_classes': 10,
            'n_channels': 3
        },
        'training': {
            'epochs': 50,
            'batch_size': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'num_workers': 4
        }
    }
    
    # 输出目录
    output_dir = Path(f"runs/{config['basic']['run_id']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_dir / 'train')
    logger.info(f"Starting training: {config['basic']['task_name']}")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # ===== 2. 准备数据集 =====
    print("\n" + "=" * 80)
    print("Step 2: Preparing Dataset")
    print("=" * 80)
    
    train_loader, valid_loader, test_loader = get_train_valid_test_dataloader(config)
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(valid_loader) if valid_loader else 0}")
    logger.info(f"Test batches: {len(test_loader) if test_loader else 0}")
    
    # ===== 3. 创建模型 =====
    print("\n" + "=" * 80)
    print("Step 3: Creating Model")
    print("=" * 80)
    
    model = get_model('torchvision', config['model'])
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # ===== 4. 定义损失函数和优化器 =====
    print("\n" + "=" * 80)
    print("Step 4: Defining Loss and Optimizer")
    print("=" * 80)
    
    criterion = nn.CrossEntropyLoss()
    logger.info(f"Loss function: {criterion.__class__.__name__}")
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    logger.info(f"Scheduler: {scheduler.__class__.__name__}")
    
    # 早停
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4, mode='min')
    
    # ===== 5. 训练模型 =====
    print("\n" + "=" * 80)
    print("Step 5: Training Model")
    print("=" * 80)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['training']['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 打印进度
            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {100.*train_correct/train_total:.2f}%"
                )
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(avg_train_loss)
        
        logger.info(
            f"  Training   - Loss: {avg_train_loss:.4f}, "
            f"Accuracy: {train_acc:.2f}%"
        )
        
        # 验证阶段
        if valid_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            avg_val_loss = val_loss / len(valid_loader)
            val_acc = 100. * val_correct / val_total
            val_losses.append(avg_val_loss)
            
            logger.info(
                f"  Validation - Loss: {avg_val_loss:.4f}, "
                f"Accuracy: {val_acc:.2f}%"
            )
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = output_dir / 'checkpoints' / 'best.pth'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                    'config': config
                }, checkpoint_path)
                logger.info(f"  Saved best model to {checkpoint_path}")
            
            # 早停检查
            if early_stopping(avg_val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # 保存最新模型
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / 'checkpoints' / f'epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
            }, checkpoint_path)
    
    # ===== 6. 测试模型 =====
    print("\n" + "=" * 80)
    print("Step 6: Testing Model")
    print("=" * 80)
    
    # 加载最佳模型
    best_checkpoint = torch.load(output_dir / 'checkpoints' / 'best.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    logger.info("Loaded best model for testing")
    
    if test_loader:
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        logger.info(f"Test Accuracy: {test_acc:.2f}%")
    
    # ===== 7. 保存训练曲线 =====
    print("\n" + "=" * 80)
    print("Step 7: Saving Results")
    print("=" * 80)
    
    # 保存损失曲线数据
    import json
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'config': config
    }
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # 可视化（可选）
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plot_path = output_dir / 'loss_curves.png'
        plt.savefig(plot_path)
        logger.info(f"Loss curves saved to {plot_path}")
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualization")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if test_loader:
        print(f"Test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()

