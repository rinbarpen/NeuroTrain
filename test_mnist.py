from pathlib import Path
from src.dataset.mnist_dataset import MNISTDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# 创建数据集
base_dir = Path("./data/mnist")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 测试训练集
train_dataset = MNISTDataset.get_train_dataset(
    base_dir=base_dir, 
    transforms=transform,
    download=True
)

print(f"训练集大小: {len(train_dataset)}")

# 测试数据加载
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
for batch_idx, (data, target) in enumerate(train_loader):
    print(f"Batch {batch_idx}: data shape = {data.shape}, target shape = {target.shape}")
    if batch_idx >= 2:  # 只测试前几个batch
        break

print("MNIST数据集集成成功！")