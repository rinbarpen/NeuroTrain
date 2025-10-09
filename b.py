from tools.analyzers.dataset_analyzer import DatasetAnalyzer
import numpy as np

# 定义MNIST数据集的标签提取器
def mnist_label_extractor(sample):
    """从MNIST数据集样本中提取标签"""
    if isinstance(sample, dict) and 'metadata' in sample:
        return np.array([sample['metadata']['label']])
    elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
        # 兼容标准的(image, label)格式
        return np.array([sample[1]])
    return None

# 定义MNIST数据集的图像提取器
def mnist_image_extractor(sample):
    """从MNIST数据集样本中提取图像"""
    if isinstance(sample, dict) and 'image' in sample:
        image = sample['image']
        if hasattr(image, 'numpy'):
            return image.numpy()
        return np.array(image)
    elif isinstance(sample, (tuple, list)) and len(sample) >= 1:
        # 兼容标准的(image, label)格式
        image = sample[0]
        if hasattr(image, 'numpy'):
            return image.numpy()
        return np.array(image)
    return None

# 分析NeuroTrain内置数据集
# 根据MNIST配置文件和dataset.py的要求，正确配置参数
config = {
    'dataset_name': 'mnist',  # 使用小写，与_get_dataset_by_case匹配
    'base_dir': './data/mnist',  # 必需的base_dir参数
    'download': True,  # MNIST数据集配置
    'is_rgb': False    # MNIST是灰度图像
}

# 创建分析器实例，传递自定义提取器
analyzer = DatasetAnalyzer(
    dataset_name='mnist',
    dataset_config=config,
    output_dir='output/analysis',
    label_extractor=mnist_label_extractor,  # 自定义标签提取器
    image_extractor=mnist_image_extractor   # 自定义图像提取器
)

# 加载数据集
analyzer.load_datasets()

# 运行完整分析
results = analyzer.run_full_analysis(splits=['train', 'test'])

print(f"分析完成！结果保存在: {results['output_directory']}")