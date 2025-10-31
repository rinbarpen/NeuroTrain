"""
数据集查询工具

这个工具允许你：
1. 查看所有可用的数据集
2. 按任务类型查询数据集
3. 查看特定数据集的详细信息
4. 列出所有支持的任务类型
"""

import sys
from pathlib import Path
import argparse

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.dataset import (
    get_datasets_by_task, 
    get_all_task_types, 
    get_dataset_info,
    list_all_datasets,
    DATASET_REGISTRY
)


def print_header(text: str, char: str = '=', width: int = 70):
    """打印标题"""
    print()
    print(char * width)
    print(text.center(width))
    print(char * width)
    print()


def print_section(text: str, char: str = '-', width: int = 70):
    """打印小节标题"""
    print()
    print(text)
    print(char * width)


def list_tasks():
    """列出所有任务类型"""
    print_header("所有支持的任务类型")
    
    tasks = get_all_task_types()
    
    # 按类别分组显示
    task_categories = {
        '分类任务': ['classification', 'image_classification', 'digit_recognition', 
                   'skin_lesion_classification'],
        '分割任务': ['segmentation', 'semantic_segmentation', 'instance_segmentation', 
                   '3d_segmentation', 'retinal_vessel_segmentation', 
                   'skin_lesion_segmentation', 'cell_segmentation'],
        '检测任务': ['detection', 'keypoint_detection'],
        '多模态任务': ['multimodal', 'vqa', 'captioning'],
        '医学影像': ['medical_imaging'],
        '推理任务': ['reasoning'],
    }
    
    categorized = set()
    for category, category_tasks in task_categories.items():
        category_found = [t for t in tasks if t in category_tasks]
        if category_found:
            print(f"\n【{category}】")
            for task in category_found:
                # 统计该任务的数据集数量
                datasets = get_datasets_by_task(task)
                print(f"  • {task:<40} ({len(datasets)} 个数据集)")
                categorized.update(category_found)
    
    # 显示未分类的任务
    uncategorized = [t for t in tasks if t not in categorized]
    if uncategorized:
        print(f"\n【其他任务】")
        for task in uncategorized:
            datasets = get_datasets_by_task(task)
            print(f"  • {task:<40} ({len(datasets)} 个数据集)")
    
    print(f"\n总计: {len(tasks)} 种任务类型")


def list_datasets_by_task(task_type: str):
    """按任务类型列出数据集"""
    print_header(f'任务类型: {task_type}')
    
    datasets = get_datasets_by_task(task_type)
    
    if not datasets:
        print(f"未找到支持 '{task_type}' 任务的数据集。")
        print("\n提示: 使用 --list-tasks 查看所有可用的任务类型")
        return
    
    print(f"找到 {len(datasets)} 个数据集支持 '{task_type}' 任务:\n")
    
    for i, ds in enumerate(datasets, 1):
        print(f"{i}. {ds['name']}")
        print(f"   类名: {ds['class']}")
        print(f"   任务类型: {', '.join(ds['task_types'])}")
        print(f"   描述: {ds['description']}")
        if len(ds['aliases']) > 1:
            print(f"   别名: {', '.join(ds['aliases'])}")
        print()


def show_dataset_info(dataset_name: str):
    """显示特定数据集的详细信息"""
    print_header(f'数据集信息: {dataset_name}')
    
    info = get_dataset_info(dataset_name)
    
    if not info:
        print(f"未找到数据集 '{dataset_name}'。")
        print("\n提示: 使用 --list-all 查看所有可用的数据集")
        return
    
    print(f"名称: {info['name']}")
    print(f"类名: {info['class']}")
    print(f"模块: {info['module']}")
    print(f"任务类型: {', '.join(info['task_types'])}")
    print(f"描述: {info['description']}")
    print(f"别名: {', '.join(info['aliases'])}")
    print()


def list_all(sort_by: str = 'name'):
    """列出所有数据集"""
    if sort_by == 'task':
        print_header("按任务类型分组的数据集")
        
        datasets_by_task = list_all_datasets(sort_by='task')
        
        for task, datasets in datasets_by_task.items():
            print_section(f"任务类型: {task}")
            for ds in datasets:
                print(f"  • {ds['name']:<35} - {ds['description']}")
        
        print(f"\n总计: {len(DATASET_REGISTRY)} 个数据集")
    
    else:
        print_header("所有可用数据集")
        
        datasets = list_all_datasets(sort_by='name')['all']
        
        print(f"{'序号':<6} {'数据集名称':<35} {'任务类型'}")
        print('-' * 100)
        
        for i, ds in enumerate(datasets, 1):
            tasks = ', '.join(ds['task_types'][:2])
            if len(ds['task_types']) > 2:
                tasks += f" (+{len(ds['task_types'])-2})"
            print(f"{i:<6} {ds['name']:<35} {tasks}")
        
        print('-' * 100)
        print(f"总计: {len(datasets)} 个数据集")


def show_statistics():
    """显示数据集统计信息"""
    print_header("数据集统计信息")
    
    total_datasets = len(DATASET_REGISTRY)
    total_tasks = len(get_all_task_types())
    
    # 统计各类任务的数据集数量
    task_counts = {}
    for task in get_all_task_types():
        task_counts[task] = len(get_datasets_by_task(task))
    
    print(f"📊 总数据集: {total_datasets}")
    print(f"📊 总任务类型: {total_tasks}")
    print()
    
    print_section("热门任务类型 (按数据集数量排序)")
    sorted_tasks = sorted(task_counts.items(), key=lambda x: x[1], reverse=True)
    
    for i, (task, count) in enumerate(sorted_tasks[:10], 1):
        bar = '█' * min(count, 50)
        print(f"{i:2}. {task:<35} {bar} {count}")
    
    print()
    
    # 统计医学影像数据集
    medical_datasets = get_datasets_by_task('medical_imaging')
    print(f"🏥 医学影像数据集: {len(medical_datasets)}")
    
    # 统计分割任务数据集
    seg_datasets = get_datasets_by_task('segmentation')
    print(f"🎯 分割任务数据集: {len(seg_datasets)}")
    
    # 统计分类任务数据集
    cls_datasets = get_datasets_by_task('classification')
    print(f"📁 分类任务数据集: {len(cls_datasets)}")


def main():
    parser = argparse.ArgumentParser(
        description='数据集查询工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/list_datasets.py --list-all                 # 列出所有数据集
  python tools/list_datasets.py --list-tasks               # 列出所有任务类型
  python tools/list_datasets.py --task classification      # 查询分类任务的数据集
  python tools/list_datasets.py --info imagenet            # 查看 ImageNet 数据集信息
  python tools/list_datasets.py --statistics               # 显示统计信息
  python tools/list_datasets.py --list-all --sort task     # 按任务类型分组显示
        """
    )
    
    parser.add_argument('--list-all', action='store_true',
                       help='列出所有可用的数据集')
    parser.add_argument('--list-tasks', action='store_true',
                       help='列出所有支持的任务类型')
    parser.add_argument('--task', type=str,
                       help='按任务类型查询数据集 (如: classification, segmentation)')
    parser.add_argument('--info', type=str,
                       help='查看特定数据集的详细信息 (数据集名称或别名)')
    parser.add_argument('--sort', type=str, choices=['name', 'task'], default='name',
                       help='列表排序方式 (name: 按名称, task: 按任务类型)')
    parser.add_argument('--statistics', action='store_true',
                       help='显示数据集统计信息')
    
    args = parser.parse_args()
    
    # 如果没有参数，显示帮助信息
    if len(sys.argv) == 1:
        parser.print_help()
        print()
        show_statistics()
        return
    
    # 执行相应的操作
    if args.statistics:
        show_statistics()
    
    if args.list_all:
        list_all(sort_by=args.sort)
    
    if args.list_tasks:
        list_tasks()
    
    if args.task:
        list_datasets_by_task(args.task)
    
    if args.info:
        show_dataset_info(args.info)


if __name__ == '__main__':
    main()

