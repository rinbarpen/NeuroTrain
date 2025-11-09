#!/bin/bash
# COCO数据集下载脚本
#
# 使用方法:
#   bash scripts/download_coco.sh [year] [split]
#
# 参数:
#   year: 数据集年份 (默认: 2017)
#   split: 数据集划分 (train, val, test, all) (默认: all)
#
# 示例:
#   bash scripts/download_coco.sh 2017 train
#   bash scripts/download_coco.sh 2017 all

set -e

# 默认参数
YEAR=${1:-2017}
SPLIT=${2:-all}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}COCO ${YEAR} 数据集下载脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查是否需要开启代理
echo -e "${YELLOW}提示: 如果下载速度慢，请先运行 proxy_on 开启代理${NC}"
read -p "按Enter键继续..."

# 创建目录
DATA_DIR="data/coco"
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

echo -e "${GREEN}数据将保存到: $(pwd)${NC}"

# 下载函数（支持断点续传）
download_file() {
    local url=$1
    local filename=$(basename "${url}")
    
    # 检查文件是否已存在
    if [ -f "${filename}" ]; then
        local local_size=$(stat -f%z "${filename}" 2>/dev/null || stat -c%s "${filename}" 2>/dev/null)
        echo -e "${YELLOW}文件已存在: ${filename} (${local_size} bytes)${NC}"
        
        # 尝试获取远程文件大小进行完整性检查
        local remote_size=$(curl -sI "${url}" | grep -i "content-length" | awk '{print $2}' | tr -d '\r')
        if [ -n "${remote_size}" ] && [ "${local_size}" -eq "${remote_size}" ] 2>/dev/null; then
            echo -e "${GREEN}✓ 文件已完整，跳过下载: ${filename}${NC}"
            return 0
        else
            echo -e "${YELLOW}文件大小不匹配，将从断点继续下载: ${filename}${NC}"
        fi
    fi
    
    echo -e "${GREEN}下载: ${filename}${NC}"
    # 使用 wget -c 支持断点续传，-t 0 表示无限重试，--progress=bar:force 显示进度条
    if wget -c -t 0 --progress=bar:force:noscroll "${url}"; then
        echo -e "${GREEN}✓ 下载完成: ${filename}${NC}"
        return 0
    else
        echo -e "${RED}✗ 下载失败: ${filename}${NC}"
        return 1
    fi
}

# 解压函数
unzip_file() {
    local filename=$1
    
    if [ ! -f "${filename}" ]; then
        echo -e "${RED}文件不存在: ${filename}${NC}"
        return 1
    fi
    
    echo -e "${GREEN}解压: ${filename}${NC}"
    if unzip -q "${filename}"; then
        echo -e "${GREEN}✓ 解压完成: ${filename}${NC}"
        return 0
    else
        echo -e "${RED}✗ 解压失败: ${filename}${NC}"
        return 1
    fi
}

# 基础URL
BASE_URL="http://images.cocodataset.org"

# 下载训练集
if [ "${SPLIT}" == "train" ] || [ "${SPLIT}" == "all" ]; then
    echo -e "\n${GREEN}[1/3] 下载训练集图像...${NC}"
    download_file "${BASE_URL}/zips/train${YEAR}.zip"
fi

# 下载验证集
if [ "${SPLIT}" == "val" ] || [ "${SPLIT}" == "all" ]; then
    echo -e "\n${GREEN}[2/3] 下载验证集图像...${NC}"
    download_file "${BASE_URL}/zips/val${YEAR}.zip"
fi

# 下载测试集
if [ "${SPLIT}" == "test" ]; then
    echo -e "\n${GREEN}[2/3] 下载测试集图像...${NC}"
    download_file "${BASE_URL}/zips/test${YEAR}.zip"
fi

# 下载标注文件
echo -e "\n${GREEN}[3/3] 下载标注文件...${NC}"
download_file "${BASE_URL}/annotations/annotations_trainval${YEAR}.zip"

# 解压文件
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}开始解压文件...${NC}"
echo -e "${GREEN}========================================${NC}"

if [ "${SPLIT}" == "train" ] || [ "${SPLIT}" == "all" ]; then
    if [ -f "train${YEAR}.zip" ]; then
        unzip_file "train${YEAR}.zip"
    fi
fi

if [ "${SPLIT}" == "val" ] || [ "${SPLIT}" == "all" ]; then
    if [ -f "val${YEAR}.zip" ]; then
        unzip_file "val${YEAR}.zip"
    fi
fi

if [ "${SPLIT}" == "test" ]; then
    if [ -f "test${YEAR}.zip" ]; then
        unzip_file "test${YEAR}.zip"
    fi
fi

if [ -f "annotations_trainval${YEAR}.zip" ]; then
    unzip_file "annotations_trainval${YEAR}.zip"
fi

# 清理zip文件
echo -e "\n${YELLOW}是否删除zip文件以节省空间? (y/N)${NC}"
read -p "" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}清理zip文件...${NC}"
    rm -f *.zip
    echo -e "${GREEN}✓ 清理完成${NC}"
fi

# 显示目录结构
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}下载完成！${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${GREEN}目录结构:${NC}"
tree -L 2 -d . || ls -lh

# 统计信息
echo -e "\n${GREEN}数据集统计:${NC}"
if [ -d "train${YEAR}" ]; then
    train_count=$(ls "train${YEAR}" | wc -l)
    echo -e "训练集图像数: ${train_count}"
fi

if [ -d "val${YEAR}" ]; then
    val_count=$(ls "val${YEAR}" | wc -l)
    echo -e "验证集图像数: ${val_count}"
fi

if [ -d "test${YEAR}" ]; then
    test_count=$(ls "test${YEAR}" | wc -l)
    echo -e "测试集图像数: ${test_count}"
fi

if [ -d "annotations" ]; then
    echo -e "标注文件:"
    ls -lh annotations/*.json
fi

echo -e "\n${GREEN}数据集下载和解压完成！${NC}"
echo -e "${GREEN}现在可以使用以下命令测试:${NC}"
echo -e "  python examples/coco_example.py"

