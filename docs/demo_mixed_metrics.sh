#!/bin/bash
# 演示混合指标方向功能

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "混合指标方向功能演示"
echo "=========================================="
echo

echo "场景1: 所有指标越高越好（标准分类任务）"
echo "--------------------------------------"
cat > /tmp/demo1.csv << EOF
model,accuracy,precision,recall,f1_score
ModelA,0.95,0.94,0.96,0.95
ModelB,0.93,0.92,0.94,0.93
OurModel,0.97,0.96,0.98,0.97
ModelC,0.91,0.90,0.92,0.91
EOF

conda run -n ntrain python tools/data_to_latex.py \
  -i /tmp/demo1.csv -t table --style booktabs \
  --highlight-best --highlight-second \
  --metric-columns accuracy precision recall f1_score \
  --higher-is-better True True True True \
  --our-model "OurModel" 2>&1 | tail -20
echo

echo "场景2: 性能与效率权衡"
echo "--------------------------------------"
cat > /tmp/demo2.csv << EOF
model,accuracy,inference_time,memory_MB
SlowModel,0.97,200,1024
FastModel,0.91,45,256
OurModel,0.95,80,512
LightModel,0.89,35,128
EOF

conda run -n ntrain python tools/data_to_latex.py \
  -i /tmp/demo2.csv -t table --style booktabs \
  --highlight-best --highlight-second \
  --metric-columns accuracy inference_time memory_MB \
  --higher-is-better True False False \
  --our-model "OurModel" 2>&1 | tail -20
echo

echo "场景3: 训练指标（accuracy高，loss低）"
echo "--------------------------------------"
cat > /tmp/demo3.csv << EOF
model,train_acc,train_loss,val_acc,val_loss
ModelA,0.95,0.12,0.93,0.15
ModelB,0.93,0.15,0.91,0.18
OurModel,0.97,0.08,0.96,0.10
ModelC,0.94,0.11,0.92,0.14
EOF

conda run -n ntrain python tools/data_to_latex.py \
  -i /tmp/demo3.csv -t table --style booktabs \
  --highlight-best --highlight-second \
  --metric-columns train_acc train_loss val_acc val_loss \
  --higher-is-better True False True False \
  --our-model "OurModel" 2>&1 | tail -20
echo

echo "场景4: 完整评估（多种指标混合）"
echo "--------------------------------------"
echo "accuracy↑ loss↓ error_rate↓ inference_time↓ params↓"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/test_mixed_metrics.csv -t table --style booktabs \
  --highlight-best --highlight-second \
  --metric-columns accuracy loss error_rate inference_time \
  --higher-is-better True False False False \
  --our-model "OurModel" 2>&1 | tail -20
echo

echo "=========================================="
echo "演示完成！"
echo "=========================================="
echo
echo "关键点："
echo "  ✓ True = 越高越好（accuracy, precision等）"
echo "  ✓ False = 越低越好（loss, time等）"
echo "  ✓ 顺序必须与--metric-columns对应"
echo
echo "详细文档: tools/MIXED_METRICS_GUIDE.md"

