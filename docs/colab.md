# Colab 使用说明

在 Google Colab 中运行 NeuroTrain 的 train / test / predict，可按以下步骤操作。

## 1. 打开笔记本

- 将项目克隆到 Colab 环境（或上传 `colab.ipynb` 及所需代码）。
- 用 Colab 打开 `colab.ipynb`。

## 2. 环境准备（第一个单元格）

- **GPU**：菜单 **运行时 → 更改运行时类型**，将硬件加速器选为 **GPU**。
- **依赖**：取消注释并执行 `!pip install -e .` 或 `!uv pip install -e .`。
- **工作目录**：若克隆到 `/content/NeuroTrain`，可取消注释 `%cd /content/NeuroTrain`。
- **默认配置**：在 Colab 下会自动设置 `sys.argv`（如 `configs/single/train-mnist.toml`），可按需修改该单元格中的 `--config` 和参数。

## 3. 运行顺序

1. **Colab setup**：运行第一个单元格（依赖、GPU、`sys.argv`）。
2. **Drive（可选）**：若需把输出写到 Google Drive，运行 `drive.mount('/content/drive')`。
3. **Import**：运行导入单元格。
4. **parse_args()**：运行解析参数并加载配置。
5. **set config**：可在此修改 `c = get_config()` 后的配置（如 `output_dir`、`input_size`）。
6. **train | test | predict**：运行主流程单元格，执行训练 / 测试 / 预测。

## 4. 输出目录

- 默认使用配置中的 `output_dir`。
- 若已挂载 Drive，可在「set config」中设置例如：`c["output_dir"] = "/content/drive/MyDrive/NeuroTrain/runs"`，以便结果持久化。

## 5. 继续训练（Continue Mode）

- 在配置中设置 `continue_checkpoint` 与 `continue_ext_checkpoint`（或通过 `--continue_id` 自动选用上次 run）。
- 笔记本中的逻辑会加载模型与扩展状态，并从 `last_epoch` 继续训练。
