# 图像盲水印与抗合谋追踪系统

一个基于 **Tardos 指纹编码 + DWT-DCT 频域嵌入 + QIM 量化调制** 的图像盲水印项目，支持：

- 多用户唯一指纹分发
- 用户身份与密钥绑定
- 盲提取（提取时不需要原图）
- 多人合谋攻击模拟
- 合谋嫌疑用户排序与追踪
- 命令行运行
- Flask 前端可视化操作

---

## 1. 项目目标

对于同一张原始图片，系统会为不同用户生成不同的指纹码字，并将其嵌入到图像中，得到多个“看起来几乎相同，但内部指纹不同”的分发版本。

当发生泄露时，系统可以：

1. 从泄露图中盲提取指纹比特串
2. 模拟 average / XOR-like 合谋攻击
3. 基于 Tardos 对所有用户进行打分
4. 输出最可疑用户的排序结果

---

## 2. 算法组成

### 2.1 Tardos 指纹编码

- 为每个用户生成一段二进制指纹码
- 每个用户的码字与 `master_key + user_id + user_key` 绑定
- 用于后续追踪与抗合谋打分

### 2.2 DWT-DCT 频域嵌入

- 先对图像做 DWT（离散小波变换）
- 再对选中的子带块做 DCT（离散余弦变换）
- 在频域中选择指定系数嵌入指纹 bit

### 2.3 QIM 量化嵌入

- bit=0 与 bit=1 对应两组不同的量化格点
- 提取时无需原图，只需重新定位嵌入位置并做最近格点判决
- 支持盲提取

### 2.4 抗合谋追踪

当前项目提供两种合谋模拟：

- `average_collusion`：对多个带指纹的用户版本求平均
- `xor_collusion_via_reembed`：提取多个用户版本的 bit 后进行 XOR 融合，并重新嵌入

最后通过 Tardos 对所有用户进行打分排序，判断最可疑的泄露者。

---

## 3. 项目结构

目录结构如下：

```text
pic-watermark/
├─ app.py                  # Flask 前端入口
├─ demo.py                 # 命令行 demo 与主实验流程
├─ tardos.py               # Tardos 码本生成与嫌疑人打分
├─ watermark.py            # DWT-DCT + QIM 盲水印算法
├─ templates/              # Flask 模板
│  ├─ index.html
│  ├─ users_view.html
│  ├─ collusion_view.html
│  └─ report.html
├─ static/
├─ data/
│  └─ users.json           # 用户列表
├─ image/
│  ├─ test1.png
│  └─ test2.png
├─ results/                # 默认输出目录（可以自己改）
└─ README.md
```

---

## 4. 环境依赖

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install numpy opencv-python matplotlib PyWavelets flask
```

如果你的环境是 conda，也可以先激活环境后再安装。

---

## 5. 命令行运行方式

### 5.1 使用真实图片运行

```bash
python demo.py \
  --image .\image\test1.png \
  --num-users 8 \
  --code-length 320 \
  --colluders user_2,user_6,user_8 \
  --delta 26 \
  --repeats 6 \
  --out-dir results_t1
```
可以适当减小参数防止图像没位置嵌入。
> 注意：`code_length * repeats` 不能超过当前图像可用嵌入容量，否则会报 `Not enough capacity`。

### 5.2 参数说明

| 参数 | 含义 |
|---|---|
| `--image` | 输入图片路径|
| `--out-dir` | 输出目录 |
| `--num-users` | 用户数量 |
| `--code-length` | Tardos 指纹码长度 |
| `--colluders` | 合谋用户列表，逗号分隔 |
| `--delta` | QIM 量化步长 |
| `--repeats` | 每个逻辑 bit 的重复嵌入次数 |
| `--master-key` | 系统主密钥 |
| `--content-id` | 内容编号，用于位置重建 |
| `--seed` | Tardos 概率向量随机种子 |
| `--cover-size` | 自动合成测试图的尺寸 |

---

## 6. 前端运行方式

本项目提供 Flask 前端，可通过网页完成用户管理、参数填写、运行实验和查看结果。

启动方式：

```bash
python app.py
```

启动后默认访问：

```text
http://127.0.0.1:5000
```

### 6.1 前端功能

- 首页填写实验参数并运行水印流程
- 添加 / 删除用户
- 自动将用户信息保存到 `data/users.json`
- 查看用户水印图与提取结果
- 查看合谋攻击结果图
- 查看 `report.json` 报告内容

---

## 7. 输出结果说明

运行完成后，输出目录中通常会包含以下文件：

### 7.1 原图与用户版图像

- `cover.png`：输入原图
- `user_i_watermarked.png`：第 i 个用户对应的带水印版本

### 7.2 提取结果

- `user_i_extracted_bits.png`：从第 i 个用户版本中盲提取得到的 bit 可视化图

### 7.3 合谋图像

- `pirate_average.png`：平均合谋生成图
- `pirate_average_bits.png`：从平均合谋图中提取得到的 bit 图
- `pirate_xor.png`：XOR-like 合谋生成图
- `pirate_xor_bits.png`：从 XOR-like 合谋图中提取得到的 bit 图

### 7.4 分析结果

- `scores.png`：嫌疑用户打分柱状图
- `diff_first_user.png`：原图与第一个用户水印图的差分图
- `report.json`：完整实验报告

---

## 8. `report.json` 字段说明

### `config`

记录本次实验的参数：

- 输入图片路径
- 用户数
- 码长
- 合谋用户
- `delta`
- `repeats`
- `content_id`
- `master_key`
- `seed`

### `clean_ber`

记录每个用户版本在**无攻击场景**下的盲提取误码率（BER）。

- 越接近 0 越好
- `0.0` 表示提取结果和嵌入码字完全一致

### `psnr`

记录每个用户版本相对于原图的图像质量指标。

- 越高越好
- 一般 40 dB 以上说明视觉质量较好

### `average_attack`

记录 average 合谋攻击下的追踪结果：

- `top5`：得分最高的前 5 个用户
- `top3_hit_count`：前 3 名中命中的真实合谋者数量

### `xor_attack`

记录 XOR-like 合谋攻击下的追踪结果：

- `top5`：得分最高的前 5 个用户
- `top3_hit_count`：前 3 名中命中的真实合谋者数量

---

## 9. 前端用户管理说明

前端用户列表保存在：

```text
data/users.json
```

推荐每个用户都具有：

- 唯一的 `user_id`
- 唯一的 `user_key`

例如：

```json
[
  {"user_id": "user_1", "user_key": "k_9f2A1x7P"},
  {"user_id": "user_2", "user_key": "k_B8mQ3d2L"}
]
```

这样能够保证：

- 每个用户对应不同的指纹码字
- 多用户实验与合谋追踪更稳定
