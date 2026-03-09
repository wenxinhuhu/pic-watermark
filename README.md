# 端到端抗合谋盲水印 Demo

Demo满足：

- 不使用 DNN / 深度学习
- 多用户绑定：每个用户的码字由 `(master_key, user_id, user_key)` 共同决定
- 抗合谋：支持平均合谋攻击与 XOR-like 合谋攻击演示
- 端到端：从 cover 图像 -> 多用户发放副本 -> 合谋图像 -> 盲提取 -> Tardos 打分追踪
- 盲验证：提取时不需要原图

## 方案结构

1. **Tardos 指纹码**
   - 为系统生成一条全局 bias 向量 `p`
   - 每个用户用 HMAC(master_key, user_id || user_key) 作为随机种子
   - 按 `p` 采样得到用户唯一二值码字

2. **DWT + DCT + QIM 嵌入**
   - 将图像转到 Y 通道
   - 做一级 DWT
   - 在 `LL` 和 `LH` 子带内选取若干 8x8 block
   - 对每个 block 做 DCT，在中频系数 `(3,3)` 上做二元 QIM
   - 每个逻辑 bit 重复嵌入多次并多数投票，增强盲提取鲁棒性

3. **盲提取**
   - 从待检图像按相同 key 与 content_id 重建位置
   - 对对应 DCT 系数做 QIM 判决
   - 多次重复位多数投票得到估计码字

4. **追踪**
   - 用 Tardos symmetric score 对所有用户打分
   - 得分越高，嫌疑越大

## 目录

- `tardos.py`：Tardos bias、用户码字生成、嫌疑人打分
- `watermark.py`：DWT-DCT-QIM 盲水印嵌入/提取与合谋攻击模拟
- `demo.py`：完整 demo
- `results/`：跑 demo 后的结果图与 `report.json`

## 配置

- 用户数：8
- Tardos 码长：320 bits
- 重复嵌入：6 次/bit
- QIM 步长：26.0
- 嵌入子带：LL + LH
- 合谋用户：user_2, user_5, user_7
