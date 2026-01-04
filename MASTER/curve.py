import re
import matplotlib.pyplot as plt

# ==== 1. 读取日志文件 ====
log_file = 'seed+step.txt'

# 定义要提取的数据
epochs, train_loss, valid_ic, icir, rankic, rankicir = [], [], [], [], [], []

# ==== 2. 解析日志 ====
def safe_float(s):
    """安全转换字符串为浮点数，去掉末尾的句号"""
    s = s.strip()
    if s.endswith('.'):
        s = s[:-1]
    return float(s)

pattern = re.compile(
    r"Epoch\s+(\d+),\s*train_loss\s+([\d\.]+),\s*valid ic\s+([\d\.\-]+),\s*icir\s+([\d\.\-]+),\s*rankic\s+([\d\.\-]+),\s*rankicir\s+([\d\.\-]+)"
)

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_loss.append(safe_float(match.group(2)))
            valid_ic.append(safe_float(match.group(3)))
            icir.append(safe_float(match.group(4)))
            rankic.append(safe_float(match.group(5)))
            rankicir.append(safe_float(match.group(6)))

# ==== 3. 找出四个最佳 Epoch ====
best_ic_epoch = epochs[valid_ic.index(max(valid_ic))]
best_ic_val = max(valid_ic)

best_icir_epoch = epochs[icir.index(max(icir))]
best_icir_val = max(icir)

best_rankic_epoch = epochs[rankic.index(max(rankic))]
best_rankic_val = max(rankic)

best_rankicir_epoch = epochs[rankicir.index(max(rankicir))]
best_rankicir_val = max(rankicir)

# ==== 4. 绘图 ====
plt.figure(figsize=(12, 8))

# (1) 训练损失
plt.subplot(2, 1, 1)
plt.plot(epochs, train_loss, label='Train Loss', color='tab:blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)

# (2) 验证指标
plt.subplot(2, 1, 2)
plt.plot(epochs, valid_ic, label='Valid IC', linewidth=2)
plt.plot(epochs, icir, label='ICIR', linestyle='--')
plt.plot(epochs, rankic, label='RankIC', linestyle=':')
plt.plot(epochs, rankicir, label='RankICIR', linestyle='-.')

# 标注最佳点
plt.scatter(best_ic_epoch, best_ic_val, color='red', zorder=5)
plt.annotate(f'Best IC={best_ic_val:.4f}\n(Epoch {best_ic_epoch})',
             xy=(best_ic_epoch, best_ic_val),
             xytext=(best_ic_epoch+3, best_ic_val+0.002),
             arrowprops=dict(facecolor='red', arrowstyle='->'),
             fontsize=9, color='red')

plt.scatter(best_icir_epoch, best_icir_val, color='purple', zorder=5)
plt.annotate(f'Best ICIR={best_icir_val:.3f}\n(Epoch {best_icir_epoch})',
             xy=(best_icir_epoch, best_icir_val),
             xytext=(best_icir_epoch+3, best_icir_val+0.002),
             arrowprops=dict(facecolor='purple', arrowstyle='->'),
             fontsize=9, color='purple')

plt.scatter(best_rankic_epoch, best_rankic_val, color='orange', zorder=5)
plt.annotate(f'Best RankIC={best_rankic_val:.4f}\n(Epoch {best_rankic_epoch})',
             xy=(best_rankic_epoch, best_rankic_val),
             xytext=(best_rankic_epoch+3, best_rankic_val+0.002),
             arrowprops=dict(facecolor='orange', arrowstyle='->'),
             fontsize=9, color='orange')

plt.scatter(best_rankicir_epoch, best_rankicir_val, color='green', zorder=5)
plt.annotate(f'Best RankICIR={best_rankicir_val:.3f}\n(Epoch {best_rankicir_epoch})',
             xy=(best_rankicir_epoch, best_rankicir_val),
             xytext=(best_rankicir_epoch+3, best_rankicir_val+0.002),
             arrowprops=dict(facecolor='green', arrowstyle='->'),
             fontsize=9, color='green')

plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Validation Metrics over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图像
plt.savefig('training_metrics_final.png', dpi=300)
# plt.show()

# 控制台输出
print("最佳指标汇总：")
print(f"   - Valid IC: Epoch {best_ic_epoch}, IC = {best_ic_val:.4f}")
print(f"   - ICIR: Epoch {best_icir_epoch}, ICIR = {best_icir_val:.3f}")
print(f"   - RankIC: Epoch {best_rankic_epoch}, RankIC = {best_rankic_val:.4f}")
print(f"   - RankICIR: Epoch {best_rankicir_epoch}, RankICIR = {best_rankicir_val:.3f}")
