import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_semantic_features(seg, num_semantics, label_nc, ignored_value=255):
    # extract semantic segmentation map
    seg = seg.clone()
    seg[seg > 999] //=  1000
    if ignored_value is not None and label_nc is not None:
        seg[seg == ignored_value] = label_nc
    # extract semantic conditioning code
    semantic_cond = torch.zeros(num_semantics)
    unique, counts = torch.unique(seg.flatten(), return_counts=True)
    semantic_cond[unique.long()] = counts.float()
    semantic_cond /= torch.sum(semantic_cond)
    # preprocess semantic segmentation map
    seg_mc = torch.zeros([num_semantics, *seg.shape])
    seg_mc = seg_mc.scatter_(0, seg.unsqueeze(0).long(), 1.0)
    return seg_mc, semantic_cond

if __name__=='__main__':
    label_path = "datasets/rock/test_label_ppl/a2_a8-.png"
    seg_img = Image.open(label_path)
    seg_img = seg_img.convert("L")  # 保证是单通道
    seg_tensor = torch.tensor(np.array(seg_img), dtype=torch.long)  # 转为Tensor

    # 设置类别数，比如你有 5 类
    num_semantics = 5
    label_nc = 0  # 把 255 当成哪一类？如无效设成 None
    seg_mc, sem_cond = get_semantic_features(seg_tensor, num_semantics, label_nc)

    # 打印比例信息
    print(f"[Semantic Condition Vector] sem_cond (length={len(sem_cond)}):")
    for i, val in enumerate(sem_cond):
        print(f"  类别 {i}: {val.item():.4f}")

    # 可视化：原始标签图 & 每个类别的 mask
    plt.figure(figsize=(10, 2 + num_semantics))
    plt.subplot(1, num_semantics + 1, 1)
    plt.imshow(seg_tensor, cmap='tab20')
    plt.title("原始标签图")
    plt.axis("off")

    for i in range(num_semantics):
        plt.subplot(1, num_semantics + 1, i + 2)
        plt.imshow(seg_mc[i].numpy(), cmap='gray')
        plt.title(f"类别 {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()