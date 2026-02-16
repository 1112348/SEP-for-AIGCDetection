"""
推理脚本 - 基于 train.py 进行最小改动
用于加载训练好的权重并在 GenImage 数据集上进行推理测试
"""
import os
import sys
import argparse

import torch
from tqdm import tqdm
import numpy as np
import csv
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, average_precision_score, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from data.genimage import read_val_dataset_GenImage
from models.model import SEP
from data.dataset import MyValDataSet
import random


def set_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)





# GenImage 数据集子集列表 (与 train.py 保持一致)
subVals = ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5',
           'VQDM', 'wukong']


@torch.no_grad()
def inference(model, data_loader, device, val_name, images_path_list=None):
    """
    推理函数 - 计算完整评估指标
    
    指标说明:
    - ROC-AUC: ROC 曲线下面积
    - EER: 等错误率 (Equal Error Rate)，即 FPR = FNR 时的错误率
    - Fake ACC (TPR): 将 Fake 视为正类时的真阳性率 (True Positive Rate)
    - Real ACC (TNR): 将 Real 视为负类时的真阴性率 (True Negative Rate = 1 - FPR)
    - mAP: 对于二分类问题，mAP 等价于 Average Precision (AP)
    - FPR@95%TPR: 当 TPR=95% 时的假阳性率
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        val_name: 验证集名称
        images_path_list: 可选，图像路径列表，用于跟踪错误分类的图像
    
    Returns:
        metrics: 包含所有指标的字典
        misclassified_info: 错误分类图像的信息列表 [(path, true_label, pred_label), ...]
    """
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    all_preds = []  # 用于存储所有批次的预测概率
    all_labels = []  # 用于存储所有批次的真实标签
    all_pred_classes = []  # 存储预测类别

    sample_num = 0
    batch_size = data_loader.batch_size
    data_loader_tqdm = tqdm(data_loader, file=sys.stdout, desc=f"[Inference {val_name}]")
    
    for step, data in enumerate(data_loader_tqdm):
        patch_imgs, dwt_imgs, labels = data
        current_batch_size = patch_imgs.shape[0]
        sample_num += current_batch_size

        # 前向推理
        pred = model(patch_imgs.to(device), dwt_imgs.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        # 收集当前批次的预测概率和标签
        all_preds.append(torch.softmax(pred, dim=1).detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_pred_classes.append(pred_classes.cpu().numpy())

        data_loader_tqdm.set_postfix({
            'loss': f'{accu_loss.item() / (step + 1):.3f}',
            'acc': f'{accu_num.item() / sample_num:.3f}'
        })

    # 计算基础指标
    avg_loss = accu_loss.item() / (step + 1)
    avg_accuracy = accu_num.item() / sample_num

    # 将所有批次的数据拼接
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_pred_classes = np.concatenate(all_pred_classes, axis=0)

    # 找出错误分类的图像
    misclassified_info = []
    if images_path_list is not None:
        misclassified_mask = all_labels != all_pred_classes
        misclassified_indices = np.where(misclassified_mask)[0]
        for idx in misclassified_indices:
            misclassified_info.append({
                'path': images_path_list[idx],
                'true_label': int(all_labels[idx]),
                'pred_label': int(all_pred_classes[idx])
            })

    # 对于二分类问题，选择正类 (Fake=1) 的概率
    # 假设: label 1 = Fake (AI生成), label 0 = Real (自然图像)
    prob_fake = all_preds[:, 1]  # Fake 类的概率
    pred_labels = all_pred_classes

    # 1. ROC-AUC
    roc_auc = roc_auc_score(all_labels, prob_fake)

    # 2. EER (Equal Error Rate)
    fpr, tpr, thresholds = roc_curve(all_labels, prob_fake)
    # EER 是 FPR = 1 - TPR 的点 (即 FPR = FNR)
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except ValueError:
        # 如果插值失败，使用近似方法
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    # 3. Fake ACC (TPR) - 真阳性率: TP / (TP + FN)
    # 4. Real ACC (TNR) - 真阴性率: TN / (TN + FP)
    # 使用混淆矩阵计算
    cm = confusion_matrix(all_labels, pred_labels)
    # cm[0][0] = TN, cm[0][1] = FP, cm[1][0] = FN, cm[1][1] = TP
    tn, fp, fn, tp = cm.ravel()
    
    fake_acc_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
    real_acc_tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR

    # 5. mAP (对于二分类问题，mAP 等价于 Average Precision)
    # AP = Average Precision，是 Precision-Recall 曲线下的面积
    mAP = average_precision_score(all_labels, prob_fake)

    # 6. FPR@95%TPR - 当 TPR >= 95% 时的最小 FPR
    target_tpr = 0.95
    # 找到 TPR >= 95% 的所有点中，FPR 最小的那个
    valid_indices = np.where(tpr >= target_tpr)[0]
    if len(valid_indices) > 0:
        fpr_at_95tpr = fpr[valid_indices[0]]  # 第一个达到 95% TPR 的点对应最小 FPR
    else:
        fpr_at_95tpr = 1.0  # 如果没有达到 95% TPR，则设为 1.0

    metrics = {
        'accuracy': avg_accuracy,
        'loss': avg_loss,
        'roc_auc': roc_auc,
        'eer': eer,
        'fake_acc_tpr': fake_acc_tpr,
        'real_acc_tnr': real_acc_tnr,
        'mAP': mAP,
        'fpr_at_95tpr': fpr_at_95tpr
    }

    return metrics, misclassified_info


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")



    # 创建模型
    model = SEP().to(device)

    # 加载训练好的权重
    print(f"正在加载模型权重: {args.weights}")
    assert os.path.exists(args.weights), f"权重文件不存在: {args.weights}"
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print("模型权重加载成功!")

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print(f'使用 {nw} 个 dataloader workers')

    # 存储所有结果
    results = []

    # 用于计算平均指标
    metric_sums = {
        'accuracy': 0.0,
        'roc_auc': 0.0,
        'eer': 0.0,
        'fake_acc_tpr': 0.0,
        'real_acc_tnr': 0.0,
        'mAP': 0.0,
        'fpr_at_95tpr': 0.0
    }
    count_val_sets = 0

    print("\n" + "=" * 80)
    print("=" * 80)
    print("开始推理测试 - 标准评估模式")
    print("=" * 80)

    for val_name in subVals:

        val_path_1 = os.path.join(args.data_root, val_name)
        val_path = os.path.join(val_path_1, 'val')

        if not os.path.exists(val_path):
            print(f"[警告] 跳过 {val_name}: 路径不存在 {val_path}")
            continue

        # 读取验证数据
        val_images_path, val_images_label = read_val_dataset_GenImage(val_path)
        
        if len(val_images_path) == 0:
            print(f"[警告] 跳过 {val_name}: 没有找到图像")
            continue

        val_dataset = MyValDataSet(images_path=val_images_path,
                                   images_class=val_images_label)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=val_dataset.collate_fn)
        
        print(f"\n--- {val_name} (共 {len(val_dataset)} 张图像) ---")

        # 推理 - 返回 metrics 字典和错误分类信息
        metrics, misclassified_info = inference(model=model,
                                                data_loader=val_loader,
                                                device=device,
                                                val_name=val_name,
                                                images_path_list=val_images_path)

        # 构建结果字典
        result = {
            'Generator': val_name,
            'Samples': len(val_dataset),
            'Accuracy': metrics['accuracy'],
            'ROC-AUC': metrics['roc_auc'],
            'EER': metrics['eer'],
            'Fake_ACC(TPR)': metrics['fake_acc_tpr'],
            'Real_ACC(TNR)': metrics['real_acc_tnr'],
            'mAP(=AP)': metrics['mAP'],
            'FPR@95%TPR': metrics['fpr_at_95tpr'],
            'Loss': metrics['loss'],
            'Misclassified': len(misclassified_info)
        }
        results.append(result)

        # 打印详细指标
        print(f"  ├─ Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  ├─ ROC-AUC:       {metrics['roc_auc']:.4f}")
        print(f"  ├─ EER:           {metrics['eer']:.4f}")
        print(f"  ├─ Fake ACC(TPR): {metrics['fake_acc_tpr']:.4f}")
        print(f"  ├─ Real ACC(TNR): {metrics['real_acc_tnr']:.4f}")
        print(f"  ├─ mAP (=AP):     {metrics['mAP']:.4f}")
        print(f"  ├─ FPR@95%TPR:    {metrics['fpr_at_95tpr']:.4f}")
        print(f"  └─ Loss:          {metrics['loss']:.4f}")

        # 累加用于计算平均值
        for key in metric_sums.keys():
            metric_sums[key] += metrics[key]
        count_val_sets += 1

    # 计算平均指标
    if count_val_sets > 0:
        avg_metrics = {key: val / count_val_sets for key, val in metric_sums.items()}
    else:
        avg_metrics = metric_sums

    # 打印总结
    print("\n" + "=" * 80)
    print("推理结果总结")
    print("=" * 80)

    if count_val_sets > 0:
        print(f"\n平均指标 (共 {count_val_sets} 个生成器):")
        print(f"  ├─ Accuracy:      {avg_metrics['accuracy']:.4f}")
        print(f"  ├─ ROC-AUC:       {avg_metrics['roc_auc']:.4f}")
        print(f"  ├─ EER:           {avg_metrics['eer']:.4f}")
        print(f"  ├─ Fake ACC(TPR): {avg_metrics['fake_acc_tpr']:.4f}")
        print(f"  ├─ Real ACC(TNR): {avg_metrics['real_acc_tnr']:.4f}")
        print(f"  ├─ mAP (=AP):     {avg_metrics['mAP']:.4f}")
        print(f"  └─ FPR@95%TPR:    {avg_metrics['fpr_at_95tpr']:.4f}")

    # 保存结果到 CSV 文件
    if args.save_results:
        # 确定 CSV 输出目录
        weights_dir = os.path.dirname(args.weights)
        if not weights_dir:
            weights_dir = '.'
        csv_file = os.path.join(weights_dir, 'inference_results.csv')
        
        # CSV 列名
        fieldnames = ['Generator', 'Samples', 'Accuracy', 'ROC-AUC', 'EER', 
                      'Fake_ACC(TPR)', 'Real_ACC(TNR)', 'mAP(=AP)', 'FPR@95%TPR', 'Loss', 'Misclassified']
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # 写入每个生成器的结果
            for r in results:
                writer.writerow(r)
            
            # 写入平均值行
            if count_val_sets > 0:
                avg_row = {
                    'Generator': 'MEAN',
                    'Samples': sum(r['Samples'] for r in results),
                    'Accuracy': avg_metrics['accuracy'],
                    'ROC-AUC': avg_metrics['roc_auc'],
                    'EER': avg_metrics['eer'],
                    'Fake_ACC(TPR)': avg_metrics['fake_acc_tpr'],
                    'Real_ACC(TNR)': avg_metrics['real_acc_tnr'],
                    'mAP(=AP)': avg_metrics['mAP'],
                    'FPR@95%TPR': avg_metrics['fpr_at_95tpr'],
                    'Loss': sum(r['Loss'] for r in results) / count_val_sets,
                    'Misclassified': sum(r['Misclassified'] for r in results)
                }
                writer.writerow(avg_row)
        
        print(f"\n结果已保存到 CSV 文件: {csv_file}")

        # 同时打印表格形式的结果
        print("\n各生成器详细结果:")
        print("-" * 120)
        header = f"{'Generator':<25} {'Acc':>8} {'AUC':>8} {'EER':>8} {'TPR':>8} {'TNR':>8} {'mAP':>8} {'FPR@95':>8}"
        print(header)
        print("-" * 120)
        for r in results:
            row = f"{r['Generator']:<25} {r['Accuracy']:>8.4f} {r['ROC-AUC']:>8.4f} {r['EER']:>8.4f} {r['Fake_ACC(TPR)']:>8.4f} {r['Real_ACC(TNR)']:>8.4f} {r['mAP(=AP)']:>8.4f} {r['FPR@95%TPR']:>8.4f}"
            print(row)
        print("-" * 120)
        if count_val_sets > 0:
            mean_row = f"{'MEAN':<25} {avg_metrics['accuracy']:>8.4f} {avg_metrics['roc_auc']:>8.4f} {avg_metrics['eer']:>8.4f} {avg_metrics['fake_acc_tpr']:>8.4f} {avg_metrics['real_acc_tnr']:>8.4f} {avg_metrics['mAP']:>8.4f} {avg_metrics['fpr_at_95tpr']:>8.4f}"
            print(mean_row)
        print("-" * 120)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/model.pth', help='模型权重路径')
    parser.add_argument('--data-root', type=str, default='/root/lanyun-tmp/GenImage', help='GenImage 数据集根目录')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    parser.add_argument('--save-results', action='store_true', default=True, help='是否保存结果到 CSV')
    parser.add_argument('--save-misclassified', action='store_true', default=False, help='是否保存错误分类的图像')
    
    args = parser.parse_args()
    
    main(args)
