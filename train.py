import os
import argparse

import torch
import torch.optim as optim
from matplotlib import pyplot as plt

from data.genimage import SD14_read_split_data, read_val_dataset_GenImage
from models.model import SEP
from data.dataset import MyDataSet, MyValDataSet
from utils import create_lr_scheduler, get_params_groups, \
    train_one_epoch, evaluate
import random
import numpy as np


def set_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

# 使用matplotlib绘制损失和准确率曲线
def plot_metrics(all_losses, all_accuracies):
    # 绘制损失曲线
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(all_losses)), all_losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制准确率曲线
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(all_accuracies)), all_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_metrics_together(all_losses, all_accuracies):
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(len(all_losses)), all_losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(len(all_accuracies)), all_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

subVals = ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5',
               'VQDM', 'wukong', 'Chameleon']



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")


    train_images_path, train_images_label, _, _ = SD14_read_split_data("/root/lanyun-tmp/GenImage/stable_diffusion_v_1_4")

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label)

    print("-----------实例化训练数据集"+str(len(train_dataset))+"-----------")


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    model = SEP().to(device)


    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_avg_acc = 0.
    all_losses = []
    all_accuracies = []

    val_losses = []
    val_accuracies = []
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc, epoch_losses, epoch_accuracies = train_one_epoch(model=model,
                                                                                optimizer=optimizer,
                                                                                data_loader=train_loader,
                                                                                device=device,
                                                                                epoch=epoch,
                                                                                lr_scheduler=lr_scheduler)
        total_val_acc = 0.0
        for v_id, val in enumerate(subVals):

            val_path_1 = os.path.join('/root/lanyun-tmp/GenImage', val)
            val_path = os.path.join(val_path_1, 'val')

            val_images_path, val_images_label = read_val_dataset_GenImage(val_path)
            # 实例化验证数据集
            val_dataset = MyValDataSet(images_path=val_images_path,
                                    images_class=val_images_label)

            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=nw,
                                                     collate_fn=val_dataset.collate_fn)
            print("-----------实例化验证数据集" + str(val) + "===" + str(len(val_dataset)) + "-----------")

            # validate
            val_loss, val_acc, val_epoch_losses, val_epoch_accuracies = evaluate(model=model,
                                                                                 data_loader=val_loader,
                                                                                 device=device,
                                                                                 epoch=epoch,
                                                                                 val=val)
            print(f"Epoch {epoch}, @@{val} Validation Accuracy: {val_acc:.8f}")

            # 将验证结果和当前 epoch 一起保存到 txt 文件
            with open("./weights/val_all_dataset_log.txt", "a") as f:
                f.write(f"Epoch {epoch}, @@{val} Validation Accuracy: {val_acc:.8f}\n")

            # total_val_acc += val_acc

            if val != 'Chameleon':  # 如果当前验证集不是Chameleon
                total_val_acc += val_acc

        # 确保目录存在
        if not os.path.exists('./weights'):
            os.makedirs('./weights')

        avg_val_acc = total_val_acc / (len(subVals)-1)

        print(f"Epoch {epoch}, Average Validation Accuracy: {avg_val_acc:.8f}")

        # 将验证结果和当前 epoch 一起保存到 txt 文件
        with open("./weights/val_every_acc_log.txt", "a") as f:
            f.write("Epoch: {:03d} | New best val_acc: {:.4f}\n".format(epoch, avg_val_acc))

        if best_avg_acc < avg_val_acc:
            torch.save(model.state_dict(), "./weights/GenImage_SSP2_NoUnet512FINAL.pth")
            best_avg_acc = avg_val_acc
            print("---------------------------------------------------------------")
            print("Max val_acc:"+str(best_avg_acc))
            print("---------------------------------------------------------------")

            # 将验证结果和当前 epoch 一起保存到 txt 文件
            with open("./weights/val_best_acc_log.txt", "a") as f:
                f.write("Epoch: {:03d} | New best val_acc: {:.4f}\n".format(epoch, best_avg_acc))

        #统计损失值和准确率
        all_losses.extend(epoch_losses)
        all_accuracies.extend(epoch_accuracies)

        val_losses.extend(val_epoch_losses)
        val_accuracies.extend(val_epoch_accuracies)
    return all_losses, all_accuracies, val_losses, val_accuracies


if __name__ == '__main__':

    #设置随机数种子
    set_random_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.0)


    parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
    parser.add_argument('--patch_num', type=int, default=3,
                        help='scale images to this size, used in detection method RPTC')


    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default=r"E:\\MyProject\\IssueDB\\cifake")

    # 预训练权重路径，如果不想载入就设置为空字符
    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    all_losses, all_acc, val_losses, val_acc = main(opt)

    plot_metrics(all_losses, all_acc)
    plot_metrics_together(all_losses, all_acc)

    #绘制train和val的准确率在同一张图的曲线图
    plt.figure()
    plt.plot(all_acc,label='Training Accuracy')
    plt.plot(val_acc,label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # 绘制train和val的损失率在同一张图的曲线图
    plt.figure()
    plt.plot(all_losses, label='Training Accuracy')
    plt.plot(val_losses, label='Validation Accuracy')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.legend()
    plt.show()
