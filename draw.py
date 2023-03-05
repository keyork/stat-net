import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def draw(path_mlp, path_stat):

    batch_gap = 20

    raw_data_mlp = np.load(path_mlp)
    train_loss_list_mlp = raw_data_mlp["train_loss"]
    valid_loss_list_mlp = raw_data_mlp["valid_loss"]
    train_acc_list_mlp = raw_data_mlp["train_acc"]
    valid_acc_list_mlp = raw_data_mlp["valid_acc"]
    valid_time_list_mlp = raw_data_mlp["valid_x"]
    test_loss_mlp = raw_data_mlp["test_loss"]
    test_acc_mlp = raw_data_mlp["test_acc"]

    raw_data_stat = np.load(path_stat)
    train_loss_list_stat = raw_data_stat["train_loss"]
    valid_loss_list_stat = raw_data_stat["valid_loss"]
    train_acc_list_stat = raw_data_stat["train_acc"]
    valid_acc_list_stat = raw_data_stat["valid_acc"]
    valid_time_list_stat = raw_data_stat["valid_x"]
    test_loss_stat = raw_data_stat["test_loss"]
    test_acc_stat = raw_data_stat["test_acc"]

    x_list = np.arange(0, len(train_loss_list_stat) * batch_gap, batch_gap)

    """
    画图
        1. statnet和mlpnet的train_loss变化对比
        2. statnet和mlpnet的train_acc变化对比
        3. statnet和mlpnet的valid_loss变化对比
        4. statnet和mlpnet的valid_acc变化对比
        5. 随着训练数据的增加, statnet和mlpnet的test_acc变化对比
    """

    plt.figure(dpi=300)

    plt.plot(x_list, train_loss_list_stat, "", color="r")
    plt.plot(x_list, train_loss_list_mlp, "", color="b")

    label = ["StatNet", "MLPNet"]

    plt.legend(label, loc="upper right")
    plt.title("Train Loss Curve")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.savefig("./训练集loss曲线图.jpg")
    plt.cla()

    plt.plot(x_list, train_acc_list_stat, "", color="r")
    plt.plot(x_list, train_acc_list_mlp, "", color="b")

    label = ["StatNet", "MLPNet"]

    plt.legend(label, loc="upper left")
    plt.title("Train Acc Curve")
    plt.xlabel("Batch")
    plt.ylabel("Acc")
    plt.savefig("./训练集acc曲线图.jpg")
    plt.cla()

    plt.plot(valid_time_list_stat, valid_loss_list_stat, "", color="r")
    plt.plot(valid_time_list_mlp, valid_loss_list_mlp, "", color="b")

    label = ["StatNet", "MLPNet"]

    plt.legend(label, loc="upper right")
    plt.title("Valid Loss Curve")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.savefig("./验证集loss曲线图.jpg")
    plt.cla()

    plt.plot(valid_time_list_stat, valid_acc_list_stat, "", color="r")
    plt.plot(valid_time_list_mlp, valid_acc_list_mlp, "", color="b")

    label = ["StatNet", "MLPNet"]

    plt.legend(label, loc="upper left")
    plt.title("Valid Acc Curve")
    plt.xlabel("Batch")
    plt.ylabel("Acc")
    plt.savefig("./验证集acc曲线图.jpg")
    plt.cla()


if __name__ == "__main__":
    draw("mlpnet-exp2-30000.npz", "statnet-exp2-30000.npz")
