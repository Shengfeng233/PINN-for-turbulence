"""
PINN融合不可压缩RANS_Niut方程
二维流动
PINN模型 + LOSS函数
"""
import numpy as np
import torch
import torch.nn as nn
from pyDOE import lhs
from read_data import *

# 定义PINN网络模块，包括数据读取函数，参数初始化
# 正问题和反问题的偏差和求导函数
# 全局参数
filename_load_model = './NS_model_train.pt'
filename_save_model = './NS_model_train.pt'
filename_data = './2d_cylinder_Re3900_100x100.mat'
filename_loss = './loss.csv'
# 训练设备为GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == 'cpu':
    print("wrong device")


# 定义网络结构,由layer列表指定网络层数和神经元数
class PINN_Net(nn.Module):
    def __init__(self, layer_mat):
        super(PINN_Net, self).__init__()
        self.layer_num = len(layer_mat) - 1
        self.base = nn.Sequential()
        for i in range(0, self.layer_num - 1):
            self.base.add_module(str(i) + "linear", nn.Linear(layer_mat[i], layer_mat[i + 1]))
            self.base.add_module(str(i) + "Act", nn.Tanh())
        self.base.add_module(str(self.layer_num - 1) + "linear",
                             nn.Linear(layer_mat[self.layer_num - 1], layer_mat[self.layer_num]))
        self.lam1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.lam2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.Initial_param()

    def forward(self, x, y, t):
        X = torch.cat([x, y, t], 1).requires_grad_(True)
        predict = self.base(X)
        return predict

    # 对参数进行初始化
    def Initial_param(self):
        for name, param in self.base.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)

    # 类内方法：求数据点的loss
    def data_mse(self, x, y, t, u, v, p):
        predict_out = self.forward(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v) + mse(p_predict, p)
        return mse_predict

    # 类内方法：求数据点的loss(不含压力数据)
    def data_mse_without_p(self, x, y, t, u, v):
        predict_out = self.forward(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v)
        return mse_predict

    # 类内方法：求方程点的loss-无量纲方程
    def equation_mse_dimensionless(self, x, y, t, Re):
        # 正问题
        predict_out = self.forward(x, y, t)
        # 获得预测的输出u,v,w,p,k,epsilon
        u = predict_out[:, 0].reshape(-1, 1)
        v = predict_out[:, 1].reshape(-1, 1)
        p = predict_out[:, 2].reshape(-1, 1)
        niu_t = predict_out[:, 3].reshape(-1, 1)
        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        # 一阶导
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        # 二阶导
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        # 计算偏微分方程的残差
        f_equation_mass = u_x + u_y
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0/Re * (1+niu_t) * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0/Re * (1+niu_t) * (v_xx + v_yy)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.from_numpy(np.zeros((x.shape[0], 1))).float().requires_grad_(True).to(device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros) + \
                       mse(f_equation_mass, batch_t_zeros)

        return mse_equation

    # 类内方法：求数据点的loss-流函数法
    def data_mse_psi(self, x, y, t, u, v, p):
        predict_out = self.forward(x, y, t)
        psi = predict_out[:, 0].reshape(-1, 1)
        u_predict = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v_predict = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        p_predict = predict_out[:, 1].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v) + mse(p_predict, p)
        return mse_predict

    # 类内方法：求方程点的loss-无量纲方程-流函数法
    def equation_mse_dimensionless_psi(self, x, y, t, Re):
        # 正问题
        predict_out = self.forward(x, y, t)
        # 获得预测的输出u,v,w,p,k,epsilon
        psi = predict_out[:, 0].reshape(-1, 1)
        p = predict_out[:, 1].reshape(-1, 1)
        u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        niu_t = predict_out[:, 2].reshape(-1, 1)
        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        # 一阶导
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        # 二阶导
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        # 计算偏微分方程的残差
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0/Re * (1+niu_t) * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0/Re * (1+niu_t) * (v_xx + v_yy)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.from_numpy(np.zeros((x.shape[0], 1))).float().requires_grad_(True).to(device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros)

        return mse_equation


# 生成矩形域方程点
def generate_eqp_rect(low_bound, up_bound, dimension, points):
    eqa_xyzt = low_bound + (up_bound - low_bound) * lhs(dimension, points)
    per = np.random.permutation(eqa_xyzt.shape[0])
    new_xyzt = eqa_xyzt[per, :]
    Eqa_points = torch.from_numpy(new_xyzt).float()
    return Eqa_points


# 定义偏微分方程（的偏差）inverse为反问题
def f_equation_inverse(x, y, t, pinn_example):
    lam1 = pinn_example.lam1
    lam2 = pinn_example.lam2
    predict_out = pinn_example.forward(x, y, t)
    # 获得预测的输出psi,p
    psi = predict_out[:, 0].reshape(-1, 1)
    p = predict_out[:, 1].reshape(-1, 1)
    # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
    u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
    v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    # 计算偏微分方程的残差
    f_equation_x = u_t + (u * u_x + v * u_y) + lam1 * p_x - lam2 * (u_xx + u_yy)
    f_equation_y = v_t + (u * v_x + v * v_y) + lam1 * p_y - lam2 * (v_xx + v_yy)
    return u, v, p, f_equation_x, f_equation_y


def f_equation_identification(x, y, t, pinn_example, lam1=1.0, lam2=0.01):
    # 正问题,需要用户自行提供系统的参数值，默认为1&0.01
    predict_out = pinn_example.forward(x, y, t)
    # 获得预测的输出psi,p
    psi = predict_out[:, 0].reshape(-1, 1)
    p = predict_out[:, 1].reshape(-1, 1)
    # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
    u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
    v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    # 计算偏微分方程的残差
    f_equation_x = u_t + (u * u_x + v * u_y) + lam1 * p_x - lam2 * (u_xx + u_yy)
    f_equation_y = v_t + (u * v_x + v * v_y) + lam1 * p_y - lam2 * (v_xx + v_yy)
    return u, v, p, f_equation_x, f_equation_y


def shuffle_data(x, y, t, u, v, p):
    X_total = torch.cat([x, y, t, u, v, p], 1)
    X_total_arr = X_total.data.numpy()
    np.random.shuffle(X_total_arr)
    X_total_random = torch.tensor(X_total_arr)
    return X_total_random


def simple_norm(x, y, t, u, v, p, feature_mat):
    x = x / feature_mat[0, 0]
    y = y / feature_mat[0, 1]
    t = t / feature_mat[0, 2]
    u = u / feature_mat[0, 3]
    v = v / feature_mat[0, 4]
    p = p / feature_mat[0, 5]
    return x, y, t, u, v, p, feature_mat
