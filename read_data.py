"""
不同的数据读取方法
"""
import scipy.io
import numpy as np
import torch


# 读取原始数据,并转化为x,y,t--u,v,p(N*T,1),返回值为Tensor类型


def read_2D_data(filename):
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']  # N*dimension*T
    X_star = data_mat['X_star']  # N*dimension
    T_star = data_mat['T_star']  # T*1
    P_star = data_mat['P_star']  # N*T
    # 读取坐标点数N和时间步数T
    N = X_star.shape[0]
    T = T_star.shape[0]
    # 将数据化为x,y,z,t--u,v,w,p形式(N*T,1)
    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(T_star, (1, N)).T
    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    PP = P_star
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.max(temp, 0)
    feature_mat[1, :] = np.min(temp, 0)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, t, u, v, p, feature_mat


def read_2D_data_surround(filename):
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']  # N*dimension*T
    X_star = data_mat['X_star']  # N*dimension
    T_star = data_mat['T_star']  # T*1
    P_star = data_mat['P_star']  # N*T
    # 读取坐标点数N和时间步数T
    N = X_star.shape[0]
    T = T_star.shape[0]
    # 将数据化为x,y,z,t--u,v,w,p形式(N*T,1)
    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(T_star, (1, N)).T
    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    PP = P_star
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.max(temp, 0)
    feature_mat[1, :] = np.min(temp, 0)
    right_index = (np.where(x == feature_mat[0, 0]))[0].reshape(-1, 1)
    left_index = (np.where(x == feature_mat[1, 0]))[0].reshape(-1, 1)
    front_index = (np.where(y == feature_mat[0, 1]))[0].reshape(-1, 1)
    back_index = (np.where(y == feature_mat[1, 1]))[0].reshape(-1, 1)
    boundary_index_temp = np.concatenate((right_index,left_index,front_index,back_index), 0)
    boundary_index = np.unique(boundary_index_temp).reshape(-1, 1)
    x_bound = x[boundary_index].reshape(-1, 1)
    y_bound = y[boundary_index].reshape(-1, 1)
    t_bound = t[boundary_index].reshape(-1, 1)
    u_bound = u[boundary_index].reshape(-1, 1)
    v_bound = v[boundary_index].reshape(-1, 1)
    p_bound = p[boundary_index].reshape(-1, 1)
    x = torch.tensor(x_bound, dtype=torch.float32)
    y = torch.tensor(y_bound, dtype=torch.float32)
    t = torch.tensor(t_bound, dtype=torch.float32)
    u = torch.tensor(u_bound, dtype=torch.float32)
    v = torch.tensor(v_bound, dtype=torch.float32)
    p = torch.tensor(p_bound, dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, t, u, v, p, feature_mat


def read_data_portion(filename, portion):
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']  # N*dimension*T
    X_star = data_mat['X_star']  # N*dimension
    T_star = data_mat['t']  # T*1
    P_star = data_mat['p_star']  # N*T

    # 读取坐标点数N和时间步数T
    N = X_star.shape[0]
    T = T_star.shape[0]

    # 将数据化为x,y,t---u,v,p(N*T,1)
    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(T_star, (1, N)).T
    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    PP = P_star
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.max(temp, 0)
    feature_mat[1, :] = np.min(temp, 0)
    x_unique = np.unique(x).reshape(-1, 1)
    y_unique = np.unique(y).reshape(-1, 1)
    index_arr_x = np.linspace(0, len(x_unique) - 1, int(len(x_unique) * portion)).astype(int).reshape(-1, 1)
    index_arr_y = np.linspace(0, len(y_unique) - 1, int(len(y_unique) * portion)).astype(int).reshape(-1, 1)
    x_select = x_unique[index_arr_x].reshape(-1, 1)
    y_select = y_unique[index_arr_y].reshape(-1, 1)
    del x_unique, y_unique, index_arr_x, index_arr_y
    index_x = np.empty((0, 1), dtype=int)
    index_y = np.empty((0, 1), dtype=int)
    for select_1 in x_select:
        index_x = np.append(index_x, np.where(x == select_1)[0].reshape(-1, 1), 0)
    for select_2 in y_select:
        index_y = np.append(index_y, np.where(y == select_2)[0].reshape(-1, 1), 0)
    index_all = np.intersect1d(index_x, index_y, assume_unique=False, return_indices=False).reshape(-1, 1)
    x = x[index_all].reshape(-1, 1)
    y = y[index_all].reshape(-1, 1)
    t = t[index_all].reshape(-1, 1)
    u = u[index_all].reshape(-1, 1)
    v = v[index_all].reshape(-1, 1)
    p = p[index_all].reshape(-1, 1)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, t, u, v, p, feature_mat

