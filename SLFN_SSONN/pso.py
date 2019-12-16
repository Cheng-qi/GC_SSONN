
import numpy as np
    
def pso(func, lb, ub,swarmsize=14, omega=0.5, phip=1.3, phig=2.7, maxiter=100,
        iter_output=False):

    assert len(lb)==len(ub), '粒子的是上下界维度不一致'
    assert hasattr(func, '__call__'), '无效的适应度函数'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>lb), '粒子下界不能大于例子上界'
    vhigh = np.abs(ub - lb) #例子速度上界
    vlow = -vhigh           #粒子速度下界

    # 初始化适应度函数
    obj = func

    # 初始化粒子群
    S = swarmsize
    D = len(lb)  #
    x = lb + np.random.rand(S, D)*(ub-lb)  # 初始化粒子位置
    v = vlow + np.random.rand(S, D)*(vhigh - vlow)  #初始化粒子速度

    fx = obj(x)              # 当前例子适应度值
    p = x.copy()             #个体最优位置
    fp = fx.copy()           #个体最优适应度值

    i_min = np.argmin(fp)
    fg = fp[i_min]           #全局最优适应度值
    g = p[i_min,:].copy()    #全局最优位置

    ##是否选择递减的惯性权重
    if type(omega)==tuple:
        omege_lb, omege_ub = omega
        omega = np.arange(omege_ub, omege_lb,-(omege_ub-omege_lb)/maxiter)
    else:
        omega = np.ones(shape=(maxiter,))*omega

    g_record = [] #记录全局最优位置的迭代过程
    fg_record = [] #记录全局最优值

    # 迭代开始 ##################################
    for it in range(maxiter):
        g_record.append(g) #记录全局最优位置
        fg_record.append(fg)#记录全局最优值
        if it>10 and np.array(fg_record[-10:]).var() == 0: #如果连续10次迭代全局最优均没有发生更新，则提前终止
            break
        rp = np.random.uniform(size=(S, D))#个体随机因子
        rg = np.random.uniform(size=(S, D))#全局随机因子

        # 粒子速度更新
        v = omega[it]*v + phip*rp*(p - x) + phig*rg*(g - x)

        # 粒子位置更新
        x = x + v

        # 矫正超界粒子
        x = np.clip(x, lb, ub)

        # 计算粒子适应度值
        fx = obj(x)

        # 储存粒子的最优位置
        i_update = fx < fp
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        #更新粒子最优位置和最优适应度值
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            g = p[i_min, :].copy()
            fg = fp[i_min]

    if iter_output:
        return g, fg, np.array(g_record), np.array(fg_record)
    else:
        return g, fg
