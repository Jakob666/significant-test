#-*- coding:utf-8 -*-
'''
Description:
    参考网上大牛的统计学方法，相关链接 https://web.as.uky.edu/statistics/users/pbreheny/621/F12/notes/10-11.pdf
    实现 bootstrap方法，对模型预测出的显著性和非显著性蛋白在特定位置修饰的频率等进行差异分析。
==========================
@author: hbs
@date: 2017-1-6
@version: 1.0
==========================
@update: 对最终的显著性检验的方式进行优化
@version: 1.1
==========================
@update: 纠正了其中一处语法错误
@version: 1.2
==========================
@update: 加了更详尽的注释，因为实际项目中出现易混淆点
@version: 1.3
'''
import numpy as np
# from matplotlib import pyplot as plt
import os
import warnings
# from t_test import T_test


class Boostrap_test:

    def __init__(self, Confidence=0.95, time=1000):
        '''
        :param Confidence: 置信区间
        :param time: boostrap抽样的次数
        '''
        self.conf = Confidence
        self.times = time
        self.t_val_list = []

    def preprocess(self, x_data, y_data):
        '''求出x、y样本的合并均值z，将x、y样本的值进行相应的替换：
            xi'= xi - x_mean + z
            yi'= yi - y_mean + z
        :param x_data: 参数类型是列表，由样本组一的样本值组成
        :param y_data: 参数类型是列表，由样本组二的样本值组成
        '''
        conbimed = x_data + y_data
        z = np.mean(conbimed)
        #未经处理的x样本组数据均值
        x_mean = np.mean(x_data)
        x_std = np.std(x_data)
        #未经处理的y样本组数据均值
        y_mean = np.mean(y_data)
        y_std = np.std(y_data)
        #计算 observed数据的t值
        t_obs = Boostrap_test.calc_t_val(x_mean, y_mean, len(x_data), len(y_data), x_std, y_std)
        #x样本组数据预处理
        x_data = list(map(lambda i: i - x_mean + z, x_data))
        # x_mean = np.mean(x_data)                                          ##################################
        #y样本组数据预处理                                                     # 在这两处求取处理后x、y样本组        #
        y_data = list(map(lambda i: i - y_mean + z, y_data))                # 的均值是无用的，因为均值都是        #
        # y_mean = np.mean(y_data)                                          # z。之前的处理是将x、y融为一组       #
                                                                            # ，z是两组样本共同的均值            #
        return x_data, y_data, t_obs, x_mean, y_mean                        ##################################

    def boostrap(self, x_data, y_data, t_obs):
        '''
        模拟boostrap抽样，每次抽样 size个样本
        :param x_data: 参数类型是列表，由样本组一的样本值组成
        :param y_data: 参数类型是列表，由样本组二的样本值组成
        :param t_obs: 是一个数值，是preprocess方法中计算出来的两个样本组原始值的 t检验值
        :return:
        '''
        x_size = len(x_data)
        y_size = len(y_data)
        for t in range(self.times):
            x_samples = np.random.choice(x_data, x_size)  #choice方法可以模拟又放回的抽取，size的值可以大于len(data)
            y_samples = np.random.choice(y_data, y_size)

            x_samples_mean = np.mean(x_samples)
            x_samples_std = np.std(x_samples)
            y_samples_mean = np.mean(y_samples)
            y_samples_std = np.std(y_samples)

            t_val = Boostrap_test.calc_t_val(x_samples_mean, y_samples_mean, x_size, y_size, x_samples_std, y_samples_std)
            self.t_val_list.append(t_val)
        p_val = self.calc_p_val(t_obs, self.t_val_list)

        p_ref = 1.0 - self.conf
        if p_val < p_ref:
            res = "Significant difference"
        else:
            res = "Insignificant difference"

        return res, p_val, p_ref

    @staticmethod
    def calc_t_val(x_mean, y_mean, x_size, y_size, x_std, y_std):
        '''计算 bootstrap抽样样本的 t值，此处两组样本的t值使用的是不具备方差齐性时的 t'检验公式
        :param x_mean: 经过预处理后样本组一的均值
        :param y_mean: 经过预处理后样本组二的均值
        :param x_size: 样本组一的样本量
        :param y_size: 样本组二的样本量
        :param x_std: 经过预处理后样本组一的标准差
        :param y_std: 经过预处理后样本组二的标准差
        '''
        numerator = np.abs(x_mean - y_mean)
        denominator = np.sqrt(np.square(x_std)/x_size + np.square(y_std)/y_size)
        t_val = numerator/denominator
        return t_val

    def calc_p_val(self, t_obs, t_boot_list):
        '''通过bootstrap抽样所得的数据求得的t值，与observed数据测得的t值比较，求出p值
        :param t_obs: 是preprocess方法中计算出来的两个样本组原始值的 t检验值
        :param t_boot_list: 是一个列表，是经过 self.time次抽样后得到的self.time个 t'值组成的列表
        '''
        count = 0.0
        for t in range(self.times):
            if t_obs < t_boot_list[t]:
                count += 1.0
        p_val = count/self.times
        return p_val

    def main(self, x_data, y_data):
        '''将之前所有方法进行整合形成一套完整的bootstrap检验流程
        :param x_data: 参数类型是list，是样本组一的数据值组成的列表
        :param y_data: 参数类型是list，是样本组二的数据值组成的列表
        :return res是最终的检验结果 "Significant difference"或是"Insignificant difference"；
                p_val是发生第一类错误的概率； p_ref是α值，即 1-self.conf；
                x_mean是样本组x的均值； y_mean是样本组y的均值；
        '''
        x_data, y_data, t_obs, x_mean, y_mean = self.preprocess(x_data, y_data)
        res, p_val, p_ref = self.boostrap(x_data, y_data, t_obs)
        return res, p_val, p_ref, x_mean, y_mean


if __name__ == "__main__":
    warnings.filterwarnings("ignore")