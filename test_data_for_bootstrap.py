# -*- coding:utf-8 -*-
'''
Description:
    改程序用于制造出正态分布的四种情况数据，四种情况中两组数据的特点如下：
    第一组数据：两个正态分布，N1~(1,2)；N2~(1,2)。即两个总体分不相同。
    第二组数据：两个正态分布，N1~(1,2)；N2~(10,2)。即两个总体均值μ有差异。
    第三组数据：两个正态分布，N1~(1,2); N2~(1,5)。即两个总体标准差有差异。
    第四组数据：两个正态分布，N1~(1,2); N2~(10,5)。即均值和标准差都有差异。

    以上制造出的数据用于检验bootstrap检验的程序是否正确。
======================================
@author: hbs
@date: 2018-1-16
@version: 1.0
'''
import numpy as np
from bootstrapTest import Boostrap_test


class Normal_test:
    @staticmethod
    def form_same_normal_distribution(miu, sigma, data_size1=1000, data_size2=1000):
        '''
        用于创建两个相同的正态分布
        :param miu: 分布总体的均值
        :param sigma: 分布总体的标准差
        :param data_size1: 总体一的大小
        :param data_size2: 总体二的大小
        :return:
        '''
        norm1 = np.random.normal(miu, sigma, data_size1)
        norm2 = np.random.normal(miu, sigma, data_size2)
        return norm1, norm2

    @staticmethod
    def different_miu_normal_distribution(miu1, miu2, sigma, data_size1=1000, data_size2=1000):
        if miu1 == miu2:
            print("can't input same miu value.")
            exit()
        norm1 = np.random.normal(miu1, sigma, data_size1)
        norm2 = np.random.normal(miu2, sigma, data_size2)
        return norm1, norm2

    @staticmethod
    def different_sigma_normal_distribution(miu, sigma1, sigma2, data_size1=1000, data_size2=1000):
        if sigma1 == sigma2:
            print("can't input same sigma value.")
            exit()
        norm1 = np.random.normal(miu, sigma1, data_size1)
        norm2 = np.random.normal(miu, sigma2, data_size2)
        return norm1, norm2

    @staticmethod
    def totally_different_normal_distribution(miu1, miu2, sigma1, sigma2, data_size1=1000, data_size2=1000):
        if miu1 == miu2 or sigma1 == sigma2:
            print("can't input same miu value or miu value.")
            exit()
        norm1 = np.random.normal(miu1, sigma1, data_size1)
        norm2 = np.random.normal(miu2, sigma2, data_size2)
        return norm1, norm2

    def test_bootstrap(self):
        b = Boostrap_test(time=1000,side="one-side")
        # 相同分布的两个总体进行检验
        norm1, norm2 = Normal_test.totally_different_normal_distribution(9.5, 10, 3, 1)
        x_data = list(norm1)
        y_data = list(norm2)
        res, p_val, p_ref, x_mean, y_mean, x_std, y_std = b.main(x_data, y_data)
        print(res, p_val, p_ref, x_mean, y_mean, x_std, y_std)



if __name__ == "__main__":
    n = Normal_test()
    n.test_bootstrap()