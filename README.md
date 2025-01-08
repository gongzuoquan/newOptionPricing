# newOptionPricing
Not surprisingly, it's my last work on GPU.

本项目的目的是实现期权定价程序的加速。

程序均采用Monte Carlo方法进行定价，实验均在Nvidia P100上进行验证。

包含了三个主要部分：

1. MCAsianOption_cuda_speedup：Black-Scholes-Merton模型下的亚式期权加速

2. MCAsianOption_cuda_MertonDJ：Merton 跳跃扩散模型下的亚式期权加速

3. MCAsianOption_cuda_American：Black-Scholes-Merton模型下的美式亚式期权加速（涉及最小二乘）

其余部分为：
1. test_cg：CUDA cooperative groups的效果测试
2. test_leastSquare：CUDA上实现最小二乘法的测试
3. test_RNG：不同随机数生成器性能测试
4. test_trngVScurand：TRNG和CURAND两个库生成随机数效率对比
5. test_var：使用CUDA实现方差计算

如有疑问请联系作者，e-mail：
zuoquangong@163.com

Thanks。