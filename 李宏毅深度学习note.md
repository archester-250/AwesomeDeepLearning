# 李宏毅深度学习

## Lecture 1:Regression

任务：回归、分类、体系化学习(产生的有结构性的东西)

MAE:误差的绝对值的平均

MSE:误差的平方和

:star:局部极小值是梯度下降法的假问题

分段线性曲线：(hard sigmoid)

![tmpEEAA](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpEEAA.png)

可以用多个曲线来模拟任意一个函数

![tmp9C04](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmp9C04.png)

可以用Sigmoid曲线来逼近

![tmpE7E8](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpE7E8.png)

batch:每次**更新**在不同的database batch上

![tmp509B](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmp509B.png)

### 迭代(Epoch)vs更新(update)

换一次batch就是一次更新，遍历完一次database是一次迭代

![tmpFDE](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpFDE.png)

`Hard || Soft Sigmoid`,`ReLU`统称为激活函数。

神经网络：由多层、每层多个`sigmoid`/`ReLU`激活函数组成，在深度学习中叫隐藏层。多个隐藏层叫深度学习。



## Lecture 2:整体指导

### 解决模型偏差的问题

遇到问题时先尝试比较小与浅的网络，甚至可以用线性模型、SVM之类的简单模型来拟合，然后再来一个深的模型来调参。

### 解决过拟合问题

- 给模型制造限制
  - 给较少的参数：全连接层
  - 使用较少的特征
  - 早停、正则化、丢弃
- 但也不能限制太大（回到模型有偏差的原始问题）

![tmpD070](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpD070.png)

### Mismatch

训练集与测试集分布不同，与过拟合不同

### 交叉验证

将训练集拆成训练集和验证集

- N-折交叉验证：将训练集分成N等份，选择其中一份作为验证集，其余为训练集



## Lecture 3:局部极小值与鞍点--梯度为0

判断其是极小值/鞍点的方法：二阶导
$$
L(\theta) \approx L(\theta ') + (\theta - \theta ')^Tg + \frac{1}{2}(\theta - \theta ')^TH(\theta - \theta ')
$$
最右的一项为二阶部分，H为海塞矩阵。

H正定：局部最小值->所有特征值均为正

H负定：局部最大值

既不是正定也不是负定：鞍点

找出负特征值所对应的特征向量即可走出鞍点（运算量最大的方法）

:star:在低维空间中的局部最小值，到了更高维的空间中说不定就会变成鞍点

:star:很多时候梯度为0只是卡在了鞍点而不是局部最小值



## Lecture 4:分批(batch)和动量方法

### Batch

shuffle（洗牌）：常见做法为每个迭代之间洗牌batch，使每次迭代的batch都不一样

当每批的样本个数和总样本个数一样时，相当于没有分批。缺点为花费时间较长。

batch size为1时，相当于每一个样本结束后都会更新一次参数。

Batch size为1还是1000，所需时间相差无几。（原因：GPU并行运算）

- 大的batch size一次迭代的时间会相对较少，但是验证集和训练集的准确率都会下滑。
- 小的batch size可以减少鞍点所带来的“卡脖子”的情况，且对测试集的准确率会有提升。原因：局部最小值也有好坏之分。

![image-20220711220145680](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\image-20220711220141107.png)

平坦的是好的，尖锐的是坏的，如果测试集分布与训练集有一定差异，平坦的最小值不易对准确率造成太大的影响。

- 大的batch size倾向于进入sharp，小的倾向于进入flat。

### Momentum

![image-20220711220904269](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\image-20220711220904269.png)

在一般的梯度下降的基础上，根据前一个位移的情况再往前助推一步。
$$
m^0 = 0\\
m^{i + 1} = \lambda m^i - \eta g^i\\
\theta^{i+1} = \theta^i + m^{i+1}
$$


## Lecture 5:有适应性的学习率

Fixed μ带来的问题：可能在前一阶段需要的特别低，后一阶段又需要高的学习率。

定制化的学习率：
$$
\theta_i^{t+1}\leftarrow \theta_i^t-\frac{\eta}{\sigma_i^t}g_i^t\\
$$
σ的计算方式：

- 平方根

$$
\sigma_i^t=\sqrt{\frac{1}{t+1}\displaystyle\sum_{j=0}^t(g_i^j)^2}
$$

这个方法在Adagrad中得到运用，实现了梯度较大时学习率较小，梯度较小时学习率较大的自适应。

- RMS Prop：相比Adagrad加了一个α参数，删除了分母

$$
\sigma_i^0=\sqrt{(g_i^0)^2}\\
\sigma_i^t=\sqrt{\alpha(\sigma_i^{t-1})^2+(1-\alpha)(g_i^t)^2}
$$

α则需要超参数进行调整。

- Adam:RMS Prop + Momentum

使用自适应性学习率有一个问题：在接近局部极小值的时候，因为平坦，小的梯度不断累积，使得可能会出现突然左右震荡的情况，不过这种情况在震荡后又会较小，如此反复。原因是震荡后的梯度又比较大，又会减缓每次走的距离。

![tmpDBBF](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpDBBF.png)

### 解决方法：学习率的调度

常见策略：

- 学习率的“腐化”，让学习率随着每一次参数的更新越来越小。
- “热身”：让η先变大后变小

![tmpAB3B](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpAB3B.png)

其速率也是需要超参数来调整。

变形之一：
$$
\theta_i^{t+1}\leftarrow \theta_i^t-\frac{\eta}{\sigma_i^t}m_i^t\\
$$


## Lecture 6:Batch的规范化

### 特征的规范化

标准化：
$$
\tilde{x}_i^r\leftarrow\frac{x_i^r-\bar x_i}{\sigma_i}
$$

### Batch规范化

在进行标准化的基础上与向量γ做内积后加上β

![tmpAB83](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpAB83.png)

作用：调整一下z帽的分布

初始值：γ全1，β全0

### 测试

测试集没法用batch nomalization的这一套解决，因为根本没有batch。实际上，这需要在训练集中完成，在每次参数更新时，
$$
\bar\mu\leftarrow p\hat\mu+(1-p)\mu^t
$$
p是超参数，pytorch中默认为0.1。



## Lecture 7:分类

### 独热向量

![tmp6A98](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmp6A98.png)

### Softmax

将任意的y值规约到0-1之间
$$
y_i'=\frac{exp(y_i)}{\sum_jexp(y_i)}
$$
将大值与小值的差距变大

计算预测y‘和标签值的距离

- MSE

$$
e=\displaystyle\sum_i(\hat y_i-y_i')^2
$$



- 交叉熵（通常配合softmax使用）

$$
e=-\displaystyle\sum_i\hat y_i\ln y_i'
$$

一般在分类问题上使用交叉熵，因为MSE在损失比较大的时候梯度就会趋于平缓。



## Lecture 8:CNN（卷积神经网络）--用于图像识别

### 一种介绍卷积层的方式

一张图片是三维的张量，三个信道（channel）分别为RGB。

张量变成向量的方法：拉直

![tmpEDDA](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpEDDA.png)

问题：如果全连接的话，假设有1000个神经元，就会产生3*10^7个参数，运算时间长且容易过拟合。

![tmp5726](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmp5726.png)

#### 观察1

选择一些有辨识度的特征进行训练

![tmp9AF1](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmp9AF1.png)

不同的神经元负责辨认不同的部位

#### 观察1-->简化1

感受野（Receptive Field）：每一个神经元都只关心自己感受野之内的东西

步骤（感受野大小3*3\*3为例）：

- 将其拉直为27维的向量并将其作为神经元的输入
- 将其加上权值和偏置之后送给下一层的神经元作为输入

注意：

- 感受野可以重合甚至可以一个感受野配备多个神经元。
- 感受野可以有不同大小，可以为任意形状

##### 经典的感受野安排方式

- 看所有的channel：因此不用讲深度，直接看宽和高，合起来叫做Kernel Size，常见的设定方式为3*3，一般会有64/128个神经元守备一个感受野的范围。

- Stride：从基点感受野移动的步长

![tmp854B](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmp854B.png)

这是一个超参数，一般为1/2.如果移动到超出范围就要做一个填充(padding)，一般为补0。

#### 观察2

同样的特征可能会出现在图片的不同区域。

![tmpD98C](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpD98C.png)

问题：不同的守备范围都需要有一个侦测鸟嘴的神经元，参数太多了。

#### 观察2-->简化2

目的：让参数共享

- 参数完全相同

![tmp2037](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmp2037.png)

##### 经典的共享方式

让每一个感受野都只有一组参数：Filter

![tmpA8F7](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpA8F7.png)

#### 总结：卷积层的好处

![tmpC456](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpC456.png)

全连接层+感受野+参数共享=卷积层

用到卷积层的网络就叫做卷积神经网络，即CNN。

### 另一种介绍卷积层的方式

- 卷积层中有很多的过滤器(Filter)

![tmp67AE](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmp67AE.png)

每一个过滤器的作用是识别图片中的一个特征。

#### 侦测特征的方法：卷积（内积）

将每个感受野中的矩阵与过滤器做卷积

![tmpD38B](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpD38B.png)

对每个过滤器都做一次卷积，得到特征映射图(Feature Map)

![tmpA2A4](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpA2A4.png)

这是第一层经过卷积层后得到的特征，前一层有多少个filter后一层的高度就是多少，这里假设第一层有64个filter，第二层高度就是64.

- 网络叠加的层数越多，看的范围越大

### 两种看卷积层方式的总结

![tmp31B5](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmp31B5.png)

- 神经元视角：仅需观察图片的一小部分，每一个小部分都检测一个小范围，对于全局出现的相同filter以共享参数的方式呈现
- filter视角：每个filter都可以覆盖图片的每个角落，在进行卷积运算之后可以映射成feature map，其中一个数值重叠起来就是处理前的一个感受野。

### 观察3--池化

将原图中的一半像素点拿掉，图片缩小为原图的1/4，但人眼不会看出内容的差别，因此出现了池化压缩图像的概念。

- 池化不是一个层，没有根据数据学习任何东西。

Example:最大池化：分组，然后选择组中最大的一个

<img src="E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmp2859.png" alt="tmp2859" style="zoom:50%;" />

<img src="E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmp60EE.png" alt="tmp60EE" style="zoom:50%;" />

### 卷积层+池化

- 实际上：卷积层和池化交替使用。
- 因为池化对于最终的模型表现会有些影响，加上算力的提升，近年也在开始放弃池化的操作。

### CNN总结

![tmpE164](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmpE164.png)

Flatten：把矩阵的变量拉直为向量，然后丢进全连接层，最后加一个softmax得到图像识别的结果。










## Lecture 23:对抗攻击

### 	--避免别有用心的人恶意用词逃过神经网络的分类/排除异常因子

#### 如何攻击？

##### 无目标攻击

交叉熵(Cross Entropy)：

![img](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\v2-ad4588debf5c0d869f3589edd0425e6c_1440w.jpg)

-H(p, q)越小越好

##### 有目标攻击

$$
L(x) = -e(y, \hat{y}) + e(y, y^{target})\\
d(x^0, x) \le \epsilon
$$

第二个式子是为了减少与原图的差距，ε为一个门槛，低于该杂讯时默认人的肉眼无法识别。

###### 第二个式子怎么算？--将图像化为向量

- 2-范式：平方和

- 无穷范式：最大值

无穷范式更接近人类的感知能力，只选取变化最大的作为距离。

![tmp1494](E:\桌面\其他文件\DL\AwesomeDeepLearning\images\tmp1494.png)

2022.7.10：开错ppt了...









