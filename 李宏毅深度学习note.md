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



## Lecture 4:分批(batch)和动力？



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









