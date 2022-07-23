# Pytorch

- `Dataset`:提供一种方式获取数据及其label

```python
from torch.utils.data import Dataset
```



- `Dataloader`:为后面的网络提供不同的数据形式
- `tensor`:张量
  - add_(num):所有元素都加上num
  - sub_(num)同理
  - 对应元素：* 矩阵乘法：matmul
- 三个参数的`conv_2d`：`in_channels`,`out_channels`,`kernel_size`(n*n)
- `transforms.Compose()`:将多个操作整合成一个操作，里面是一个list，填入一系列操作
- `transforms.ToTensor()`:将数值转换为0-1间
- `transforms.Normalize()`:将数标准化，第一部分是平均值，第二部分是标准差