# Pytorch 自定义 Dataset

> 在上节中我们使用了 Pytorch 自带的`torchvision.dataset`中的数据集，可是如果我们的数据集是自己做的或者从其他地方下载的数据集时怎么办？那么我们就需要自定义自己的`Dataset`数据集。

## 1.torch.utils.data.Dataset

`torch.utils.data.Dataset`是 Pytorch 为用户自定义数据集所设计的基类，当我们继承该类时需要完成三个函数的重载：

- `__init__()` : 初始化函数。
- `__len__()` ：该方法返回数据集的大小。
- `__getitem__()` : 该方法通过索引返回数据集中的一个文件。

那么本节将实现一个图像数据集的自定义`Dataset`。

## 2.数据集介绍

本节数据集为论文 [_Hybrid LSTM and Encoder–Decoder Architecture for Detection of Image Forgeries_](https://ieeexplore.ieee.org/abstract/document/8626149) 和 [_Two-stream encoder–decoder network for localizing image forgeries_](https://www.sciencedirect.com/science/article/pii/S1047320321002777) 所提供的图像数据集。其中包括 6 万张被篡改的图像和对应的篡改掩码标签图像。文件结构如下：

```bash
Dataset\
    |---- Tp\                             # 篡改图像数据集
        |---- dresden_spliced\
            |---- 1.png
            |---- ...
        |---- spliced_copymove_NIST\
            |---- 1.png
            |---- ...
        |---- spliced_NIST\
            |---- 1.png
            |---- ...
    |---- Gt\                              # 掩码标签数据集
        |---- dresden_spliced\
            |---- 1_gt.png
            |---- ...
        |---- spliced_copymove_NIST\
            |---- 1_gt.png
            |---- ...
        |---- spliced_NIST\
            |---- 1_gt.png
            |---- ...
```

其中每张图像的标签则是文件名后加`_gt`。例如 `./Dataset/Tp/dresden_spliced/100.png`对应的标签则是`./Dataset/Gt/dresden_spliced/100_gt.png`。

## 3.自定义数据集

```python
import os
import glob

from PIL import Image
from torch.utils.data import Dataset

# 自定义数据集需要继承 torch.utils.data.Dataset
class Imgdata(Dataset):
    def __init__(self, root_tp, root_gt, transform=None, train=None, pct=0.8):
        ''' 初始化方法

        对Dataset类进行初始化。

        Args:
            * root_tp : Tp文件夹路径
            * root_gt : Gt文件夹路径
            * transform : Pytorch transfroms预处理方法
            * train : 是否为训练集
            * pct : 训练集占比
            可以自定义需要的参数，一般包括：数据集路径、transform方法、测试集/训练集标识符、训练集占比.

        Return:
            * None
        '''
        super(Imgdata, self).__init__()

        self.transform = transform
        self.images = []
        self.labers = []

        # 获取所有图片文件夹名称
        namedir = []
        for name in sorted(os.listdir(os.path.join(root_tp))):
            if not os.path.isdir(os.path.join(root_tp, name)):
                continue
            namedir.append(os.path.join(root_tp, name))

        # 获取所有图片名称
        images = []
        for name in namedir:
            images += glob.glob(os.path.join(root_tp, name, "*.png"))

        # 排序
        # 防止测试集和训练集发生交集
        images.sort()

        # 获取所有图片标签文件名称
        for image in images:
            self.images.append(image)
            image = image[:-4]
            image = image + '_gt.png'
            self.labers.append(os.path.join(root_gt, image.split(os.sep)[-2], image.split(os.sep)[-1]))

        # 分割训练集、测试集
        if train:
            self.images = self.images[:int(pct * len(self.images))]
            self.labers = self.labers[:int(pct * len(self.labers))]
        else:
            self.images = self.images[int(pct * len(self.images)):]
            self.labers = self.labers[int(pct * len(self.labers)):]

    def __len__(self):
        '''返回数据集大小

        返回Dateset类中数据集的大小/长度。

        Args:
            * None

        Return :
            * (int) 数据集大小

        '''
        return len(self.images)

    def __getitem__(self, item):
        '''根据索引获取数据

        根据item索引返回Dataset中的数据。

        Args :
            * item : 数据索引

        Return:
            * * 索引对应的数据
            可以返回多个数据，在接收时只需要有对应的变量接收即可。
            具体细节可以在 4.自定义数据集的使用 中获取。
        '''

        # 打开图像
        image = Image.open(self.images[item]).convert("RGB")
        laber = Image.open(self.labers[item]).convert("1")

        # 通过 transform 对图像预处理
        image = self.transform(image)
        laber = self.transform(laber)

        # 返回数据和标签
        return image, laber
```

## 4.自定义数据集的使用

```python
root_tp = ".\\Dataset\\Tp"
root_gt = ".\\Dataset\\Gt"

# 初始化自定义Dataset
data_train = Imgdata(root_tp, root_gt, train=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Resize([256, 256])
                              ]))
data_test = Imgdata(root_tp, root_gt, train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Resize([256, 256])
                             ]))

# 创建迭代对象
# 此步操作和Pytorch自带的数据集操作相同
data_train = DataLoader(data_train, batch_size=8, shuffle=True)
data_test  = DataLoader(data_test,  batch_size=8, shuffle=True)
```

此时我们就完成了自定义数据集的加载。当我们使用迭代对象时，迭代对象将返回 N+1 个数据，其中 N 个数据是你在`Dataset`中`__getitem__()`函数中返回的数据的种类，在本节中`N=2`，还有一个参数则是`batch_idex`，是`batch`的索引号代表这是第几个`batch`。

```python
for batch_idx, (img,laber) in enumerate(data_train):
        ...
```

请注意，迭代器返回数据的的顺序如下:

- batch_index 先返回 batch 的索引号。
- 之后按`__getitem__()`中`return`的顺序，在本节中为：
  - image
  - laber

如果你有特殊的需求，比如你不仅想返回`image`和`laber`，你还想返回`image`中的 R、G、B 三个通道，那么就可以修改如下:

```python
# Dataset中的修改
def __getitem__(self, item):

        # 打开图像
        image = Image.open(self.images[item]).convert("RGB")
        laber = Image.open(self.labers[item]).convert("1")

        # 通过 transform 对图像预处理
        image = self.transform(image)
        laber = self.transform(laber)

        # 获取R、G、B通道
        R, G, B = image.split()
        R = self.transform(R)
        G = self.transform(G)
        B = self.transform(B)

        # 返回数据
        return image, laber, R, G, B


# 使用时的修改
for batch_idx, (img,laber,R,G,B) in enumerate(data_train):
        ...
```

## 5.参考资料

- [torch.utils.data.Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Mazumdar A, Bora P K. Two-stream encoder–decoder network for localizing image forgeries[J]. Journal of Visual Communication and Image Representation, 2022, 82: 103417.](https://www.sciencedirect.com/science/article/pii/S1047320321002777)
- [J. H. Bappy, C. Simons, L. Nataraj, B. S. Manjunath and A. K. Roy-Chowdhury, "Hybrid LSTM and Encoder–Decoder Architecture for Detection of Image Forgeries," in IEEE Transactions on Image Processing, vol. 28, no. 7, pp. 3286-3300, July 2019, doi: 10.1109/TIP.2019.2895466.](https://ieeexplore.ieee.org/abstract/document/8626149)
