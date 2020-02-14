# classification-with-resnext

“恒锐杯”图像对象分类 

                                     
                                                        
step：
(1) 数据读取和扩充

(2) 用resnext进行freature提取

(3) 基于提取的特征尝试各种MLP模型

(4) 异步权值训练，最后拼接模型，对整体SGD+学习率衰减训练权值



note:
在训练和预测时可根据电脑配置选择双显卡载入，
torch中有专门的DataParallel函数，在resnext.py对应处有注释（可选），可以很大程度上提高运行速度。
