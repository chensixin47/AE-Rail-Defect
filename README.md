# AE-Rail-Defect
Main_Study 是主程序，做训练和测试，其中调用了Dataset_Model模块，定义模型和数据集，调用了training_setp训练。



Generating_dataset是产生数据集的程序，它调用了Feature_Generator，然后把四种情况的数据叠在一起存起来。

Feature Generator读取useful_name_list里面的名字，把原始mat文件读进来，然后调用AE_Feature_Extractor提取特征。

AE_Feature_Extractor把时间序列转换成时频图，再用extractor调用预训练的模型network_architectures进行forward propagation。预训练的模型参数存在mx-h64-1024_0d3-1.17.pkl里面。



T-SNE读取提取出来的特征，做降维然后画图。
