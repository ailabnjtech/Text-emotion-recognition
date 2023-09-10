中文文本情绪识别

##
机器：一块3090Ti ， 训练时间：5分钟。  

## 环境
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX  
 

## 更换自己的数据集
 - 按照[THUCNews]数据集的格式来格式化你的中文数据集。  

## 效果
模型|acc|备注
--|--|--
bert|88.57%|单纯的bert
bert_CNN|86.55%|bert + CNN  
bert_RCNN|86.09%|bert + RCNN  
bert_DPCNN|87.55%|bert + DPCNN
bert_BiLSTM|86.79%|bert + BiLSTM
ERNIE|91.21%|中文碾压bert
ERNIE_CNN|91.75%|ERNIE + CNN    
ERNIE_RCNN|92.35%|ERNIE + RCNN
ERNIE_DPCNN|91.41%|ERNIE + DPCNN
ERNIE_BiLSTM|93.29%|ERNIE + BiLSTM


## 预训练语言模型
bert模型放在 bert_pretain目录下，ERNIE模型放在ERNIE_pretrain目录下，每个目录下都是三个文件：
 - pytorch_model.bin  
 - bert_config.json  
 - vocab.txt  

预训练模型下载地址：  
bert_Chinese: 模型 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz  
              词表 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt  

ERNIE_Chinese: http://image.nghuyong.top/ERNIE.zip  
解压后，按照上面说的放在对应目录下，文件名称确认无误即可。  

## 使用说明
下载好预训练模型就可以跑了。

# 训练并测试：
# bert
python run.py --model bert

# bert + 其它
python run.py --model bert_CNN

# ERNIE
python run.py --model ERNIE
```

## 参数
模型都在models目录下，超参定义和模型定义在同一文件中。  


## 对应论文
[1] B. Gao and F. Zhang, "Manually Crafted Chinese Text Corpus for Text Emotion Recognition," 2023 International Joint Conference on Neural Networks (IJCNN), Gold Coast, Australia, 2023, pp. 1-7, doi: 10.1109/IJCNN54540.2023.10191747.  
