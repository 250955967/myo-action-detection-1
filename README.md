# myo-action-detection
运行程序前需要运行myo connect程序，并将手环通过蓝牙连接到电脑，完成数据同步
## 环境搭建
```bash
# 安装虚拟环境-python3
virtualenv venv
source venv/bin/activate

# 安装包
pip install -r requirements.txt

# 复制sdk文件夹到项目根目录,

# make dir
mkdir models
mkdir output
```



## data collection

## 离线收集数据
```bash
python3 collect_data_csv.py
# 按照提升摆pose，该程序会将收集的手环数据放在output文件夹下，并且是以“gesture_data”为前缀的csv文件
```


## data process
## 模型加载
```bash
# 复制模型到models文件夹
cp le.pkl models/
cp random_forest.pkl models/

# 运行程序
python3 models.py load
```
## 模型训练train
```bash
python3 models.py train
# 该程序从mysql数据库读取收集的手环收据，单机环境运行需要搭建mysql数据库，并将手环数据导入数据库使用
```

## 手势

伸手指
单指：
大拇指：Thumbs_up
食指：
中指
无名指
小拇指

双指
大拇指+食指
大拇指+小指
食指+中指
食指+小指
无名指+小指

三指：
大拇指+食指+中指
大拇指+食指+小指
食指+中指+无名指
食指+无名指+小指
中指+无名指+小指

四指：
食指+中指+无名指+小指

五指伸开
食指按压桌面
中指按压桌面
五指呈现抓形状
握拳

