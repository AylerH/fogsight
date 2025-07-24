# 安装

## 虚拟环境准备

创建虚拟环境：

```
conda create -n base python=3.9
conda activate base
```

安装支持包：

```
pip install -r requirements.txt
```



# 运行
## 编辑大模型的信息
 编辑 credentials.json 文件，填入您的 API_KEY、BASE_URL 和 MODEL
## docker运行

```
docker compose up -d
```

## 浏览器网页访问

```
http://localhost:8000
```



