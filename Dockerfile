# 使用官方 Python 3.9 基础镜像
FROM m.daocloud.io/docker.io/library/python:3.9-slim
# FROM python:3.9-slim
# FROM ayler/py39

# 设置工作目录
WORKDIR /app

# 设置pip镜像源
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 如果没有credentials.json文件，则复制demo-credentials.json
RUN if [ ! -f credentials.json ]; then cp demo-credentials.json credentials.json; fi

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]