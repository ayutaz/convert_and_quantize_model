# ベースイメージの指定（CPU 版の Python 3.11）
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必須のパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# リポジトリの内容をコピー
COPY . /app

# requirements.txt をコピー
COPY requirements.txt /app/requirements.txt

# ライブラリのインストール
RUN python -m pip install --upgrade pip

# 他の依存関係をインストール
RUN pip install -r requirements.txt

# 作業ディレクトリを設定
WORKDIR /app

# エントリーポイントを設定（必要に応じて）
CMD ["bash"]