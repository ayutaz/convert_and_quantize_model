# ベースイメージの指定（CPU版のPython 3.11）
FROM python:3.11-slim

# 必須のパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# GitHub CLI のインストール
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
    dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
    https://cli.github.com/packages stable main" > \
    /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && apt-get install -y gh && \
    rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# リポジトリの内容をコピー
COPY . /app

# ライブラリのインストール
RUN python -m pip install --upgrade pip

# PyTorchのインストール（CPU版）
RUN pip install torch==2.1.0

# 他の依存関係をインストール
RUN pip install -r requirements.txt

# エントリーポイントを設定（オプション）
CMD ["bash"]