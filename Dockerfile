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

# GitHub Actions 用に、CUDA がない環境では CPU 版の PyTorch をインストール
# 環境変数またはビルド引数を使用して切り替え可能にする
ARG INSTALL_CUDA=false

RUN if [ "$INSTALL_CUDA" = "true" ] ; then \
        echo "Installing CUDA version of PyTorch" && \
        pip install torch==2.1.0+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 ; \
    else \
        echo "Installing CPU version of PyTorch" && \
        pip install torch==2.1.0 ; \
    fi

# 他の依存関係をインストール
RUN pip install -r requirements.txt

# 作業ディレクトリを設定
WORKDIR /app

# エントリーポイントを設定（必要に応じて）
CMD ["bash"]