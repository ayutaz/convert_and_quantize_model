# ビルドステージ
FROM python:3.11-slim AS builder

# 必要なビルドツールをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
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

# ソースコードをコピー
COPY . /app

# ライブラリのインストール
RUN pip install --no-cache-dir --upgrade pip

# PyTorchのインストール（CPU版の軽量ホイールを指定）
RUN pip install --no-cache-dir https://download.pytorch.org/whl/cpu/torch-2.1.0%2Bcpu-cp311-cp311-linux_x86_64.whl

# 他の依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

# ランタイムステージ
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# ビルドステージから必要なファイルをコピー
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# ソースコードをコピー
COPY . /app

# 依存関係に必要なランタイムライブラリをインストール（必要に応じて）
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ランタイムに必要なパッケージをここに記載
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# キャッシュの削除（念のため）
RUN rm -rf ~/.cache/pip

# エントリーポイントを設定（オプション）
CMD ["bash"]