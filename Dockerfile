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
    apt-get update && apt-get install -y --no-install-recommends gh && \
    rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# 必要なファイルのみをコピー
COPY requirements.txt ./
COPY process_issues.py ./
# 他に必要なファイルがあれば、ここで個別にコピーします
# 例: COPY src/ ./src/

# ライブラリのインストール
RUN pip install --no-cache-dir --upgrade pip

# PyTorch のインストール（CPU 版の軽量ホイールを指定）
RUN pip install --no-cache-dir https://download.pytorch.org/whl/cpu/torch-2.1.0%2Bcpu-cp311-cp311-linux_x86_64.whl

# 他の依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

# ランタイムステージ
FROM python:3.11-slim

# 必要なランタイムパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
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
    apt-get update && apt-get install -y --no-install-recommends gh && \
    rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# ビルドステージから必要なファイルをコピー
COPY --from=builder /usr/local /usr/local

# 必要なファイルのみをコピー
COPY process_issues.py ./
# 他に必要なファイルがあれば、ここで個別にコピーします
# 例: COPY src/ ./src/

# キャッシュの削除（念のため）
RUN rm -rf /root/.cache/pip \
    && find /usr/local/lib/python3.11/site-packages/ -name '__pycache__' -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages/ -type d -name 'tests' -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages/ -name '*.pyc' -delete \
    && rm -rf /usr/local/lib/python3.11/site-packages/**/*.dist-info

# エントリーポイントを設定（オプション）
CMD ["bash"]