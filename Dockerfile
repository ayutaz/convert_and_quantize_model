# ビルドステージ
FROM python:3.11-slim AS builder

# 必要なビルドツールとシステムパッケージをインストール
# 不要なパッケージのインストールを避けるため、必要最小限に絞ります
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    mecab \
    libmecab-dev \
    swig \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# 必要なファイルのみをコピー
COPY requirements.txt ./
COPY process_issues.py ./
COPY convert_model.py ./
COPY readme_generator.py ./

# ライブラリのインストール
RUN pip install --no-cache-dir --upgrade pip

# PyTorch のインストール（ホイールファイルをダウンロードしてインストール）
RUN curl -L -o torch-2.5.1+cpu.cxx11.abi-cp311-cp311-linux_x86_64.whl \
    "https://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.5.1%2Bcpu.cxx11.abi-cp311-cp311-linux_x86_64.whl" && \
    pip install --no-cache-dir torch-2.5.1+cpu.cxx11.abi-cp311-cp311-linux_x86_64.whl && \
    rm torch-2.5.1+cpu.cxx11.abi-cp311-cp311-linux_x86_64.whl

# 他の依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

# 不要なパッケージとファイルの削除を一度にまとめて実行
RUN apt-get purge -y build-essential libmecab-dev swig && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    rm -rf /root/.cache/pip && \
    find /usr/local/lib/python3.11/site-packages/ -type d -name '__pycache__' -exec rm -rf {} + && \
    find /usr/local/lib/python3.11/site-packages/ -name '*.pyc' -delete && \
    rm -rf /usr/local/lib/python3.11/site-packages/**/*.dist-info

# 不要なファイルの削除（テストディレクトリのみ削除）
RUN find /usr/local/lib/python3.11/site-packages/ -type d -name 'tests' -exec rm -rf {} + && \
    find /usr/local/lib/python3.11/site-packages/ -type d -name 'test' -exec rm -rf {} +

# ランタイムステージ
FROM python:3.11-slim

# 必要なランタイムパッケージをインストール（不要なパッケージを省略）
RUN apt-get update && apt-get install -y --no-install-recommends \
    mecab \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# ビルドステージから必要な Python パッケージのみをコピー
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 必要なスクリプトをコピー
COPY process_issues.py ./
COPY convert_model.py ./
COPY readme_generator.py ./

# エントリーポイントを設定（オプション）
CMD ["bash"]