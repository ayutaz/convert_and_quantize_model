# ビルドステージ
FROM python:3.11-slim AS builder

# 必要なビルドツールとシステムパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
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

# 不要なパッケージの削除
RUN apt-get remove --purge -y build-essential libmecab-dev swig && \
    apt-get autoremove -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# キャッシュと不要なファイルの削除
RUN rm -rf /root/.cache/pip \
    && find /usr/local/lib/python3.11/site-packages/ -name '*.pyc' -delete \
    && find /usr/local/lib/python3.11/site-packages/ -name '__pycache__' -type d -exec rm -rf '{}' + \
    && rm -rf /usr/local/lib/python3.11/site-packages/**/*.dist-info

# ランタイムステージ
FROM python:3.11-slim

# 必要なランタイムパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    mecab \
    mecab-ipadic-utf8 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# ビルドステージから必要な Python パッケージのみをコピー
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 必要なファイルのみをコピー
COPY process_issues.py ./
COPY convert_model.py ./
COPY readme_generator.py ./

# エントリーポイントを設定（オプション）
CMD ["bash"]