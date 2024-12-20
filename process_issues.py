import json
import subprocess
import sys
import os
import urllib.parse
import re

def main(issues_json_path, repository, run_id):
    # キャッシュのクリーンアップ
    cache_dir = os.getenv('HF_HOME')
    if not cache_dir:
        cache_dir = os.path.expanduser('~/.cache/huggingface')
    if os.path.exists(cache_dir):
        print(f"Deleting cache directory before processing issues: {cache_dir}")
        shutil.rmtree(cache_dir)

    # 環境変数から GITHUB_TOKEN を取得
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("GITHUB_TOKEN が設定されていません。")
        sys.exit(1)

    # gh CLI の認証ステータスを確認（任意）
    try:
        subprocess.run(["gh", "auth", "status"], check=True)
    except subprocess.CalledProcessError as e:
        print("GitHub CLI の認証に失敗しました。")
        print(e)
        sys.exit(1)

    # Issues の読み込み
    with open(issues_json_path, 'r', encoding='utf-8') as f:
        issues_json = f.read()
    issues = json.loads(issues_json)
    if issues:
        issue = issues[0]
        title = issue['title']
        issue_number = issue['number']
        issue_labels = [label['name'] for label in issue.get('labels', [])]

        print(f"Processing Issue #{issue_number}: {title}")

        # タイトルがモデル変換用のフォーマットに合致するかチェック
        # 英数字、アンダースコア、ハイフン、およびスラッシュのみを含むタイトルをモデル変換用とみなす
        if re.match(r'^[a-zA-Z0-9_\-]+\/[a-zA-Z0-9_\-]+$', title.strip()):
            print(f"Issue #{issue_number} is a model conversion request.")
        else:
            print(f"Issue #{issue_number} is not a model conversion request. Skipping.")
            return

        # "Failed" ラベルが付いている場合はスキップ
        if "Failed" in issue_labels:
            print(f"Issue #{issue_number} has 'Failed' label. Skipping.")
            return

        # 重複している Issue がないかチェック
        duplicate_issue_number = check_for_duplicate_issue(title, issue_number, repository)
        if duplicate_issue_number:
            print(f"Issue #{issue_number} is a duplicate of Issue #{duplicate_issue_number}.")
            # "Duplicate" ラベルを追加し、コメントを投稿してクローズ
            add_duplicate_label_and_comment(issue_number, duplicate_issue_number)
            return

        # ここからモデル変換処理を開始
        # タイトルからモデル名を抽出
        model_name = title.strip()
        if '/' in title:
            repo_model_name = title.split('/', 1)[1].strip()
        else:
            repo_model_name = model_name

        # モデル名を安全なリポジトリ名に変換
        safe_model_name = repo_model_name.replace('/', '_')  # スラッシュをアンダースコアに置換

        # Issue に「In Progress」ラベルを追加
        add_label_cmd = f'gh issue edit {issue_number} --add-label "In Progress"'
        subprocess.run(add_label_cmd, shell=True)

        # モデルの変換を実行
        cmd = f"python convert_model.py --model '{model_name}' --upload"
        print(f"Running command: {cmd}")
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if process.returncode != 0:
            print(f"Model conversion failed for {model_name}")
            # エラーログを取得
            error_log = process.stderr

            # ジョブのURLを生成
            job_url = f"https://github.com/{repository}/actions/runs/{run_id}"

            # Issue にコメントと「Failed」ラベルを追加
            comment_body = f"モデル **{model_name}** の変換中にエラーが発生しました。\n\nエラーログ:\n```\n{error_log}\n```\n\n[ジョブの詳細はこちら]({job_url})"

            # コメントをファイルに書き出す
            with open('comment_body.txt', 'w', encoding='utf-8') as f:
                f.write(comment_body)

            comment_cmd = f'gh issue comment {issue_number} --body-file "comment_body.txt"'
            subprocess.run(comment_cmd, shell=True)
            label_cmd = f'gh issue edit {issue_number} --remove-label "In Progress" --add-label "Failed"'
            subprocess.run(label_cmd, shell=True)
            return

        # モデルのアップロードが成功したら、Issue にコメントしてラベルを更新してクローズ
        # アップロード先のURLを生成
        repo_name = f"{safe_model_name}-ONNX-ORT"
        encoded_repo_name = urllib.parse.quote(repo_name)
        uploaded_model_url = f"https://huggingface.co/ort-community/{encoded_repo_name}"

        comment_body = f"モデル **{model_name}** の処理が完了しました。\n\nアップロード先: [{uploaded_model_url}]({uploaded_model_url})"

        # コメントをファイルに書き出す
        with open('comment_body.txt', 'w', encoding='utf-8') as f:
            f.write(comment_body)

        # コメントを投稿
        comment_cmd = f'gh issue comment {issue_number} --body-file "comment_body.txt"'
        subprocess.run(comment_cmd, shell=True)

        # "In Progress" と "Failed" ラベルを削除し、"Completed" ラベルを追加
        label_cmd = f'gh issue edit {issue_number} --remove-label "In Progress,Failed" --add-label "Completed"'
        subprocess.run(label_cmd, shell=True)

        # Issue をクローズ
        close_cmd = f'gh issue close {issue_number}'
        subprocess.run(close_cmd, shell=True)
    else:
        print("処理すべきIssueはありません。")

def check_for_duplicate_issue(title, current_issue_number, repository):
    # 同じタイトルを持つクローズされた Issue を検索
    search_cmd = f'gh issue list --search "{title} in:title is:closed repo:{repository}" --json number,title,labels'
    result = subprocess.run(search_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to search for duplicate issues. Error: {result.stderr}")
        return None

    try:
        issues = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Failed to parse JSON output from gh issue list command.")
        return None

    for issue in issues:
        issue_number = issue['number']
        issue_labels = [label['name'] for label in issue.get('labels', [])]
        # "Completed" ラベルが付いている Issue を対象とする
        if "Completed" in issue_labels and issue_number != current_issue_number:
            return issue_number  # 重複する過去の Issue の番号を返す

    return None  # 重複する Issue が見つからなかった

def add_duplicate_label_and_comment(issue_number, duplicate_issue_number):
    # "Duplicate" ラベルを追加
    label_cmd = f'gh issue edit {issue_number} --add-label "Duplicate"'
    subprocess.run(label_cmd, shell=True)
    # コメントを投稿
    comment_body = f"この Issue は過去の Issue #{duplicate_issue_number} と重複しています。そちらをご参照ください。"
    with open('comment_body.txt', 'w', encoding='utf-8') as f:
        f.write(comment_body)
    comment_cmd = f'gh issue comment {issue_number} --body-file "comment_body.txt"'
    subprocess.run(comment_cmd, shell=True)
    # Issue をクローズ
    close_cmd = f'gh issue close {issue_number}'
    subprocess.run(close_cmd, shell=True)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python process_issues.py '<issues_json_path>' '<repository>' '<run_id>'")
        sys.exit(1)
    issues_json_path = sys.argv[1]
    repository = sys.argv[2]
    run_id = sys.argv[3]
    main(issues_json_path, repository, run_id)