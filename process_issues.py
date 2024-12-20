import sys
import os
import urllib.parse
import re
import subprocess
from github import Github

def main(repository, run_id):
    # 環境変数から GITHUB_TOKEN を取得
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("GITHUB_TOKEN が設定されていません。")
        sys.exit(1)

    # GitHub API クライアントを初期化
    g = Github(github_token)

    # リポジトリの取得
    repo = g.get_repo(repository)

    # 処理すべき Issue の取得
    issues = repo.get_issues(state='open', labels=[])

    found_issue_to_process = False  # 処理した Issue があるかどうかのフラグ

    for issue in issues:
        title = issue.title
        issue_number = issue.number
        issue_labels = [label.name for label in issue.labels]

        print(f"Processing Issue #{issue_number}: {title}")

        # タイトルがモデル変換用のフォーマットに合致するかチェック
        if re.match(r'^[a-zA-Z0-9_\-]+\/[a-zA-Z0-9_\-]+$', title.strip()):
            print(f"Issue #{issue_number} is a model conversion request.")
        else:
            print(f"Issue #{issue_number} is not a model conversion request. Skipping.")
            continue  # 次の Issue をチェック

        # "Failed" ラベルが付いている場合はスキップ
        if "Failed" in issue_labels:
            print(f"Issue #{issue_number} has 'Failed' label. Skipping.")
            continue  # 次の Issue をチェック

        # 重複している Issue がないかチェック
        duplicate_issue_number = check_for_duplicate_issue(g, title, issue_number, repository)
        if duplicate_issue_number:
            print(f"Issue #{issue_number} is a duplicate of Issue #{duplicate_issue_number}.")
            # "Duplicate" ラベルを追加し、コメントを投稿してクローズ
            add_duplicate_label_and_comment(issue, duplicate_issue_number)
            continue  # 次の Issue をチェック

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
        issue.add_to_labels("In Progress")

        try:
            # モデルの変換を実行
            cmd = f"python convert_model.py --model '{model_name}' --upload"
            print(f"Running command: {cmd}")
            process = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Model conversion failed for {model_name}")
            # エラーログを取得
            error_log = e.stderr if e.stderr else str(e)

            # ジョブのURLを生成
            job_url = f"https://github.com/{repository}/actions/runs/{run_id}"

            # Issue にコメントと「Failed」ラベルを追加
            comment_body = f"モデル **{model_name}** の変換中にエラーが発生しました。\n\nエラーログ:\n```\n{error_log}\n```\n\n[ジョブの詳細はこちら]({job_url})"
            issue.create_comment(comment_body)
            issue.remove_from_labels("In Progress")
            issue.add_to_labels("Failed")

            found_issue_to_process = True  # 処理した Issue がある
            break  # 処理を終えてループを抜ける

        # モデルのアップロードが成功したら、Issue にコメントしてラベルを更新してクローズ
        # アップロード先のURLを生成
        repo_name = f"{safe_model_name}-ONNX-ORT"
        encoded_repo_name = urllib.parse.quote(repo_name)
        uploaded_model_url = f"https://huggingface.co/ort-community/{encoded_repo_name}"

        comment_body = f"モデル **{model_name}** の処理が完了しました。\n\nアップロード先: [{uploaded_model_url}]({uploaded_model_url})"
        issue.create_comment(comment_body)

        # ラベルの更新
        issue.remove_from_labels("In Progress")
        if "Failed" in [label.name for label in issue.labels]:
            issue.remove_from_labels("Failed")
        issue.add_to_labels("Completed")

        # Issue をクローズ
        issue.edit(state="closed")

        found_issue_to_process = True  # 処理した Issue がある
        break  # 処理を終えてループを抜ける

    if not found_issue_to_process:
        print("処理すべきモデル変換の Issue はありません。")

def check_for_duplicate_issue(g, title, current_issue_number, repository):
    # タイトルに含まれる単語で検索範囲を狭める（不要な場合があるため、場合によっては削除してもよい）
    keywords = title.strip().split('/')
    query = f'repo:{repository} is:issue is:closed'
    for keyword in keywords:
        query += f' {keyword} in:title'

    issues = g.search_issues(query)
    for issue in issues:
        issue_number = issue.number
        issue_labels = [label.name for label in issue.labels]
        issue_title = issue.title.strip()
        # タイトルが完全一致し、"Completed" ラベルが付いている Issue を対象とする
        if issue_title == title.strip() and "Completed" in issue_labels and issue_number != current_issue_number:
            return issue_number  # 重複する過去の Issue の番号を返す

    return None  # 重複する Issue が見つからなかった

def add_duplicate_label_and_comment(issue, duplicate_issue_number):
    # "Duplicate" ラベルを追加
    issue.add_to_labels("Duplicate")
    # コメントを投稿
    comment_body = f"この Issue は過去の Issue #{duplicate_issue_number} と重複しています。そちらをご参照ください。"
    issue.create_comment(comment_body)
    # Issue をクローズ
    issue.edit(state="closed")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_issues.py '<repository>' '<run_id>'")
        sys.exit(1)
    repository = sys.argv[1]
    run_id = sys.argv[2]
    main(repository, run_id)