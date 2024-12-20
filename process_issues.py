import json
import subprocess
import sys
import os
import urllib.parse

def main(issues_json_path, repository, run_id):
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
    for issue in issues:
        title = issue['title']
        issue_number = issue['number']
        print(f"Processing Issue #{issue_number}: {title}")

        # タイトルからモデル名を抽出
        # 先頭の「xxx/」を削除してモデル名を取得
        if '/' in title:
            model_name = title.strip()
            repo_model_name = title.split('/', 1)[1].strip()
        else:
            model_name = title.strip()
            repo_model_name = model_name

        # モデル名を安全なリポジトリ名に変換
        safe_model_name = repo_model_name.replace('/', '_')  # スラッシュをアンダースコアに置換

        # Issueに「In Progress」ラベルを追加
        add_label_cmd = f'gh issue edit {issue_number} --add-label "In Progress"'
        subprocess.run(add_label_cmd, shell=True)

        # モデルの変換を実行
        cmd = f"python convert_model.py --model '{model_name} --upload'"
        print(f"Running command: {cmd}")
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if process.returncode != 0:
            print(f"Model conversion failed for {model_name}")
            # エラーログを取得
            error_log = process.stderr

            # ジョブのURLを生成
            job_url = f"https://github.com/{repository}/actions/runs/{run_id}"

            # Issueにコメントと「Failed」ラベルを追加
            comment_body = f"モデル **{model_name}** の変換中にエラーが発生しました。\n\nエラーログ:\n```\n{error_log}\n```\n\n[ジョブの詳細はこちら]({job_url})"

            # コメントをファイルに書き出す
            with open('comment_body.txt', 'w', encoding='utf-8') as f:
                f.write(comment_body)

            comment_cmd = f'gh issue comment {issue_number} --body-file "comment_body.txt"'
            subprocess.run(comment_cmd, shell=True)
            label_cmd = f'gh issue edit {issue_number} --remove-label "In Progress" --add-label "Failed"'
            subprocess.run(label_cmd, shell=True)
            continue

        # モデルのアップロードが成功したら、Issueにコメントしてラベルを更新してクローズ
        # アップロード先のURLを生成
        repo_name = f"{safe_model_name}-ONNX-ORT"
        encoded_repo_name = urllib.parse.quote(repo_name)
        uploaded_model_url = f"https://huggingface.co/ort-community/{encoded_repo_name}"

        comment_body = f"モデル **{model_name}** の処理が完了しました。\n\nアップロード先: [{uploaded_model_url}]({uploaded_model_url})"

        # コメントをファイルに書き出す
        with open('comment_body.txt', 'w', encoding='utf-8') as f:
            f.write(comment_body)

        comment_cmd = f'gh issue comment {issue_number} --body-file "comment_body.txt"'
        subprocess.run(comment_cmd, shell=True)
        label_cmd = f'gh issue edit {issue_number} --remove-label "In Progress" --add-label "Completed"'
        subprocess.run(label_cmd, shell=True)
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