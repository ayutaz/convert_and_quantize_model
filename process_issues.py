import json
import subprocess
import sys
import re
import os
import urllib.parse

def main(issues_json, repository, run_id):
    issues = json.loads(issues_json)
    username = os.getenv('USERNAME')  # ユーザー名を環境変数から取得
    for issue in issues:
        title = issue['title']
        issue_number = issue['number']
        print(f"Processing Issue #{issue_number}: {title}")

        # タイトルからモデル名を抽出
        # 先頭の「xxx/」を削除してモデル名を取得
        if '/' in title:
            model_name = title.split('/', 1)[1].strip()
        else:
            model_name = title.strip()

        # モデル名を安全なリポジトリ名に変換
        safe_model_name = model_name.replace('/', '_')  # スラッシュをアンダースコアに置換

        # Issueに「In Progress」ラベルを追加
        add_label_cmd = f'gh issue edit {issue_number} --add-label "In Progress"'
        subprocess.run(add_label_cmd, shell=True)

        # モデルの変換を実行
        cmd = f"python convert_model.py --model '{model_name}'"
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
        repo_name = f"{safe_model_name}_onnx_ort"
        encoded_repo_name = urllib.parse.quote(repo_name)
        uploaded_model_url = f"https://huggingface.co/{username}/{encoded_repo_name}"

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
        print("Usage: python process_issues.py '<issues_json>' '<repository>' '<run_id>'")
        sys.exit(1)
    issues_json = sys.argv[1]
    repository = sys.argv[2]
    run_id = sys.argv[3]
    main(issues_json, repository, run_id)