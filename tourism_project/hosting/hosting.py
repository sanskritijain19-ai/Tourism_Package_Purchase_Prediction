from huggingface_hub import HfApi, login
import os

#api = HfApi(token=os.getenv("HF_TOKEN"))
#api = HfApi(token=os.getenv("HF_TOKEN"))
token = os.environ.get("HF_TOKEN")

if token:
    token = token.strip()
    token = token.replace("\n", "").replace("\r", "")

login(token=token)
api = HfApi()

api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="sanskritijain27/Tourism-Package-Purchase-Prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
