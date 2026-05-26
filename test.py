from huggingface_hub import HfApi
import os

REPO_ID = "nbanfltgc/my-videos"
FOLDER = "/output_videos"

api = HfApi()

# create repo if missing
api.create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    exist_ok=True
)

for file in os.listdir(FOLDER):
    path = os.path.join(FOLDER, file)

    if os.path.isfile(path):
        print("Uploading:", file)

        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=file,
            repo_id=REPO_ID,
            repo_type="dataset"
        )

print("done")