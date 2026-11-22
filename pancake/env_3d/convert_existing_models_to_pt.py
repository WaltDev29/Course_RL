import os, sys, shutil
import torch
from stable_baselines3 import SAC

ENV3D_DIR = r"d:\course\RL\pancake\env_3d"
models_dir = os.path.join(ENV3D_DIR, "models")

for d in os.listdir(models_dir):
    full_d = os.path.join(models_dir, d)
    if not os.path.isdir(full_d):
        continue
    zip_p = os.path.join(full_d, "model.zip")
    pt_p = os.path.join(full_d, "model.pt")
    policy_pt_p = os.path.join(full_d, "policy.pt")

    if os.path.exists(zip_p):
        print(f"Processing model folder: {d}")
        # 1. model.pt 복사/저장
        if not os.path.exists(pt_p):
            shutil.copyfile(zip_p, pt_p)
            print(f"  -> Created model.pt")
        # 2. policy.pt 저장
        if not os.path.exists(policy_pt_p):
            try:
                model = SAC.load(zip_p, device="cpu")
                torch.save(model.policy.state_dict(), policy_pt_p)
                print(f"  -> Created policy.pt (PyTorch state_dict)")
            except Exception as e:
                print(f"  -> Failed to save policy.pt: {e}")

print("All existing models processed successfully!")
