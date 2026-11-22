"""3D 팬케이크 뒤집기 RL 대시보드 실행 진입점."""
import os
import sys

ENV3D_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ENV3D_DIR)
sys.path.insert(0, os.path.join(ENV3D_DIR, "ui"))
sys.path.insert(0, os.path.join(ENV3D_DIR, "utils"))
sys.path.insert(0, os.path.join(ENV3D_DIR, "envs"))
sys.path.insert(0, os.path.join(ENV3D_DIR, "training"))

from ui.dashboard_3d import Dashboard3D

def main():
    app = Dashboard3D()
    app.mainloop()

if __name__ == "__main__":
    main()
