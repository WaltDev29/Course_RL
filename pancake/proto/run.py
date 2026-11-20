"""팬케이크 뒤집기 RL 대시보드 실행 진입점."""
import os
import sys

PROTO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROTO_DIR)
sys.path.insert(0, os.path.join(PROTO_DIR, "ui"))
sys.path.insert(0, os.path.join(PROTO_DIR, "utils"))
sys.path.insert(0, os.path.join(PROTO_DIR, "envs"))
sys.path.insert(0, os.path.join(PROTO_DIR, "training"))

from ui.dashboard import Dashboard

def main():
    app = Dashboard()
    app.mainloop()

if __name__ == "__main__":
    main()
