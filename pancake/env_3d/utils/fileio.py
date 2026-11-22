"""프로세스 간 JSON 파일 교환 유틸 (워커 <-> 대시보드 공용).

Windows는 상대 프로세스가 파일을 열고 있으면 os.replace가 PermissionError로 실패하므로 재시도한다.
"""
import json
import os
import time


def write_json_atomic(path: str, obj: dict, retries: int = 10, wait: float = 0.05) -> bool:
    """임시파일에 쓴 뒤 교체. 끝까지 실패하면 False (호출 측이 학습을 중단하지 않도록 예외를 던지지 않음)."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(obj, fp, ensure_ascii=False)
    except OSError as e:
        print(f"[fileio] 임시파일 쓰기 실패 {tmp}: {e}")
        return False
    for _ in range(retries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(wait)
        except OSError as e:
            print(f"[fileio] 교체 실패 {path}: {e}")
            return False
    print(f"[fileio] 상대 프로세스가 파일을 점유 중이라 이번 갱신은 건너뜀: {path}")
    return False


def read_json(path: str) -> dict:
    """읽기 실패(쓰는 중/없음)는 빈 dict로 처리."""
    try:
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return {}
