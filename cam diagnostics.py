import cv2, time

def fourcc_str(cap):
    v = int(cap.get(cv2.CAP_PROP_FOURCC)) & 0xFFFFFFFF   # mask off the sign
    return "".join(chr((v >> (8*i)) & 0xFF) for i in range(4)) if v else "----"

def probe(index, backend, w, h):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"backend={backend}: failed to open"); return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # request MJPG
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(10): cap.read()                     # warm up
    t, n = time.time(), 0
    while n < 60:
        ok, _ = cap.read()
        if not ok: break
        n += 1
    dt = time.time() - t
    print(f"backend={backend:>4}  ask {w}x{h}  got {int(cap.get(3))}x{int(cap.get(4))}  "
          f"fourcc={fourcc_str(cap)}  measured_fps={n/max(dt,1e-6):.1f}")
    cap.release()

for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
    for (w, h) in [(1280, 720), (1920, 1080)]:
        probe(0, backend, w, h)
    print()