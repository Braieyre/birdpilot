import time
import onnxruntime as ort
import numpy as np

ONNX_PATH = "outputs/mobilenetv3_birds.onnx"

session = ort.InferenceSession(ONNX_PATH)

x = np.random.randn(1, 3, 224, 224).astype(np.float32)

# warmup
for _ in range(10):
    _ = session.run(None, {"input": x})

# benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    _ = session.run(None, {"input": x})
    end = time.perf_counter()
    times.append((end - start) * 1000)

avg_ms = sum(times) / len(times)
fps = 1000 / avg_ms

print(f"Average latency: {avg_ms:.2f} ms")
print(f"Approx FPS: {fps:.2f}")