## 🚧 Background

Triton officially targets NVIDIA GPUs with **SM 8.0+**, leaving Turing (SM 7.5) in an awkward state:

`tl.dot` is effectively degraded or unusable:

- FP16 → silently lowered to CUDA Core FMA (~2× slower)
- INT8 → compilation failure  
- Tensor Cores exist on Turing — but remain untapped

This is not a hardware limitation, but a **missing compiler path**.

---

## 🔍 Reality Check

Despite having Tensor Cores, RTX 20-series GPUs are currently:

> “Tensor Core-capable, but Triton-disabled.”

Relevant issues reflect this gap:

- #189 (2021, still open): Low RTX20 Tensor Core performance  
- #9349 (2025, wontfix): `tl.dot` failed for INT8 on Turing  

---

## 💡 This Project

This repository is a Triton fork that brings Turing back into the game:

- 🔓 Re-enables MMA (Tensor Core) support for SM75  
- 🧠 Integrates into Triton’s existing pipeline  
- 🚫 No separate backend, no hacks — just filling the missing piece