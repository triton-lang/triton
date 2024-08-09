#### Agenda:
1. Triton-CPU Update
2. Intel GPU backend update

##### Items:
Meeting notes:
1. Triton-CPU Update: Intel and Meta jointly presented the work on Triton-CPU, highlighting good progress on coverage and performance improvements. They also covered some of the optimizations they leveraged to get performance comparable to torch-native and torch-inductor. More details are in their slides.
2. Intel GPU Backend: Intel GPU backend shows good performance close to expert-tuned kernels and the use of block pointers for performance gains. There were questions around the future of block pointers and their importance for performance gains. With block-pointer deprecation there is a need for a more generic interface to support various backends including Intel GPU.
3. The 2024 Triton conference is on September 17th 2024 in Fremont California! Please register [here](README.md).
##### Minutes:
Recording link [here](https://youtu.be/dfL3L4_3ujg)

Presentations repo [here](https://drive.google.com/drive/folders/1fQ3zVrM7DT8W8FGJWKx1wNr2X53tYbeT?usp=sharing)
