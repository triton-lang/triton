#### Agenda:

##### Items:
1. 3rd party refactoring backend update.
2. AMD update about experience with refactored backend and new process.
3. Plan to restore the Intel XPU backend as third-party module.
4. Open discussion.

##### Minutes:
Recording link [here](https://youtu.be/uRlqolhNbRk)

1. 3rd party refactoring backend update.
   - Backends are passes and IRs are shared by the backends to avoid divergence and duplications so that developers do not have to change the Triton source code
   - To discover backend forks in directories, put environment vars in setup.py.
   - Backends can link whatever library they want, they don’t need to copy paste Nvidia code.
   - Nvidia uses the same API as other backends, (refactoring of the C++ code is still remaining). No special casing for Nvidia code.
   - If Triton dependency is on top of the main branch then it will work for forks/branches.
   - Still remaining: LLVM IR conversion – reusuable pattern rewriters update; Reduce complexity in statefulness in Triton GPU - inherit from base pattern
2. AMD update about experience with refactored backend and new process.
   - Skipped due to lack of time. Will be covered in February meetup
3. Plan to restore the Intel XPU backend as third-party module.
   - Prereqs to upstream – Will take into account the system HW and SW, with perf to be ~80% of Nvidia, to allow upstreaming.
   - Consider how useful it is for AI research to allow upstreaming – as it impacts maintenance cost of the backends.
   - Don’t have plans to upstream mobile backends
   - Intel will hold offline discussion with Open AI for being in-tree.
