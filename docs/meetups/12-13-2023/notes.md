#### Agenda:

##### Items:
1. Refactoring plan for 3rd party backends
2. Front end refactoring (AMD)
3. Things like block pointers, ptr_analysis, mask_analysis can be used for GPUs, is there a plan to incrementally include components from Triton shared for GPU development.

##### Minutes:
Recording link [here](https://youtu.be/Lo43DQYkOWM)

1. Refactoring plan for 3rd party backends
   - Refactoring to be completed by end of the year so that all GPU backends can be individual passes on Triton GPU IR instead of being completely out of tree. The goal is for users to get other GPUs besides Cuda when they install Triton. Non-GPU Triton IR expected to stay as is.
3. Front end refactoring (AMD)
   - Will work with Phil for AMD related refactoring. Will share more details in next meetup about where AMD has diverged from Triton GPU IR and in the codeflow.
4. Things like block pointers, ptr_analysis, mask_analysis can be used for GPUs, is there a plan to incrementally include components from Triton shared for GPU development.
   - Can look at it on a case by case basis.
