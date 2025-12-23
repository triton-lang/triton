# Agenda:
* Community discussion:  *Gluon, TLX, CuTeDSL, cutile, tileIR etc. ... with so many choices, how do I decide on what I should use to write my next kernel/model*
* Post Triton Conference discussion:
    * Ofer: recap of the event.
    * What did you like
    * What was shocking
    * What would you like to see more of/less of next year.
* Flex Attention questions - (Whitney, Intel)

# Notes:
* Post Triton Conference discussion:
    * Luka - Liked the breadth and interest in Triton, extensions and examples. Liked talks on warp specializaiton. Interestes: vLLM,  torch.compile() and  abstractions.
    * Simon Waters, kernelize.ai - Lots of great content. Next time, try and get presentations on the big screen center stage.
    * Bryan Bowyer, kernelize.ai - Liked the step by step walk throughs. Lets you see exactly how to use Triton/extensions. Would like to see more talks about novel AI hardware. Knows more devices are ready. Would like to see more Triton demos/especially hardware demos.
    * Puyan Lotfi, Meta - Also saw good talks at [PTC 2025](https://pytorch.org/event/pytorch-conference-2025/) & [2025 LLVM Developers Meeting](https://llvm.swoogo.com/2025devmtg/home)- quite a few DSL extensions for more hardware features. Would like a more unified extension system. Proposed/saw an interesting idea: creating an MLIR dialect that doesn’t take fixed sized tensors, imbeds them in inline assembly.  Maybe we could do this in Triton.
    * Sara - Enjoyed presenting posters with colleagues. Liked Helion talk. Looking at Helion tutorials now. Interested in Triton kernels for vLLM and deploying to different hardware platforms (Nvidia, AMD and ???)
    * Corbin Robeck, Meta - is working on Triton extensions. Currently reviewing proposals from teams interested in adding distributed Triton, Triton for different architectures (integrated in an extension). Looking for mostly mature implementations. He's currrently in the process of open sourcing this extension framework.
    * Dhruva Kaushal, Meta - Flex attention make the attention context parallel (Monarch announcement), Pytorch support for different data types MXFP8 and NVFP4, can Triton adopt and emulate these.
    * Jason Furmanek, AMD - AMD sharing some of their latest improvements (e.g. performant flash attention on MI350s) at both Triton conference and PTC.
    * Hongtao Yu, Meta - Liked seeing kernel performance numbers on AMD and GPU platforms, Triton DSL, understanding what the hard blockers are for customers adopting these DSLs. Happy to see more people using Triton and building more Triton libraries.
    * Jamie Yang - Seeing some divergence in the ML compiler landscape, of the different levels of abstraction, which will survive? He's seeing attempts to do similar things as [Triton-distributed](https://arxiv.org/abs/2504.19442) like what Meta is doing. Will they converge?  Interested in vLLM gpu kernels like llama 2 in Triton.
    * Jie Liu, Meta - Talks on Nvidia Blackwell extension & abstractions were good.  ByteDance talk was good (nice to see presentations).  Would like to see a panel discussion. Suggested topics: common concerns & directions and collaboration and brainstorming. Interested in: optimizing Blackwell attention & automatic warp specialization (that is, the compiler should handle partitioning and scheduling.)
    * Keshav Singh - Thought presentations were insightful. Liked that he could review them online.  Interested in non-transformer models. Disappointed that there aren't a lot of good example kernels though.
    * Kuy Mainwaring, Google - Leads XLA effort at Google. He's an unusual user of Triton. They generate Triton IR! He's interested in AMD & Nvidia roadmaps. Wants to know what is the evolving future of these architectures. Where is Triton is going in the future?  Interested in families of templates, attention masking, scaling topologies. Currently, Google's TPUs aren’t supported by Triton. There are quantization schemes that are unique to TPUs... how to map from one to another?  They want to be sure that Gemini works well on GPUs. Examples include INT4 dtype and proprietary data types, looking at normalization diamonds and softmax. Currently, XLA runs on many platforms. Maybe we could have covolution in Triton?
    Ettore Tiotto, Intel - more important Jason’s talk on Helion, because triton is only mostly portable.  Intel has AMD, OAI doesn’t care about Intel.  MSFT asked how AMD got its backend into.  Get more backends into OpenAI community.  How to get its backends into triton.  Would like an easyway to push a plugin. (Reach out to Corbin Robeck
    * Luka Govedic - I'd like to make this more of a community similar to vLLM. Triton doesn't support plugable backends. Would like to do something like vLLM where Huawei and other companies can add their own backends. You shouldn't need to fork to support a new backend.

* Community discussion:  "Gluon, TLX, CuTeDSL, cutile, tileIR etc. ... with so many choices, how do I decide on what I should use to write my next kernel/model"
    * Hongtao Yu, Meta - Most people start with Triton. Once they get a kernel that does functionall what they want, they then think about performance. Typically, they try optimizations directly available in Triton. Some customers will go directly to cutlass/CuTeDSL. Scheduling is usually a question that drives this choice (how soo do you need it and what is acceptable performance). Other critera folks use when deciding on what language/framework to pick include: feature completeness and maturity.  Is the language/framework in startup phase, are there teams using/supporting it, is it still evolving.
    * Minjang Kim, Meta - Has similar concerns. Our customers want hardware heterogeneity but the introduction of Nvidia Blackwell introduced lots of divergence in the codebase. The PyTorch org has voiced lots of concern about this. Tile-based programming is a good thing. We don’t know what the winner will be but we would hope the winner enables hardware portability.  Helion is a good approach.
    * Sara - Looking forward to trying them all out!
    * Prithvi Patel, Meta - The Triton/Helion/Gluon/etc. tutorials give me a good handle on how to use these languages.
    * Hongtao Yu, Meta - If you want to see performance numbers, Meta/tritonbench has benchmark numbers for cuDNN, gluon, and cutlass too.
    * Whitney Tsang, Intel - I could try all of them but its still not clear which one to pick. I'd like a better idea of what the future for each of these solutions looks like. I've heard TLX is temporary and should be gone. Is Gluon is expected to stay in place and never be replaced? What are the choices if you want 100% or 90% of the hardware limit? I'd like it if triton, as a whole, were better.
    * Hongtao Yu, Meta - Meta is still looking at making the compiler more intelligent.
    * Luka - Gluon is not a short term soluton. It is a lower level dialect meant to help compiler writers.  Nvidia demonstrated they can successly implement autoWS in Gluon.
    * Whitney Tsang, Intel - Gluon is used in OpenAI's production models.
    * Hongtao Yu, Meta - It depends on how the hardware is designed. If scheduling is better on chip, we won’t need to do it in software. Nvidia HW is super configurable but the HW can’t schedule efficiently.  Nvidia needs to invest more in hardware scheduling.  We'll be keeping an eye on this.
    * Whitney Tsang, Intel - Triton isn’t dead because PyTorch continues to use Triton.
    * Corbin Robeck, Meta - Triton and CUTLASS have different internal layout systems and debugging porting a CUTLASS kernel to Triton requires very solid knowledge of both. Writing a CuTeDSL kernel requires knowledge of the underlying CUTLASS layouts as well.
    * Jason Furmanek, AMD  - AMD likes Triton and gluon for empowering developers. The closer you get to the hardware, the more you’re locked in. What are benefits of a new DSL? Gluon allows you to go deeper than out-of-the-box Triton. The question is do we need another DSL? What is the niche? Are people going to use inductor or XLA?
    * Luka - Announced TileIR is going into the LLVM stack. It will be like PTX and can be compiled into something more portable.  Is AMD interested in supporting this?
    * Jason Furmanek, AMD - AMD hasn’t looked at this level, that is, layers below DSLs, lowering paths, etc. AMD relies on LLVM both for good and for bad. It would be interesting to standardize on a different backend.
    * Kui Mainwaring, Google - We want our customers to identify the best DSL for themselves.  Jax on GPUs uses a mixture of interface: foreign function calls to cutlass, pallas lowering to TPU and mosaicGPU to gpus. AMD uses pallas to lower too.
    * Bryan Bowyer, kernelize.ai - Everyone uses what they want. Do what you can to reuse what you can and don’t diverge too soon in the stack.

* What is the status of flex attention tensor descriptor? PR for flex attention in PyTorch created by Intel [Whitney Tseng, Intel]
    * Dhruva Kaushal, Meta - Saw the draft and commenting on it. Happy to see folks contributing to flex attention.
    * Whitney Tsang, Intel - Tensor descriptors are critical for Intel and Nvidia Blackwell. Can we change tutorials/etc. to use tensor descriptors?  .
    * Dhruva Kaushal, Meta - Please suggest changes to docs. If it improves performance, by all means please do.
    * Whitney Tsang, Intel - Any benchmarks on tensor descriptor vs regular pointer performance on non TMA hardware?
    * Dhruva Kaushal, Meta - No. Meta has benchmarks only for TMA hardware. Flex Attention for document Mask +30%-50% win. Sliding window, lower.
    * Ettore Tiotto, Intel - Tensor descriptors have more information than Tensor pointers. Pass exists to lower tensor descriptors to tensor pointers. Tensor descriptors should always have at least the same level of performance as tensor pointers on any architecture. Not true for Nvidia GPUs though! On Nvidia,indexes for offsets are 64-bit and tensor pointers use 32-bit (we should upstream this)

# Minutes
* Recording link [here](https://www.youtube.com/watch?v=gaP6PpfPiEk)
