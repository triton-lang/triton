# Agenda:
1. What are the plans for existing block pointer programming model? (Context: Intel GPU backend relies heavily on it an will need time to fully move to tensor descriptor programming model) - Jianhui Li (Intel)
2. Infrastructure for Triton performance tests - Sayce Falk (Google)
3. What talks/tutorials/open discussions would you like to see at the 2025 Triton Developers' Summit? How can we help? Adnan Aziz (Meta)

# Notes:

## What are the plans for existing block pointer programming model? (Context: Intel GPU backend relies heavily on it an will need time to fully move to tensor descriptor programming model)
Speakers: Jianhui Li (Intel), Keren Zhou (George Mason Univ)

* Glad to see Triton moving toward generic tensor descriptor vs vendor-specific TMA.
* Intel is still relying on older block pointer programming model. Will take some time to migrate to new tensor descriptor model

### Questions
* Q> What is timeline for deprecation of block pointer?
* Q> Looked at code examples. Two flavors of tensor descriptor. We'd prefer keeping one: **CreateTensorDescriptorFromHost** Why are there two flavors?  WHy not just keep the device side one?
* A> You want to know why we have one device side and one host side.
* Q> Ok to have tensor descriptors in global memory. We want tensor descriptors to reside on the device.
* A> We have descriptor API on device because when you update the descriptor from the kernel and not from the device.
* Q> Performance. Would like to limit choices to programmer. Don't need to enable other programming models. Makes it easier to support triton on other platforms.
* A> Is it a problem if you only support device side descriptor and update?
* Q> No.
* A> Probably still need to keep 2 APIs.
* Q> What do other vendors think?
* A> Try the tutorial 0.9. Exercises differ tensor descriptor APIs demostrating different performance characteristics.
* Q> OpenAI support both APIs? on the device and the off-site?
* A> Yes
* Q> Removing support for block pointers
* A> Yes, I'm proposing removing block pointers from triton. Tensor descriptor support all use-cases covered by block pointers.
* Q> I've got a GEMM kernel written with block pointers, rewrote using on-device tensor descriptors and it works. Tensor descriptor doesn't have the offset information on the load, we need to look at the load & tensor descriptor to materialize the block pointer. Works interprocedurally because we can reconstruct the block pointer in the same function. Intra procedurally, problematic, tensor descriptor is only in caller, not the callee (info not available to do reconstruction in callee)
* A> Calling convention is a bit confusing if using non-inline functions.
* Q> Concerning because we're using a lot of block pointers.
* Q> We're also heavy users of block pointers and have wrappers on both APIs (creates either a block pointer or a tensor descriptor.)  Block pointer is superset of tensor descriptor. Just carry load params in a tuple. Limitation though. Least significant stride must be 1. All other strides must be a multiple of 16. No performance sensitive stuff using this. We use block pointers for some small writes and these aren't supported by TMA.
* A> Block pointers can't just be lowered to TMA. We want intermediate passes that translate it into something similar to block pointers.
* Q> If CMA incompatible, would be lowered to TMA.
* A> Talked to Peter, no time to work on this.
* Q> We don't mind what API. What is the transition plan for block pointer API? Timeline?
* A> No timeline yet.
* Q> Need a grace period.

## Infrastructure for Triton performance tests
Speaker: Sayce Falk (Google), Cicie Wang (Meta), Jason Knight (Nvidia), Keren Zhou (George Mason University), Areg Melik-Adamyan (Intel)

* Q> Any near term plans for setting up public benchmarks for Nvidia's newest hardware? Maybe through PyTorch or TorchBench.
* A> Cicie Wang (Meta): Meta discussed with Nvidia about running TritonBench on B200. Nvidia suggested working with OpenAI (OpenAI has hardware). We now have hardware. Jason from Nvidia working on setting up CI. First steps: get TritonBench running on this hardware.
* Q> Need devops/infra side to setup devrunners (complexity/security of setting up these machines is high). Possible to use existing GB200 triton runner in triton CI.
* Q> You want to run torchbench? Is this on the triton main project?
* A> Possibly using the facebookexperimental/triton repo. Maybe a second repo. Maybe the PyTorch repo?
* A> Also looking at the AMD MI300x and AMD MI350x.
* Q> Xu Zhao (Meta) is currently running triton bench.
* A> Yes. But only for internal Meta consumption. Goal is to expose this externally.
* Q> Maybe we can leverage Intel's backend? (to Jason Knight).
* A> We currently have OpenAI's hosted triton CI, PyTorch's CI & performance.
* Q> Intel has its on repo. Interested in contributing data to a shared dashboard.
* A> Maybe talk to the PyTorch folks
* A> DevOps support not up and running (months out) for B200.
* Q> Where are the B200s hosted?
* A> Pytorch foundation: all cloud instances funded by credits (Top N cloud providers). CI for Triton.
* A> Blackwell is in house for Triton.  We'd like have better sources (only one node per type for testing.)
* Q> Jason do you have local hosted cloud?
* A> Yea, but security is hard.
* Q> Progress on PyTorch foundation to get DevOps (Meta needs to look into this).
* Q> More interested in regression testing.  Are you finding regressions?
* A> Intel is usually not seeing regressions from OpenAI (because they only have a 1 week lag).
* Q> Google XLA experience - could you set this up?
* A> Yes, we could talk through personnel/resourcing but need to know what community goals are.
* Q> Some performance tests, some regression tests to start. (Including Llama 4 and MoE operators).
* Q> What kernels and operators should block releases?
* Q> Intel would be interested in developing common benchmarking infrastructure.
* Q> Intel would be interested regression testing infrastructure.
* Q> Interested in collaborating on developing tests that don't just look at lit-like tests but how do changes in passes affect generated code.
* Q> Anyone interested in this?
* A> Maybe first step, identify how much generated code is affected by a pull request (give a signal to say something about the blast radius of a change).
* Q> Intel had an intern looking at this.
* Q> Intel<Alexander> - if you're interested reach out over slack.

## What talks/tutorials/open discussions would you like to see at the 2025 Triton Developers' Summit? How can we help?
Speaker: Adnan Aziz (Meta)

* Phil, Elena Mithra & Adnan Aziz pulled together last year's Triton Developers' Summit.
* Mlir tutorials, keynotes, closed-end backends, OSS projects, Intel triton efforts.
* Heterogeneous hardware.
* Over 500 people attended!
* Microsoft running it in 2025.
* Ideas:
  * Tutorials for users: writing triton code, kernel profilers
  * Panel of triton users: power users and new users.
  * Keren: academic/scientific domains. Physicists are using triton for simulations. Broader HPC.
  * Jason: EVO and mosaic talks (embracing sharing). Cutlass dsl, we should be learning form them.
  * Cicie: do we have proposal submission process? No. We had a compressed timeframe-10 weeks. Some proposals didn't make it due to time.
* Please give us feedback.
* We promised to give Microsoft feedback to the process.
* Triton summit will try to colocate with PyTorch conference.  Probably at the Mosconi Center in SF (but still needs to be verified from Microsoft).
* What is Microsoft's timeline/plans?

##### Minutes:
Recording link [here](https://youtu.be/W16BrXc5BYE)
