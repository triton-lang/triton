# Problem Statement

The current limitations of metaprogramming in Triton have led major users, such as Torch Inductor, to resort to using string-based templating. This RFC aims to address some of these limitations by extending Triton's metaprogramming capabilities.

I also found several performance issues (like backtracking codegen) with the current code generator that I intend to fix.

# Current metaprogramming limitations
Except for simple assignments that are marked constexpr, if conditions and simple loops it's not possible to embed python expressions inside triton.
Current design relies on interpreting python expressions inside `CodeGenerator`. This approach is inherently limited because it's not possible to have good metaprogramming support without building a full python interpreter inside this class.

This proposal also makes it possible to use `while` loops and use `for` loops with arbitrary iterators for metaprogramming.

# Proposal Overview
I propose that instead of converting the Python AST directly to Triton IR, we make a code generator generator that for a given Triton AST generates a function of tensor argument types and constant args that returns a Triton IR function as a result. I also propose a technique to differentiate Triton expressions from python metaprogramming expressions.

This approach allows you to **embed any Python expression you want inside Triton**.

**Input Triton Function**
```python
@triton.jit(use_experimental_frontend=True)
def test_function(test_arg, count: tl.constexpr):
    for a in ["arbitrary", "iterable"]
        while test_arg < count:
            test_arg += 1
        test_arg += 1

```
**Resulting Function Generator for the Triton Code(exec'ed)**
```python
ddef test_function(builder, module, test_arg, count: tl.constexpr):
    _prototype = tl.function_type([], [test_arg])
    _ret_val = None
    function = builder.get_or_insert_function(module, 'test_function', tl.function_type([], [test_arg]).to_ir(builder), 'public', False)
    module.push_back(function)
    entry = GenScope(builder, None, block=function.add_entry_block())
    with entry:
        entry.define('test_arg', tl.tensor(function.args(0), test_arg))
        for a in ['arbitrary', 'iterable']:
            _while_op_0 = WhileOp(builder, entry, ['test_arg'])
            with _while_op_0.Cond() as __scope_0:
                _while_op_0.condition(__scope_0.resolve('test_arg').__lt__(count, _builder=builder))
            with _while_op_0.Body() as __scope_1:
                __scope_1.define('test_arg', __scope_1.resolve('test_arg').__add__(1, _builder=builder))
            _while_op_0.finalize()
            entry.define('test_arg', entry.resolve('test_arg').__add__(1, _builder=builder))
        if not _ret_val:
            builder.ret([])
    function.reset_type(_prototype.to_ir(builder))
    function.finalize()
```

## Code Generator generation from Triton AST
At this stage we process the Python AST and generate a new python function that will generate the triton IR. We also do loop-carried variables analysis to later make it easier to construct the SSA correctly. 

### Separating Triton and Python expressions from each other

To distinguish between Triton expressions and Python metaprogramming expressions, we will use the following rules:

1. **Triton Function definition arguments:** We assume all arguments not marked as `tl.constexpr` are triton variables. 

2. **Binary expressions:** If the left or right part of a binary expression is a Triton expression, it's assumed to be a Triton expression.
 
3. **Control flow:**
   - If `if` or `while` blocks use Triton expressions as conditions, these are interpreted as Triton control flow blocks.
   - For `for` loops that iterate over Triton iterables, the loop is considered a Triton `for` loop.

4. **Function Calls:** Function calls that are going to Triton builtins and other Triton functions are considered to be Triton expressions.
One exception to this rule is `min` and `max`, for those functions we look at the arguments and assume the expression is a Triton expression if any of the arguments is a trition expression.

We use the global scope of the function to resolve things like `tl.full((1,), 1.0, tl.float32)` the `PseudoInterpreter` class uses the global scope to resolve the called function. Limitations of this approach, which I think won't affect backwards compatibility, are discussed later.

For builtins we inject the `_builder` keyword argument to the call (note: the PoC does not currently support _generator arg, this breaks reductions. This limitation will be addressed later)

5. **Assignments:** Triton only supports simple assignments of the form `name = ...`. Any more complex expressions are considered Python metaprogramming expressions. A simple assignment is considered a Triton assignment if one of the following is true:
   - The variable `name` is recognized as a Triton variable.
   - The value being assigned is a Triton expression.

Most of these rules are implemented in the `ExpressionAnalyser`.

# Kernel Launch performance
Triton must generate different kernels for different constant expressions and call argument types. Generating a function to generate the IR moves some work from kernel launch time to code initialization time. Python Bytecode interpretation will have better performance compared to AST based interpretation done by the older approach.

Also, old code generator had a bad backtracking behaviour that the new code generator fixes.

Old code:
```python 
    # create loop body block
    block = self.builder.create_block()
    self.builder.set_insertion_point_to_start(block)
    # dry visit loop body
    self.scf_stack.append(node)

    # This is kinda expensive, specially for nested loops 
    self.visit_compound_statement(node.body)
    self.scf_stack.pop()
    block.erase()
```
Old generator does this to find loop carried variables and construct SSA correctly. Instead of compiling the loop body twice my approach patches the generated AST after compiling the loop body.

Return support is not complete yet but,  I intend to also add caching to ContainsReturnChecker.

# Status of Implementation

You can try the PoC implementation like this:
```python
@jit(use_experimental_frontend=True)
def my_kernel():
    ...
```

Along side this RFC, a PR with a PoC implementation of my ideas is included, I believe it will be enough to demonstrate the bulk of my ideas. 
I already invested **a lot** of time without getting any feedback from the community.

I am already aware that there are some features that are not supported in the experimental frontend, I don't have a complete list of missing features as of now but here are some known ones:

- [ ] Return values [WIP]
- [ ] Argument specilization 
- [ ] Non SCF if blocks (top_level) [WIP]
- [ ] Subscripting
- [ ] Support Calling User defined functions [WIP]
- [ ] Source code location annotation in the IR
- [ ] Port some of the error checking code (like verifying types of loop carried variables).
- [ ] Fix Reductions (_generator interface) [WIP]
- [ ] Support for F-string expressions inside Triton expressions, used for print
- [ ] Support older python versions(< 3.8)

# Discussion Questions 

I would like some feedback from community about overriden builtins. My goal with this proposal was to turn Triton into a superset of Python, but triton overriding the behaviours of some builtins makes that k. 

- As noted earlier we have special rules for `min` and `max`. We can implemenet a similar rule for `print` (by default overriden with device_print) but since it has side effects (printing to the console) it matters if we run it in code generation time or code execution time (eg, in the gpu)

- Functionality of `range` is also overriden by triton. Assuming all `range`s with non-Triton arguments are python expressions would be like fully unrolling them, which would not be desirable.