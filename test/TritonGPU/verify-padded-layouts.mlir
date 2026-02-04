// RUN: triton-opt --split-input-file %s --verify-diagnostics

// expected-error@+1 {{interval values must all be power of two}}
#shared = #ttg.padded_shared<[3:+2] {offset=[[0]], block=[]}>

// -----

// expected-error@+1 {{interval values must all be power of two}}
#shared = #ttg.padded_shared<[0:+2] {offset=[[0]], block=[]}>

// -----

// expected-error@+1 {{padding values must all be power of two}}
#shared = #ttg.padded_shared<[2:+3] {offset=[[0]], block=[]}>

// -----

// expected-error@+1 {{padding values must all be power of two}}
#shared = #ttg.padded_shared<[2:+0] {offset=[[0]], block=[]}>

// -----

// expected-error@+1 {{interval values cannot have duplicates}}
#shared = #ttg.padded_shared<[2:+1, 2:+4] {offset=[[0]], block=[]}>

// -----

// expected-error@+1 {{Unexpected attribute}}
#shared = #ttg.padded_shared<[2:+1, 4:+2] {unknown = 5}>

// -----

// expected-error@+1 {{Unexpected attribute "order" found}}
#shared = #ttg.padded_shared<[2:+1, 4:+2] {offset = [[1, 0], [2, 0]], block = [], order=[0, 1]}>

// -----

// expected-error@+1 {{Each offset basis must be 0 or a power of two}}
#shared = #ttg.padded_shared<[2:+1, 4:+2] {offset = [[1, 0], [3, 0]], block = []}>

// -----

// expected-error@+1 {{Unexpected attribute "register" found}}
#shared = #ttg.padded_shared<[2:+1, 4:+2] {order = [1, 0], register = [[0, 1], [0, 2]]}>

// -----

// expected-error@+1 {{Expected basis of 'block' not found}}
#shared = #ttg.padded_shared<[2:+1, 4:+2] {offset = [[1, 0], [1, 1]]}>

// -----

// expected-error@+1 {{Expected basis of 'block' not found}}
#shared = #ttg.padded_shared<[2:+1, 4:+2] {offset = [[0 , 1]]}>

// -----

// expected-error@+1 {{Expected basis of 'offset' not found}}
#shared = #ttg.padded_shared<[2:+1, 4:+2] {block = [[0 , 1]]}>

// -----

// expected-error@+1 {{Broadcasting in offset dimension is not supported.}}
#shared = #ttg.padded_shared<[2:+1] {offset = [[0]], block = []}>

// -----

// Broadcasting in block dim is allowed
#shared = #ttg.padded_shared<[2:+1] {offset = [[1, 0], [0, 1]], block = [[1, 0], [2, 0]]}>
