import time
import argparse


# Custom ceil division function to mimic C++ behavior
def ceil_div(n: int, d: int) -> int:
    """Performs `(n + d - 1) / d`."""
    return (n + d - 1) // d


def num_iters_per_cta(m: int, n: int, k: int, g: int, blk_m: int, blk_n: int, blk_k: int) -> int:
    return ceil_div(ceil_div(m, blk_m) * ceil_div(n, blk_n) * ceil_div(k, blk_k), g)


def number_of_output_tiles(m: int, n: int, blk_m: int, blk_n: int) -> int:
    m_tiles = ceil_div(m, blk_m)
    n_tiles = ceil_div(n, blk_n)
    return m_tiles * n_tiles


def num_fixup_peers(k: int, iters_per_cta: int, blk_k: int) -> int:
    return ceil_div(ceil_div(k, blk_k), iters_per_cta)


def predicted_runtime(
    m: int,
    n: int,
    k: int,
    g: int,
    a: float,
    b: float,
    c: float,
    d: float,
    blk_m: int,
    blk_n: int,
    blk_k: int,
):
    iters_per_cta = num_iters_per_cta(m, n, k, g, blk_m, blk_n, blk_k)
    fixup_peers = num_fixup_peers(k, iters_per_cta, blk_k)

    runtime = (a + (b * (fixup_peers > 1)) + (c * iters_per_cta) + (d * (fixup_peers - 1)))
    return runtime, iters_per_cta, fixup_peers


def grid_model(
    m: int,
    n: int,
    k: int,
    blk_m: int,
    blk_n: int,
    blk_k: int,
    grid_start: int = 1,
    grid_end: int = 304,
    verbose: bool = False,
) -> int:

    # Fixed overhead alpha (a), fixed-size cost incurred by
    # each work-group, e.g. the grid launch latency, the initial
    # compulsary cache misses, the cost of writing the final output tile
    # to C.
    a = 5.04 + 8.30
    # Beta (b) incorporates conditional costs of outputting temporary partial
    # sums for scenarios where the number of output tiles does not quantize
    # perfectly across the number of processors.
    b = 5.47
    # c represents instruction and stall workload of each MAC-iteration.
    c = 4.17
    # Delta (d) is the cost of reading and accumulating the partial sums from
    # other work-groups covering the same tile.
    d = 18.59

    min_grid_runtime = (None, float("inf"))

    # Predict grid sizes
    for g in range(grid_start, grid_end + 1):
        runtime, iters_per_cta, fixup_peers = predicted_runtime(m, n, k, g, a, b, c, d, blk_m, blk_n, blk_k)

        if verbose:
            print(f"grid size: {g}, runtime: {runtime}, iters_per_cta: {iters_per_cta}, "
                  f"fixup_peers: {fixup_peers}, m: {m}, n: {n}, k: {k}, a: {a}, b: {b}, c: {c}, d: {d}")

        if min_grid_runtime[1] > runtime:
            min_grid_runtime = (g, runtime)

    if verbose:
        print(f"Number of Output Tiles: {number_of_output_tiles(m, n, blk_m, blk_n)}")
        print(f"Minimum runtime: {min_grid_runtime[1]} @ grid size: {min_grid_runtime[0]}")

    return min_grid_runtime[0]


def main(m: int, n: int, k: int, grid: int, num_runs: int, verbose: bool = False) -> int:
    # Block sizes
    BLK_M = 256
    BLK_N = 256
    BLK_K = 64

    # Start timing
    start_time = time.time()

    # Run the prediction for the specified number of runs
    g = 0
    for _ in range(num_runs):
        g = grid_model(m, n, k, BLK_M, BLK_N, BLK_K, 1, grid, verbose)

    # End timing
    end_time = time.time()
    elapsed_time = (end_time - start_time) / num_runs

    print(f"Best predicted grid size: {g}")
    if verbose:
        print(f"Elapsed: {elapsed_time * 1e6:.6f} microseconds")
    return g


if __name__ == "__main__":
    # Argument parser for initial command-line inputs if needed
    parser = argparse.ArgumentParser(description="Stream-K Library for GEMM")
    parser.add_argument("-m", type=int, default=3072, help="Rows of A-Matrix (default: 3072)")
    parser.add_argument("-n", type=int, default=4096, help="Columns of B-Matrix (default: 4096)")
    parser.add_argument("-k", type=int, default=4096, help="Columns of A-Matrix (default: 4096)")
    parser.add_argument(
        "-g",
        "--grid",
        type=int,
        default=304,
        help="Grid size used for Stream-K approach (default: 304)",
    )
    parser.add_argument("--num_runs", type=int, default=10, help="Number of Runs (default: 10)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Run the main function with initial command-line arguments
    main(args.m, args.n, args.k, args.grid, args.num_runs, args.verbose)
