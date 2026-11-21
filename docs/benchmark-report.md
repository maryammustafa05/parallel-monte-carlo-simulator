# Benchmark report

> Real numbers are added in Phase 6. This section records the test machine now.

## Test machine

| Item | Value |
|---|---|
| Host CPU | Intel Core i7-10610U @ 1.80 GHz (4 physical cores / 8 threads on the host) |
| VM | VirtualBox, Ubuntu 24.04 LTS |
| Cores given to the VM (`nproc`) | **5** (1 thread per core, as seen by the VM) |
| RAM given to the VM | 10 GiB |
| Compiler / MPI | g++ 13.3.0, Open MPI 4.1.6 |

**What this means for speedup:** the ceiling is the 5 cores the VM sees, not the host's 8.
The host CPU has only 4 *physical* cores, so VirtualBox's 5 virtual cores are partly backed
by hyper-threads. Expect speedup to start flattening around 4 workers.
