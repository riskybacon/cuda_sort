#include <iostream>
#include <iterator>
#include <stdio.h>
#include "utils.h"

/**
 * Performs the reduce step of a Blelloch prefix sum scan
 *
 * Results are undefined if size is not a power of 2.
 *
 * Assumptions:
 *     size == number of threads * 2;
 *
 * @param val   The values to be scanned. Will be overwritten
 * @param size  Number of values
 * @param tid   The thread id
 */
__device__
void blelloch_reduce_sum(
    unsigned int* const val,
    const size_t size,
    const size_t tid
) {
    size_t offset = 1;
    for (size_t d = size >> 1; d > 0; d >>= 1, offset <<= 1) {
        if (tid < d) {
            const size_t s_ai = offset * (2 * tid + 1) - 1;
            const size_t s_bi = offset * (2 * tid + 2) - 1;
            val[s_bi] += val[s_ai];
        }
        __syncthreads();
    }
}

/**
 * Performs the downsweep step of a Blelloch prefix sum scan.
 *
 * Results are undefined if size is not a power of 2.
 *
 * Assumptions:
 *     size == number of threads * 2;
 *
 * @param val    The values to be scanned. Will be overwritten
 * @param size   Number of values, must be a power of 2
 * @param tid    The thread id
 */
__device__
void blelloch_downsweep_sum(
    unsigned int* const val,
    const size_t size,
    const size_t tid
) {
    size_t offset = size;
    for (size_t d = 1; d < size; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            // Index into shared memory
            const size_t s_ai = offset * (2 * tid + 1) - 1;
            const size_t s_bi = offset * (2 * tid + 2) - 1;

            const unsigned int temp = val[s_ai];
            val[s_ai] = val[s_bi];
            val[s_bi] += temp;
        }
    }
    __syncthreads();
}

/**
* Perform an exclusive scan on a single thread block.
*
* @param val  The array of values to be scanned. Will be overwritten.
* @param size The size of the val array
* @param tid  The thread tid of the kernel calling this function
 */
 __device__
 void exclusive_scan(
     unsigned int* const val,
     const size_t size,
     const size_t tid
 ) {
    blelloch_reduce_sum(val, size, tid);

    if (tid == 0) {
        val[size - 1] = 0;
    }
    __syncthreads();

    blelloch_downsweep_sum(val, size, tid);
}

/**
 * Perform an exclusive scan on a single thread block, but store the final
 * result into aux[tid].
 *
 * @param val  The array of values to be scanned. Will be overwritten.
 * @param size The size of the val array
 * @param aux  The auxilliary array for the final value. Must be at least
 *             gridDim.x * gridDim.y * gridDim.z in size
 * @param tid  The thread tid of the kernel calling this function
 */
 __device__
 void exclusive_scan(
     unsigned int* const val,
     const size_t size,
     unsigned int* const aux,
     const size_t tid
 ) {
    blelloch_reduce_sum(val, size, tid);

    if (tid == 0) {
        // This is the final value. It is overwritten by zero in a sum scan,
        // so it needs to be stored someplace for the final scan
        aux[blockIdx.x] = val[size - 1];
        val[size - 1] = 0;
    }

    blelloch_downsweep_sum(val, size, tid);
}

/**
 * Perform an exclusive sum scan. The input must already in shared memory.
 * The input is overwritten by the results. After this function call, each
 * thread block will only be able to see its portion of the results.
 *
 * @param s_val   Pointer to input values. Must be shared memory. Will be
 *                overwritten
 * @param s_size  Number of elements in shared memory block
 * @param d_aux   Pointer to global auxillary memory to store result from each
 *                thread block
 * @param s_aux   Pointer to shared auxillary memory to store results from each
 *                thread block
 * @param a_size  Size of auxillary memory pointed to by d_aux and s_aux. Must
 *                be at least gridDim.x * gridDim.y * gridDim.z and a power
 *                of 2
 * @param tid     Thread ID
 */
__device__
void exclusive_scan_shared(
    unsigned int* const s_val,
    const size_t s_size,
    unsigned int* const d_aux,
    unsigned int* const s_aux,
    const size_t a_size,
    const size_t tid
) {
    const size_t ai = tid * 2;
    const size_t bi = tid * 2 + 1;
    const size_t s_ai = tid * 2;
    const size_t s_bi = tid * 2 + 1;

    // Perform exclusive scan on this thread block. The final step of the
    // reduction will be stored in the global memory referenced by d_aux at
    // d_aux[blockIdx.x]
    exclusive_scan(s_val, s_size, d_aux, tid);

    // Copy the auxillary elements into shared memory
    if (ai < a_size) {
        s_aux[s_ai] = d_aux[ai];
    }

    if (bi < a_size) {
        s_aux[s_bi] = d_aux[bi];
    }

    __syncthreads();

    // Perform an exclusive scan on the auxillary elements
    exclusive_scan(s_aux, a_size, tid);

    // Add the final value to each entry in this block
    unsigned int val = s_aux[blockIdx.x];

    s_val[s_ai] += val;
    s_val[s_bi] += val;
}

/**
 * Perform an exclusive sum scan. Source values are stored in d_val and are
 * overwritten during the scan.
 *
 * @param d_val   Pointer to source data. Will be overwritten.
 * @param d_size  Total size of array to scan
 * @param s_val   Pointer to shared memory.
 * @param s_size  Size of shared memory block. Must be
 *                blockDim.x * blockDim.y * blockDim.z * 2 in size
 * @param d_aux   Pointer to global auxillary memory to store result from each
 *                thread block
 * @param s_aux   Pointer to shared auxillary memory to store results from each
 *                thread block
 * @param a_size  Size of auxillary memory pointed to by d_aux and s_aux. Must
 *                be at least gridDim.x * gridDim.y * gridDim.z and a power
 *                of 2
 * @param tid     Thread ID
 * @param gid     Global ID of thread
 */
__device__
void exclusive_scan(
    unsigned int* const d_val,
    const size_t d_size,
    unsigned int* const s_val,
    const size_t s_size,
    unsigned int* const d_aux,
    unsigned int* const s_aux,
    const size_t a_size,
    const size_t tid,
    const size_t gid
) {
    const size_t ai = gid * 2;
    const size_t bi = gid * 2 + 1;
    const size_t s_ai = tid * 2;
    const size_t s_bi = tid * 2 + 1;

    // Copy values from global memory into shared memory. The initial and final
    // step of the scan handle 2 * num threads elements
    s_val[s_ai] = ai < d_size ? d_val[ai] : 0;
    s_val[s_bi] = bi < d_size ? d_val[bi] : 0;

    // Initialize auxillary memory to all zeros.
    //
    // Why bother? Because the number of blocks might be less than a_size.
    // The final scan is performed on d_aux, which might not be totally filled
    // because num_blocks < a_size. But the scan needs to be performed on an
    // array that has a size that is a power of 2. This operation pads the aux
    // array with zero so that the scan is correct.
    if (gid < a_size) {
        d_aux[gid] = 0;
    }

    __syncthreads();

    exclusive_scan_shared(s_val, s_size, d_aux, s_aux, a_size, tid);

    // Write the final values to memory
    if (ai < d_size) {
        d_val[ai] = s_val[s_ai];
    }

    if (bi < d_size) {
        d_val[bi] = s_val[s_bi];
    }
    __syncthreads();
}

/**
 * Build an output list of zeros and ones where the input array has the bit
 * set specified by the predicate
 *
 * @param blocks        (kernel launch) must satisfy
 *                      blocks >= d_size / (threads per block * 2)
 * @param threads       (kernel launch param) number of threads per block
 * @param shared        (kernel launch param) unused
 * @param d_predicate   The result, 1 or 0, of applying the predicate to the
 *                      d_input array
 * @param d_in          The input array
 * @param d_size        Number of elements in the array
 * @param bit           The bit to check
 * @param check_on      If 1, check that the bit is on, otherwise, check if
 *                      bit is off
 */
__global__
void map_kernel(
    unsigned int* const d_predicate,
    const unsigned int* const d_in,
    const size_t d_size,
    const unsigned int bit,
    const unsigned int check_on
) {
    const size_t tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + tid;
    const size_t ai = gid * 2;
    const size_t bi = gid * 2 + 1;
    const unsigned int bit_value = 1 << bit;

    if (ai < d_size) {
        unsigned int ai_predicate = (d_in[ai] & bit_value) == 0;
        d_predicate[ai] = check_on ? !ai_predicate : ai_predicate;
    }

    if (bi < d_size) {
        unsigned int bi_predicate = (d_in[bi] & bit_value) == 0;
        d_predicate[bi] = check_on ? !bi_predicate : bi_predicate;
    }
}

/**
 * Perform a non-destructive exclusive prefix sum scan
 *
 * Each thread is responsible for two elements.
 *
 * @param blocks  (kernel launch)  must satisfy blocks * threads * 2 >= d_size
 * @param threads (kernel launch)  must be a power of 2
 * @param shared  (kernel launch)  must be threads * 2 + a_size
 * @param d_out   The output of the scan
 * @param d_in    Input to the scan
 * @param d_size  The size of d_scan and d_in
 * @param s_size  The size of shared memory, must be threads * 2
 * @param d_aux   Global memory for storing 2nd part of the scan
 * @param a_size  Size of d_aux
 */
__global__
void prefix_sum_scan_kernel(
    unsigned int* const d_out,
    const unsigned int* const d_in,
    const size_t d_size,
    const size_t s_size,
    unsigned int* const d_aux,
    const size_t a_size
) {
    const size_t tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + tid;

    const size_t ai = gid * 2;
    const size_t bi = gid * 2 + 1;

    extern __shared__ unsigned int s_buffer[];

    unsigned int* const s_val = s_buffer;
    unsigned int* const s_aux = &s_val[s_size];

    if (ai < d_size) {
        d_out[ai] = d_in[ai];
    }

    if (bi < d_size) {
        d_out[bi] = d_in[bi];
    }

    exclusive_scan(d_out, d_size, s_val, s_size, d_aux, s_aux, a_size, tid,
                   gid);
}

/**
 * Compacts elements in d_input_val and d_input_pos into d_output_val and
 * d_output_pos based on d_predicate and d_address. If d_predicate[idx] == 1,
 * then the value in d_input_pos and d_input_val is placed into the address at
 * d_address[idx] + offset.
 *
 * @param (kernel launch) blocks   must satisfy blocks >= d_size / (threads * 2)
 * @param (kernel launch) threads  number of threads per block
 * @param (kernel launch) shared   unused
 * @param d_output_val  output values
 * @param d_output_pos  output positions
 * @param d_input_val   input values
 * @param d_input_pos   input positions
 * @param d_predicate   Array of 1s and 0s where d_input_val matches a filter
 * @param d_address     Destination addresses
 * @param d_size        The size of input and output arrays
 * @param offset        The starting offset for the scatter / compact
 */
__global__
void compact_kernel(
    unsigned int* const d_output_val,
    unsigned int* const d_output_pos,
    const unsigned int* const d_input_val,
    const unsigned int* const d_input_pos,
    const unsigned int* const d_predicate,
    const unsigned int* const d_address,
    const size_t d_size,
    const unsigned int offset
) {
    const size_t tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + tid;
    const size_t ai = gid * 2;
    const size_t bi = gid * 2 + 1;
    size_t ai_address = d_size + 1;
    size_t bi_address = d_size + 1;
    unsigned int ai_predicate = 0;
    unsigned int bi_predicate = 0;

    if (ai < d_size) {
        ai_address = d_address[ai] + offset;
        ai_predicate = d_predicate[ai];
    }

    if (bi < d_size) {
        bi_address = d_address[bi] + offset;
        bi_predicate = d_predicate[bi];
    }

    if (ai_address < d_size && ai_predicate) {
        d_output_val[ai_address] = d_input_val[ai];
        d_output_pos[ai_address] = d_input_pos[ai];
    }

    if (bi_address < d_size && bi_predicate) {
        d_output_val[bi_address] = d_input_val[bi];
        d_output_pos[bi_address] = d_input_pos[bi];
    }
}

/**
 * Build an output list of zeros and ones where the input array has the bit
 * set specified by the predicate
 *
 * @param d_out         The result, 1 or 0, of applying the predicate to the
 *                      d_input array
 * @param d_in          The input array
 * @param d_size        Number of elements in the array
 * @param bit           The bit to check
 * @param check_on      If 1, check that the bit is on, otherwise, check if
 *                      bit is off
 * @param threads_per_block  Number of threads in a thread block. Each thread
 *                  operates on two elements
 */
void map(
    unsigned int * const d_out,
    const unsigned int * const d_in,
    const size_t d_size,
    unsigned int bit,
    unsigned int check_on,
    const size_t threads_per_block
) {
    const size_t blocks = std::ceil(float(d_size) / (threads_per_block * 2));

    map_kernel<<<blocks, threads_per_block>>>(
        d_out, d_in, d_size, bit, check_on
    );
}

/**
 * Perform a non-destructive exclusive prefix sum scan
 *
 * @param d_out   The output of the scan
 * @param d_in    Input to the scan
 * @param d_size  The size of d_scan and d_in
 * @param threads_per_block  Number of threads in a thread block. Each thread
 *                  operates on two elements
 */
void prefix_sum_scan(
    unsigned int * const d_out,
    const unsigned int * const d_in,
    const size_t d_size,
    const size_t threads_per_block
) {
    using value_type = unsigned int;
    const size_t s_size = threads_per_block * 2;
    const size_t blocks = std::ceil(float(d_size) / s_size);

    // Allocate memory for aux storage for 2nd level of scan
    unsigned int* d_aux;
    const size_t a_size = next_pow2(blocks);
    const size_t a_mem_size = a_size * sizeof(value_type);
    checkCudaErrors(cudaMalloc(&d_aux, a_mem_size));

    if (a_size > s_size) {
        std::cerr << "The size of auxillary memory (" << a_size
                  << ") is larger than the number of threads * 2 per block ("
                  << s_size << "). The results of the scan will be incorrect "
                  << "after element " << threads_per_block * blocks
                  << std::endl;
        exit(1);
    }

    // Need shared memory for scan values and auxillary scan
    size_t shared = s_size * sizeof(value_type) + a_mem_size;

    prefix_sum_scan_kernel<<<blocks, threads_per_block, shared>>>(
        d_out, d_in, d_size, s_size, d_aux, a_size
    );

    checkCudaErrors(cudaFree(d_aux));
}

/**
 * Compacts elements in d_input_val and d_input_pos into d_output_val and
 * d_output_pos based on d_predicate and d_address. If d_predicate[idx] == 1,
 * then the value in d_input_pos and d_input_val is placed into the address at
 * d_address[idx] + offset.
 *
 * @param d_output_val  output values
 * @param d_output_pos  output positions
 * @param d_input_val   input values
 * @param d_input_pos   input positions
 * @param d_predicate   Array of 1s and 0s where d_input_val matches a filter
 * @param d_address     Destination addresses
 * @param d_size        The size of input and output arrays
 * @param offset        The starting offset for the scatter / compact
 * @param threads_per_block  Number of threads in a thread block. Each thread
 *                  operates on two elements
 */
void compact(
    unsigned int* const d_output_val,
    unsigned int* const d_output_pos,
    const unsigned int* const d_input_val,
    const unsigned int* const d_input_pos,
    const unsigned int* const d_predicate,
    const unsigned int* const d_address,
    const size_t d_size,
    const unsigned int offset,
    const size_t threads_per_block
) {
    const size_t blocks = std::ceil(float(d_size) / (threads_per_block * 2));

    compact_kernel<<<blocks, threads_per_block>>>(
        d_output_val, d_output_pos, d_input_val, d_input_pos, d_predicate,
        d_address, d_size, offset
    );
}

/**
 * Performs the map, scan, and compact portion of radix sort.
 * Map: set d_predicate[idx] to 1 if d_input_pos[idx] is bit is on and
 *      check_on == 1. Set to 1 if check_on == 0 and bit is off
 * Scan: perform a prefix sum scan on the map results
 * Compact: Scatter input values and positions into the output arrays using
 *      the results from scan[idx] and if d_predicate[idx] == 1
 *
 * @param d_output_val  output values
 * @param d_output_pos  output positions
 * @param d_input_val   input values
 * @param d_input_pos   input positions
 * @param d_predicate   Array of 1s and 0s where d_input_val matches a filter
 * @param d_address     Destination addresses
 * @param d_size        The size of input and output arrays
 * @param offset        The starting offset for the scatter / compact
 * @param bit           The bit to check
 * @param check_on      If 1, check that the bit is on, otherwise, check if
 *                      bit is off
 * @param threads_per_block  Number of threads in a thread block. Each thread
 *                  operates on two elements
 */
unsigned int map_scan_compact(
    unsigned int* const d_output_val,
    unsigned int* const d_output_pos,
    const unsigned int* const d_input_val,
    const unsigned int* const d_input_pos,
    unsigned int* const d_predicate,
    unsigned int* const d_address,
    const size_t d_size,
    const unsigned int offset,
    const unsigned int bit,
    const unsigned int check_on,
    const size_t threads_per_block
) {
    using value_type = unsigned int;

    map(d_predicate, d_input_val, d_size, bit, check_on, threads_per_block);
    prefix_sum_scan(d_address, d_predicate, d_size, threads_per_block);
    compact(d_output_val, d_output_pos, d_input_val, d_input_pos,
        d_predicate, d_address, d_size, offset, threads_per_block);

    unsigned int new_offset;
    checkCudaErrors(cudaMemcpy(&new_offset, &d_address[d_size - 1],
        sizeof(value_type), cudaMemcpyDefault));
    unsigned int last_predicate;
    checkCudaErrors(cudaMemcpy(&last_predicate, &d_predicate[d_size - 1],
        sizeof(value_type), cudaMemcpyDefault));

    return new_offset + last_predicate;
}

void print_array(
    const char* prefix,
    const unsigned int * const start,
    const size_t size,
    std::ostream& out = std::cout
) {
    std::ostream_iterator<unsigned int> out_it(out, ", ");
    out << prefix;
    std::copy(start, start + size, out_it);
    out << std::endl;
}

/**
 * Display debug information for each step of the map_scan_reduce call
 */
void print_debug(
    unsigned int* const d_output_val,
    unsigned int* const d_output_pos,
    const unsigned int* const d_input_val,
    const unsigned int* const d_input_pos,
    unsigned int* const d_predicate,
    unsigned int* const d_address,
    const size_t d_size,
    const unsigned int offset,
    const unsigned int bit,
    const unsigned int check_on,
    const size_t threads_per_block
) {
    return;
    using value_type = unsigned int;

    value_type* const h_output_val = new value_type[d_size];
    value_type* const h_output_pos = new value_type[d_size];
    value_type* const h_input_val = new value_type[d_size];
    value_type* const h_input_pos = new value_type[d_size];
    value_type* const h_predicate = new value_type[d_size];
    value_type* const h_address = new value_type[d_size];

    const size_t mem_size = sizeof(value_type) * d_size;

    std::ostream_iterator<unsigned int> out_it(std::cout, ", ");

    checkCudaErrors(cudaMemcpy(h_output_val, d_output_val, mem_size,
        cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(h_output_pos, d_output_pos, mem_size,
        cudaMemcpyDefault));

    checkCudaErrors(cudaMemcpy(h_input_val, d_input_val, mem_size,
        cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(h_input_pos, d_input_pos, mem_size,
        cudaMemcpyDefault));

    checkCudaErrors(cudaMemcpy(h_predicate, d_predicate, mem_size,
        cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(h_address, d_address, mem_size,
        cudaMemcpyDefault));

    std::cout << "------------------------------" << std::endl;
    std::cout << "bit: " << bit << " check_on: " << check_on << " offset: "
              << offset << std::endl;
    print_array("in:    ", h_input_val, d_size);
    print_array("pred:  ", h_predicate, d_size);
    print_array("addr:  ", h_address, d_size);
    print_array("out:   ", h_output_val, d_size);
    print_array("pos:   ", h_output_pos, d_size);
    std::cout << std::endl;

    // Clean up
    delete[] h_output_val;
    delete[] h_output_pos;
    delete[] h_input_val;
    delete[] h_input_pos;
    delete[] h_predicate;
    delete[] h_address;
}

/**
 * Sort entry. Input values and positions will be reordered.
 *
 * @param d_input_val   input values
 * @param d_input_pos   input positions
 * @param d_output_val  output values
 * @param d_output_pos  output positions
 * @param d_size        The size of input and output arrays
 * @param num_bits      number of bits in The starting offset for the scatter / compact
 * @param bit           The bit to check
 * @param check_on      If 1, check that the bit is on, otherwise, check if
 *                      bit is off
 * @param threads_per_block  Number of threads in a thread block. Each thread
 *                  operates on two elements

 */
void radix_sort(
    unsigned int* const d_input_val,
    unsigned int* const d_input_pos,
    unsigned int* const d_output_val,
    unsigned int* const d_output_pos,
    const size_t d_size,
    const size_t num_bits,
    const size_t threads_per_block
) {
    unsigned int* d_val[2] = {d_input_val, d_output_val};
    unsigned int* d_pos[2] = {d_input_pos, d_output_pos};

    // Need some auxillary memory for the scan
    // Allocate memory for scan output and predicate output
    unsigned int* d_address;
    unsigned int* d_predicate;
    size_t mem_size = d_size * sizeof(unsigned int);
    checkCudaErrors(cudaMalloc(&d_address, mem_size));
    checkCudaErrors(cudaMalloc(&d_predicate, mem_size));
    checkCudaErrors(cudaMemset(d_address, 0, mem_size));
    checkCudaErrors(cudaMemset(d_predicate, 0, mem_size));

    size_t input = 1;
    size_t output = 0;

    const unsigned int check_on = 1;
    const unsigned int check_off = 0;

    for (unsigned int bit = 0; bit < num_bits; ++bit) {
        input = !input;
        output = !output;

        unsigned int offset = 0;

        offset = map_scan_compact(d_val[output], d_pos[output], d_val[input],
            d_pos[input], d_predicate, d_address, d_size, offset, bit,
            check_off, threads_per_block);

        print_debug(d_val[output], d_pos[output], d_val[input], d_pos[input],
            d_predicate, d_address, d_size, offset, bit, check_off,
            threads_per_block);

        map_scan_compact(d_val[output], d_pos[output], d_val[input],
            d_pos[input], d_predicate, d_address, d_size, offset, bit,
            check_on, threads_per_block);

        print_debug(d_val[output], d_pos[output], d_val[input], d_pos[input],
            d_predicate, d_address, d_size, offset, bit, check_on,
            threads_per_block);
    }

    if (output != 1) {
        checkCudaErrors(cudaMemcpy(d_output_val, d_input_val, mem_size,
            cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(d_output_pos, d_input_pos, mem_size,
            cudaMemcpyDefault));
    }

    // Clean up
    checkCudaErrors(cudaFree(d_address));
    checkCudaErrors(cudaFree(d_predicate));
}

/**
 * Set up the sort, call radix_sort.
 */
void sort_entry(
    const size_t d_size,
    const size_t num_bits,
    const size_t threads_per_block
) {
    // How much memory to allocate?
    // Need 4 buffers: input pos and val, output pos and val.
    const size_t mem_elements = 4 * d_size;
    const size_t mem_size = mem_elements * sizeof(unsigned int);

    // Allocate memory on the device in one big chunk
    unsigned int* d_mem;
    checkCudaErrors(cudaMalloc(&d_mem, mem_size));

    // Allocate memory on the host in one big chunk
    unsigned int* const h_mem = new unsigned int[mem_elements];
    memset(h_mem, 0, mem_size);

    // Partition memory on device
    unsigned int* const d_input_val = d_mem;
    unsigned int* const d_input_pos = &d_input_val[d_size];
    unsigned int* const d_output_val = &d_input_pos[d_size];
    unsigned int* const d_output_pos = &d_output_val[d_size];

    // Partition memory on host
    unsigned int* const h_input_val = h_mem;
    unsigned int* const h_input_pos = &h_input_val[d_size];
    unsigned int* const h_output_val = &h_input_pos[d_size];
    unsigned int* const h_output_pos = &h_output_val[d_size];

    // Allocate space for reference values
    unsigned int* const h_ref_val = new unsigned int[d_size];
    unsigned int* const h_ref_pos = new unsigned int[d_size];

    // Initialize host and reference values
    for (int idx = 0; idx < d_size; ++idx) {
        h_input_val[idx] = d_size - idx;
        h_input_pos[idx] = idx;
        h_ref_val[idx] = 1 + idx;
        h_ref_pos[idx] = d_size - idx - 1;
    }

    // Copy host values to device
    checkCudaErrors(cudaMemcpy(d_mem, h_mem, mem_size, cudaMemcpyDefault));

    // Run the sort
    float time;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));
    radix_sort(d_input_val, d_input_pos, d_output_val, d_output_pos, d_size,
               num_bits, threads_per_block);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
    printf("Time to sort:  %3.1f ms \n", time);

    // Copy values from device to host
    checkCudaErrors(cudaMemcpy(h_mem, d_mem, mem_size, cudaMemcpyDefault));

    // Make sure that the sort worked correctly
    check_results(h_ref_val, h_output_val, d_size);
    check_results(h_ref_pos, h_output_pos, d_size);

    // Clean up
    checkCudaErrors(cudaFree(d_mem));
    delete[] h_mem;
    delete[] h_ref_val;
    delete[] h_ref_pos;
}

std::ostream& operator<<(std::ostream& out, const cudaDeviceProp& prop) {
    out << "Device name: " << prop.name << std::endl
        << "  PCI Address: " << prop.pciBusID << ":" << prop.pciDeviceID << ":"
        << prop.pciDomainID << std::endl
        << "  Compute capability: "
        << prop.major << "." << prop.minor << std::endl
        << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl
        << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl
        << "  Peak Memory Bandwidth (GB/s): "
        << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6
        << std::endl
        << "  Number of multiprocessors: "
        << prop.multiProcessorCount
        << std::endl
        << "  Max threads per block: " << prop.maxThreadsPerBlock
        << std::endl
        << "  Max threads per multiprocessor: "
        << prop.maxThreadsPerMultiProcessor
        << std::endl
        << "  Shared memory per block: "
        << prop.sharedMemPerBlock / 1024 << " KB"
        << std::endl
        << "  Shared memory per multiprocessor: "
        << prop.sharedMemPerMultiprocessor / 1024 << " KB"
        << std::endl
        << "  Total global memory: "
        << prop.totalGlobalMem / 1024 / 1024 / 1024 << " GB";

    return out;
}

void device_info() {
    int num_devices;

    cudaGetDeviceCount(&num_devices);
    for (int i = 0; i < num_devices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << prop << std::endl << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [size] [num_bits = 32] "
                  << "[threads_per_block = 1024]" << std::endl;
        return EXIT_FAILURE;
    }

    device_info();
    size_t d_size = atoi(argv[1]);
    size_t threads_per_block = 1024;
    size_t num_bits = 32;

    if (argc >= 3) {
        num_bits = atoi(argv[2]);
    }

    if (argc >= 4) {
        threads_per_block = atoi(argv[3]);
    }

    if (d_size < threads_per_block) {
        threads_per_block = next_pow2(d_size);

    }

    sort_entry(d_size, num_bits, threads_per_block);

    return EXIT_SUCCESS;
}
