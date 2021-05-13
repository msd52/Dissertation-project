// TODO: Add OpenCL kernel code here.

#define TS 16
#define WPT 16
#define RTS 1
#define UNROLLFACTOR 8

__kernel void Multiply_Buffer_Identity(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases)
{
 // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
    __local float Biassub[TS];

    float acc[WPT];
    //__attribute__((opencl_unroll_hint(WPT)))
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;

    /*if (col==0){
        Biassub[row] = biases[globalCol];
    }*/
    //__attribute__((opencl_unroll_hint(4)))
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        /*for (int w=0; w<WPT; w++) {
            //prefetch(&(matrixA[globalRow*pDim+(tiledCol + w+1)]), 1);
            //prefetch(&(matrixB[(globalCol + w+1) + tiledRow*nDim]), 1);
            Asub[row][col+w] = matrixA[globalRow*pDim+(tiledCol + w)];
            Bsub[row][col+w] = matrixB[(globalCol + w) + tiledRow*nDim];
        }*/
        //__attribute__((opencl_unroll_hint(WPT)))
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
            //Csub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            //Dsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];

        }
        
        /*for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            Bsub[row+w*RTS][col] = matrixB[globalCol + (tiledRow+ w*RTS)*nDim];
        }*/

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        //__attribute__((opencl_unroll_hint(TS)))
        for (int k=0; k<TS; k++) {
            /*for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w];
            }*/
            //__attribute__((opencl_unroll_hint(WPT)))
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    //__attribute__((opencl_unroll_hint(WPT)))
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = acc[w]+biases[globalRow];
    }
}



__kernel void Multiply_Buffer_Sigmoid(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases)
{
 // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation registers
    float acc[WPT];
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = 1.0 / (1.0 + exp(-acc[w]-biases[globalRow]));
    }
}

__kernel void Multiply_Buffer_Tanh(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases)
{
 // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation registers
    float acc[WPT];
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = tanh(acc[w]+biases[globalRow]);
    }
}

__kernel void Multiply_Buffer_ReLU(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases)
{
 // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation registers
    float acc[WPT];
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = fmax(acc[w]+biases[globalRow],0);
    }
}

__kernel void Multiply_Deltas_Buffers_Identity(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
 // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation registers
    float acc[WPT];
    //__attribute__((opencl_unroll_hint(1)))
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        //__attribute__((opencl_unroll_hint(1)))
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow+(tiledCol + w*RTS)*mDim];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        //__attribute__((opencl_unroll_hint(1)))
        for (int k=0; k<TS; k++) {
            //__attribute__((opencl_unroll_hint(1)))
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    //__attribute__((opencl_unroll_hint(1)))
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = clamp((float)acc[w],-0.005f,0.005f);
    }
}


__kernel void Multiply_Deltas_Buffers_Sigmoid(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
 // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation registers
    float acc[WPT];
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow+(tiledCol + w*RTS)*mDim];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = clamp((float)(acc[w]* matrixD[(globalCol + w*RTS) + globalRow*nDim] * (1.0f - matrixD[(globalCol + w*RTS) + globalRow*nDim])),-0.005f,0.005f);
    }
}

__kernel void Multiply_Deltas_Buffers_Tanh(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
  // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation registers
    float acc[WPT];
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow+(tiledCol + w*RTS)*mDim];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Store the final result in C
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = clamp((float)(acc[w]*(1 - pow(matrixD[(globalCol + w*RTS) + globalRow*nDim],2))),-0.005f,0.005f);
    }
}

__kernel void Multiply_Deltas_Buffers_ReLU(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
// Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation registers
    float acc[WPT];
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow+(tiledCol + w*RTS)*mDim];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Store the final result in C
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = clamp((float)(acc[w]*(matrixD[(globalCol + w*RTS) + globalRow*nDim]>0.0? 1.0:0.0)),-0.005f,0.005f);
    }
}

__kernel void Update_Weights_Buffers(global float* matrixA, global float* matrixB, global float* matrixC
,const int mDim, const int pDim, const int nDim, global float* biases, const float offset)
{
// Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation registers
    float acc[WPT];
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS)*pDim + tiledRow];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Store the final result in C
    __attribute__((opencl_unroll_hint(UNROLLFACTOR)))
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = matrixC[(globalCol + w*RTS) + globalRow*nDim] - offset*acc[w]/(float)pDim;
    }
    if (globalCol==0){
        float tempBias=0.0f;
        for (int p = 0 ; p < pDim; p++){
            tempBias+= matrixA[globalRow*pDim+p];
        }
        biases[globalRow] = biases[globalRow] - offset*tempBias / (float)pDim;
    }
} 