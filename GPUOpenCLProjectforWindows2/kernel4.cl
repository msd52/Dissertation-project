

__kernel void Matrix_Multiply_Kernel_3b(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim)
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
    for (int w=0; w<WPT; w+=4) {
        acc[w] = 0.0f;
        acc[w+1] = 0.0f;
        acc[w+2] = 0.0f;
        acc[w+3] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        for (int w=0; w<WPT; w+=4) {
            Asub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*pDim];
            Asub[row][col+(w+1)*RTS] = matrixA[globalRow*pDim+(tiledCol + (w+1)*RTS)];
            Bsub[row][col+(w+1)*RTS] = matrixB[(globalCol + (w+1)*RTS) + tiledRow*pDim];
            Asub[row][col+(w+2)*RTS] = matrixA[globalRow*pDim+(tiledCol + (w+2)*RTS)];
            Bsub[row][col+(w+2)*RTS] = matrixB[(globalCol + (w+2)*RTS) + tiledRow*pDim];
            Asub[row][col+(w+3)*RTS] = matrixA[globalRow*pDim+(tiledCol + (w+3)*RTS)];
            Bsub[row][col+(w+3)*RTS] = matrixB[(globalCol + (w+3)*RTS) + tiledRow*pDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            //UNROLLED
            for (int w=0; w<WPT; w+=4) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
                acc[w+1] += Asub[row][k] * Bsub[k][col + (w+1)*RTS];
                acc[w+2] += Asub[row][k] * Bsub[k][col + (w+2)*RTS];
                acc[w+3] += Asub[row][k] * Bsub[k][col + (w+3)*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    for (int w=0; w<WPT; w+=4) {
        matrixC[(globalCol + w*RTS) + globalRow*pDim] = acc[w];
        matrixC[(globalCol + (w+1)*RTS) + globalRow*pDim] = acc[w+1];
        matrixC[(globalCol + (w+2)*RTS) + globalRow*pDim] = acc[w+2];
        matrixC[(globalCol + (w+3)*RTS) + globalRow*pDim] = acc[w+3];
    }
}

#define TS 16
#define WPT 8
#define RTS 2
#define WPTM 4
#define WPTN 4

__kernel void Multiply_Buffer_Identity(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim)
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
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            //UNROLLED
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = acc[w];
    }
    //printf("id IBuffer is %d %d, final value is %f \n \n \n ", globalRow, globalCol, acc);
}



__kernel void Multiply_Buffer_Sigmoid(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim)
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
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            //UNROLLED
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = 1.0 / (1.0 + exp(-acc[w]));
    }
}

__kernel void Multiply_Buffer_Tanh(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim)
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
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            //UNROLLED
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = tanh(acc[w]);
    }
}

__kernel void Multiply_Buffer_ReLU(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim)
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
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            //UNROLLED
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = fmax(acc[w],0);
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
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow+(tiledCol + w*RTS)*mDim];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            //UNROLLED
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
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
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow+(tiledCol + w*RTS)*mDim];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            //UNROLLED
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = clamp((float)(acc[w]* matrixD[(globalCol + w*RTS) + globalRow*nDim] * (1.0 - matrixD[(globalCol + w*RTS) + globalRow*nDim])),-0.005f,0.005f);
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
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow+(tiledCol + w*RTS)*mDim];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            //UNROLLED
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Store the final result in C
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
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow+(tiledCol + w*RTS)*mDim];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*nDim];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            //UNROLLED
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Store the final result in C
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = clamp((float)(acc[w]*(matrixD[(globalCol + w*RTS) + globalRow*nDim]>0.0? 1.0:0.0)),-0.005f,0.005f);
    }
}

__kernel void Update_Weights_Buffers(global float* matrixA, global float* matrixB, global float* matrixC
,const int mDim, const int pDim, const int nDim, const float offset)
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
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS)*pDim + tiledRow];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            //UNROLLED
            for (int w=0; w<WPT; ++w) {
                acc[w] += Asub[row][k] * Bsub[k][col + w*RTS];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Store the final result in C
    for (int w=0; w<WPT; ++w) {
        matrixC[(globalCol + w*RTS) + globalRow*nDim] = matrixC[(globalCol + w*RTS) + globalRow*nDim] - offset*acc[w]/(float)pDim;
    } 
}