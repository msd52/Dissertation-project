//File containing implementation with basic cache blocking using local memory

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
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)


    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[row][col] = matrixA[tiledCol + globalRow*pDim];
        Bsub[row][col] = matrixB[globalCol+ tiledRow*nDim];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    matrixC[globalCol + globalRow*nDim] = acc;
    //printf("id IBuffer is %d %d, final value is %f \n \n \n ", globalRow, globalCol, acc);
}



__kernel void Multiply_Buffer_Sigmoid(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim)
{
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[row][col] = matrixA[tiledCol + globalRow*pDim];
        Bsub[row][col] = matrixB[globalCol+ tiledRow*nDim];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    matrixC[globalCol + globalRow*nDim] = 1.0 / (1.0 + exp(-acc));
}

__kernel void Multiply_Buffer_Tanh(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim)
{
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[row][col] = matrixA[tiledCol + globalRow*pDim];
        Bsub[row][col] = matrixB[globalCol+ tiledRow*nDim];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    matrixC[globalCol + globalRow*nDim] = tanh(acc);
}

__kernel void Multiply_Buffer_ReLU(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim)
{
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[row][col] = matrixA[tiledCol + globalRow*pDim];
        Bsub[row][col] = matrixB[globalCol+ tiledRow*nDim];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    matrixC[globalCol + globalRow*nDim] = fmax(acc,0);
}

__kernel void Multiply_Deltas_Buffers_Identity(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)


    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;

        Asub[row][col] = matrixA[tiledCol*mDim + globalRow];
        Bsub[row][col] = matrixB[globalCol+ tiledRow*nDim];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    matrixC[globalCol + globalRow*nDim] = acc;
}


__kernel void Multiply_Deltas_Buffers_Sigmoid(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[row][col] = matrixA[tiledCol*mDim + globalRow];
        Bsub[row][col] = matrixB[globalCol+ tiledRow*nDim];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    matrixC[globalCol + globalRow*nDim] = acc* matrixD[globalCol + globalRow*nDim] * (1.0 - matrixD[globalCol + globalRow*nDim]);
}

__kernel void Multiply_Deltas_Buffers_Tanh(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[row][col] = matrixA[tiledCol*mDim + globalRow];
        Bsub[row][col] = matrixB[globalCol+ tiledRow*nDim];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    matrixC[globalCol + globalRow*nDim] = acc*  (1 - pow(matrixD[globalCol + globalRow*nDim],2));
}

__kernel void Multiply_Deltas_Buffers_ReLU(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[row][col] = matrixA[tiledCol*mDim + globalRow];
        Bsub[row][col] = matrixB[globalCol+ tiledRow*nDim];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            //if (globalRow %100 ==0 && globalCol %100 ==0){
            //    printf("Ins are %f %f", Asub[row][k] ,Bsub[k][col]);
            //}
            acc += Asub[row][k] * Bsub[k][col];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    matrixC[globalCol + globalRow*nDim] = acc* (matrixD[globalCol + globalRow*nDim] > 0.0? 1.0:0.0);
}

__kernel void Update_Weights_Buffers(global float* matrixA, global float* matrixB, global float* matrixC
,const int mDim, const int pDim, const int nDim, const float offset)
{
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[row][col] = matrixA[tiledCol + globalRow*pDim];
        Bsub[row][col] = matrixB[globalCol*pDim+ tiledRow];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }
        acc = acc/(float)pDim;
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    matrixC[globalCol + globalRow*nDim] = matrixC[globalCol + globalRow*nDim] - clamp(offset*acc/(float)pDim,-0.02f,0.02f);
}