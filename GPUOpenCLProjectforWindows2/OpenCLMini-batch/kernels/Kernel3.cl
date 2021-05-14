#define WGS 16 //optimized value, used for kernel 3b
#define TW 16//optimized value, used for kernel 3b
#define STRIDE 1
#define WGS2 WGS
#define TW2 TW
#define DEBUG_FORWARD false
#define KERNEL3C false //kernels 3b but with loop unrolling

__kernel void Multiply_Buffer_Identity(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = WGS*get_group_id(0)+row;
    const int globalCol = WGS*get_group_id(1)+col;
 
    float temp[TW];

    local float Biassub[WGS];
    local float tempA[WGS][WGS];
    local float tempB[WGS][WGS];

    #if KERNEL3C == true
    __attribute__((opencl_unroll_hint(TW)))
    #endif
    for (int w=0; w<TW; w++) {
        temp[w] = 0.0f;
    }
    
    const int numTiles = pDim/WGS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = WGS*t + row;
        const int tiledCol = WGS*t + col;

        #if KERNEL3C == true
            __attribute__((opencl_unroll_hint(TW2)))
        #endif
        for (int w=0; w<TW; w++) {
            tempA[row][col+w*STRIDE] = matrixA[globalRow*pDim+(tiledCol + w*STRIDE)];
            tempB[row][col+w*STRIDE] = matrixB[(globalCol + w*STRIDE) + tiledRow*nDim];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
 
        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(WGS2)))
        #endif        
        for (int k=0; k<WGS; k++) {

            #if KERNEL3C == true
            __attribute__((opencl_unroll_hint(TW2)))
            #endif
            for (int w=0; w<TW; ++w) {
                temp[w] += tempA[row][k] * tempB[k][col + w*STRIDE];
            }
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    #if KERNEL3C == true
         __attribute__((opencl_unroll_hint(TW2)))
    #endif   
    for (int w=0; w<TW; ++w) {
        matrixC[(globalCol + w*STRIDE) + globalRow*nDim] = temp[w]+biases[globalRow];
    }
    #if DEBUG_FORWARD == true
        if (row==0 && col==0){
            printf("in mul id kernel 1\n");
        }
        printf("Identity forward is %d %d, is %f \n", globalRow, globalCol, matrixC[(globalCol + STRIDE) + globalRow*nDim]);
    #endif
}



__kernel void Multiply_Buffer_Sigmoid(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = WGS*get_group_id(0)+row;
    const int globalCol = WGS*get_group_id(1)+col;
 
    local float tempA[WGS][WGS];
    local float tempB[WGS][WGS];
 
    float temp[TW];

    #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; w++) {
        temp[w] = 0.0f;
    }
    
    const int numTiles = pDim/WGS;
    for (int t=0; t<numTiles; t++) {
 
        const int tiledRow = WGS*t + row;
        const int tiledCol = WGS*t + col;

        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
        #endif
        for (int w=0; w<TW; w++) {
            tempA[row][col+w*STRIDE] = matrixA[globalRow*pDim+(tiledCol + w*STRIDE)];
            tempB[row][col+w*STRIDE] = matrixB[(globalCol + w*STRIDE) + tiledRow*nDim];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
 
        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(WGS2)))
        #endif
        for (int k=0; k<WGS; k++) {
            #if KERNEL3C == true
            __attribute__((opencl_unroll_hint(TW2)))
            #endif
            for (int w=0; w<TW; ++w) {
                temp[w] += tempA[row][k] * tempB[k][col + w*STRIDE];
            }
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; ++w) {
        matrixC[(globalCol + w*STRIDE) + globalRow*nDim] = 1.0 / (1.0 + exp(-temp[w]-biases[globalRow]));
    }
}

__kernel void Multiply_Buffer_Tanh(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = WGS*get_group_id(0)+row;
    const int globalCol = WGS*get_group_id(1)+col;
 
    local float tempA[WGS][WGS];
    local float tempB[WGS][WGS];
 
    float temp[TW];

    #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; w++) {
        temp[w] = 0.0f;
    }
    
    const int numTiles = pDim/WGS;


    for (int t=0; t<numTiles; t++) {
 
        const int tiledRow = WGS*t + row;
        const int tiledCol = WGS*t + col;

        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
        #endif
        for (int w=0; w<TW; w++) {
            tempA[row][col+w*STRIDE] = matrixA[globalRow*pDim+(tiledCol + w*STRIDE)];
            tempB[row][col+w*STRIDE] = matrixB[(globalCol + w*STRIDE) + tiledRow*nDim];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(WGS2)))
        #endif
        for (int k=0; k<WGS; k++) {
            #if KERNEL3C == true
            __attribute__((opencl_unroll_hint(TW2)))
            #endif
            for (int w=0; w<TW; ++w) {
                temp[w] += tempA[row][k] * tempB[k][col + w*STRIDE];
            }
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; ++w) {
        matrixC[(globalCol + w*STRIDE) + globalRow*nDim] = tanh(temp[w]+biases[globalRow]);
    }
}

__kernel void Multiply_Buffer_ReLU(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1); 
    const int globalRow = WGS*get_group_id(0)+row;
    const int globalCol = WGS*get_group_id(1)+col;
 
    local float tempA[WGS][WGS];
    local float tempB[WGS][WGS];
 
    float temp[TW];

     #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; w++) {
        temp[w] = 0.0f;
    }
    
    const int numTiles = pDim/WGS;
    for (int t=0; t<numTiles; t++) {
 
        const int tiledRow = WGS*t + row;
        const int tiledCol = WGS*t + col;

        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
        #endif
        for (int w=0; w<TW; w++) {
            tempA[row][col+w*STRIDE] = matrixA[globalRow*pDim+(tiledCol + w*STRIDE)];
            tempB[row][col+w*STRIDE] = matrixB[(globalCol + w*STRIDE) + tiledRow*nDim];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
 
        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(WGS2)))
        #endif
        for (int k=0; k<WGS; k++) {
            #if KERNEL3C == true
            __attribute__((opencl_unroll_hint(TW2)))
            #endif
            for (int w=0; w<TW; ++w) {
                temp[w] += tempA[row][k] * tempB[k][col + w*STRIDE];
            }
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; ++w) {
        matrixC[(globalCol + w*STRIDE) + globalRow*nDim] = fmax(temp[w]+biases[globalRow],0);
    }
}

__kernel void Multiply_Deltas_Buffers_Identity(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
    const int row = get_local_id(0); 
    const int col = get_local_id(1);
    const int globalRow = WGS*get_group_id(0)+row;
    const int globalCol = WGS*get_group_id(1)+col; 
 
    local float tempA[WGS][WGS];
    local float tempB[WGS][WGS];
 
    float temp[TW];
    #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; w++) {
        temp[w] = 0.0f;
    }
    
    const int numTiles = pDim/WGS;
    for (int t=0; t<numTiles; t++) {
 
        const int tiledRow = WGS*t + row;
        const int tiledCol = WGS*t + col;
        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
        #endif
        for (int w=0; w<TW; w++) {
            tempA[row][col+w*STRIDE] = matrixA[globalRow+(tiledCol + w*STRIDE)*mDim];
            tempB[row][col+w*STRIDE] = matrixB[(globalCol + w*STRIDE) + tiledRow*nDim];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
 
        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(WGS2)))
        #endif
        for (int k=0; k<WGS; k++) {
            #if KERNEL3C == true
            __attribute__((opencl_unroll_hint(TW2)))
            #endif
            for (int w=0; w<TW; ++w) {
                temp[w] += tempA[row][k] * tempB[k][col + w*STRIDE];
            }
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; ++w) {
        matrixC[(globalCol + w*STRIDE) + globalRow*nDim] = clamp((float)temp[w],-0.005f,0.005f);
    }
}


__kernel void Multiply_Deltas_Buffers_Sigmoid(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
    const int row = get_local_id(0); // local row ID (max: WGS)
    const int col = get_local_id(1); // local col ID (max: WGS/TW == RWGS)
    const int globalRow = WGS*get_group_id(0)+row; // Row ID of C (0..M)
    const int globalCol = WGS*get_group_id(1)+col; // Col ID of C (0..N)
 
    local float tempA[WGS][WGS];
    local float tempB[WGS][WGS];
 
    float temp[TW];
    #if KERNEL3C == true
    __attribute__((opencl_unroll_hint(TW)))
    #endif
    for (int w=0; w<TW; w++) {
        temp[w] = 0.0f;
    }
    
    const int numTiles = pDim/WGS;
    for (int t=0; t<numTiles; t++) {
 
        const int tiledRow = WGS*t + row;
        const int tiledCol = WGS*t + col;
        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
        #endif
        for (int w=0; w<TW; w++) {
            tempA[row][col+w*STRIDE] = matrixA[globalRow+(tiledCol + w*STRIDE)*mDim];
            tempB[row][col+w*STRIDE] = matrixB[(globalCol + w*STRIDE) + tiledRow*nDim];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(WGS2)))
        #endif
        for (int k=0; k<WGS; k++) {
            #if KERNEL3C == true
            __attribute__((opencl_unroll_hint(TW2)))
            #endif
            for (int w=0; w<TW; ++w) {
                temp[w] += tempA[row][k] * tempB[k][col + w*STRIDE];
            }
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
     #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; ++w) {
        matrixC[(globalCol + w*STRIDE) + globalRow*nDim] = clamp((float)(temp[w]* matrixD[(globalCol + w*STRIDE) + globalRow*nDim] * (1.0f - matrixD[(globalCol + w*STRIDE) + globalRow*nDim])),-0.005f,0.005f);
    }
}

__kernel void Multiply_Deltas_Buffers_Tanh(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = WGS*get_group_id(0)+row;
    const int globalCol = WGS*get_group_id(1)+col;
 
    local float tempA[WGS][WGS];
    local float tempB[WGS][WGS];
 
    float temp[TW];
    #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; w++) {
        temp[w] = 0.0f;
    }
    
    const int numTiles = pDim/WGS;
    for (int t=0; t<numTiles; t++) {
 
        const int tiledRow = WGS*t + row;
        const int tiledCol = WGS*t + col;
        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
        #endif
        for (int w=0; w<TW; w++) {
            tempA[row][col+w*STRIDE] = matrixA[globalRow+(tiledCol + w*STRIDE)*mDim];
            tempB[row][col+w*STRIDE] = matrixB[(globalCol + w*STRIDE) + tiledRow*nDim];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(WGS2)))
        #endif
        for (int k=0; k<WGS; k++) {
            #if KERNEL3C == true
            __attribute__((opencl_unroll_hint(TW2)))
            #endif
            for (int w=0; w<TW; ++w) {
                temp[w] += tempA[row][k] * tempB[k][col + w*STRIDE];
            }
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; ++w) {
        matrixC[(globalCol + w*STRIDE) + globalRow*nDim] = clamp((float)(temp[w]*(1 - pow(matrixD[(globalCol + w*STRIDE) + globalRow*nDim],2))),-0.005f,0.005f);
    }
}

__kernel void Multiply_Deltas_Buffers_ReLU(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = WGS*get_group_id(0)+row;
    const int globalCol = WGS*get_group_id(1)+col;
 
    local float tempA[WGS][WGS];
    local float tempB[WGS][WGS];
 
    float temp[TW];
    #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; w++) {
        temp[w] = 0.0f;
    }
    
    const int numTiles = pDim/WGS;
    for (int t=0; t<numTiles; t++) {
 
        const int tiledRow = WGS*t + row;
        const int tiledCol = WGS*t + col;
        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
        #endif
        for (int w=0; w<TW; w++) {
            tempA[row][col+w*STRIDE] = matrixA[globalRow+(tiledCol + w*STRIDE)*mDim];
            tempB[row][col+w*STRIDE] = matrixB[(globalCol + w*STRIDE) + tiledRow*nDim];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
 
        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(WGS2)))
        #endif
        for (int k=0; k<WGS; k++) {
            #if KERNEL3C == true
            __attribute__((opencl_unroll_hint(TW2)))
            #endif
            for (int w=0; w<TW; ++w) {
                temp[w] += tempA[row][k] * tempB[k][col + w*STRIDE];
            }
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; ++w) {
        matrixC[(globalCol + w*STRIDE) + globalRow*nDim] = clamp((float)(temp[w]*(matrixD[(globalCol + w*STRIDE) + globalRow*nDim]>0.0? 1.0:0.0)),-0.005f,0.005f);
    }
}

__kernel void Update_Weights_Buffers(global float* matrixA, global float* matrixB, global float* matrixC
,const int mDim, const int pDim, const int nDim, global float* biases, const float offset)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = WGS*get_group_id(0)+row;
    const int globalCol = WGS*get_group_id(1)+col;
 
    local float tempA[WGS][WGS];
    local float tempB[WGS][WGS];
 
    float temp[TW];
    for (int w=0; w<TW; w++) {
        temp[w] = 0.0f;
    }
    
    const int numTiles = pDim/WGS;
    for (int t=0; t<numTiles; t++) {
 
        const int tiledRow = WGS*t + row;
        const int tiledCol = WGS*t + col;

        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
        #endif
        for (int w=0; w<TW; w++) {
            tempA[row][col+w*STRIDE] = matrixA[globalRow*pDim+(tiledCol + w*STRIDE)];
            tempB[row][col+w*STRIDE] = matrixB[(globalCol + w*STRIDE)*pDim + tiledRow];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(WGS2)))
        #endif
        for (int k=0; k<WGS; k++) {
            #if KERNEL3C == true
            __attribute__((opencl_unroll_hint(TW2)))
            #endif
            for (int w=0; w<TW; ++w) {
                temp[w] += tempA[row][k] * tempB[k][col + w*STRIDE];
            }
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
    #endif
    for (int w=0; w<TW; ++w) {
        matrixC[(globalCol + w*STRIDE) + globalRow*nDim] = matrixC[(globalCol + w*STRIDE) + globalRow*nDim] - offset*temp[w]/(float)pDim;
    }
    if (globalCol==0){
        float tempBias=0.0f;
        #if KERNEL3C == true
        __attribute__((opencl_unroll_hint(TW2)))
        #endif
        for (int p = 0 ; p < pDim; p++){
            tempBias+= matrixA[globalRow*pDim+p];
        }
        biases[globalRow] = biases[globalRow] - offset*tempBias / (float)pDim;
    }
}