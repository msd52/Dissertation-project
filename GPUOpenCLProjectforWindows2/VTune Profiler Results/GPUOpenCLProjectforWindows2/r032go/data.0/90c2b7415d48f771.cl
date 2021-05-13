//File containing implementation with basic blocked matrix multiplication using local memory

//debug macros allow monitoring the values to detect underflow or overflow
#define WGS 16
#define DEBUG_FORWARD false
#define DEBUG_DELTAS false
#define DEBUG_UPDATE false

__kernel void Multiply_Buffer_Identity(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases)
{
   
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    float temp = 0.0f;

    local float Asub[2*WGS][2*WGS];
    local float Bsub[2*WGS][2*WGS];
     
    const int blocks= pDim/WGS;
    for (int t=0; t<blocks; t++) {
 
        const int tempRow = WGS*t + row;
        const int tempCol = WGS*t + col;
        Asub[row][col] = matrixA[tempCol + globalRow*pDim];
        Bsub[row][col] = matrixB[globalCol+ tempRow*nDim];
        barrier(CLK_LOCAL_MEM_FENCE);
 
        for (int k=0; k<WGS; k++) {
            temp += Asub[row][k] * Bsub[k][col];
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    matrixC[globalCol + globalRow*nDim] = temp+biases[globalRow];
    #if DEBUG_FORWARD == true
        printf("Identity forward is %d %d, is %f \n", globalRow, globalCol, matrixC[globalCol + globalRow*nDim]);
    #endif
}




__kernel void Multiply_Buffer_Sigmoid(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
 
    local float Asub[WGS][WGS];
    local float Bsub[WGS][WGS];
 
    float temp = 0.0f;
    
    const int blocks = pDim/WGS;
    for (int t=0; t<blocks; t++) {
 
        const int tempRow = WGS*t + row;
        const int tempCol = WGS*t + col;
        Asub[row][col] = matrixA[tempCol + globalRow*pDim];
        Bsub[row][col] = matrixB[globalCol+ tempRow*nDim];
 
        barrier(CLK_LOCAL_MEM_FENCE);
 
        for (int k=0; k<WGS; k++) {
            temp += Asub[row][k] * Bsub[k][col];
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    matrixC[globalCol + globalRow*nDim] = 1.0 / (1.0 + exp(-temp-biases[globalRow]));
    #if DEBUG_FORWARD == true
        printf("Sigmoid forward is %d %d, is %f \n", globalRow, globalCol, matrixC[globalCol + globalRow*nDim]);
    #endif
}

__kernel void Multiply_Buffer_Tanh(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
 
    float temp = 0.0f;

    __local float Asub[WGS][WGS];
    __local float Bsub[WGS][WGS];
     
    const int blocks = pDim/WGS;
    for (int t=0; t<blocks; t++) {
 
        const int tempRow = WGS*t + row;
        const int tempCol = WGS*t + col;
        Asub[row][col] = matrixA[tempCol + globalRow*pDim];
        Bsub[row][col] = matrixB[globalCol+ tempRow*nDim];
 
        barrier(CLK_LOCAL_MEM_FENCE);
 
        for (int k=0; k<WGS; k++) {
            temp += Asub[row][k] * Bsub[k][col];
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    matrixC[globalCol + globalRow*nDim] = tanh(temp+biases[globalRow]);
    #if DEBUG_FORWARD == true
        printf("Tanh forward is %d %d, is %f \n", globalRow, globalCol, matrixC[globalCol + globalRow*nDim]);
    #endif
}

__kernel void Multiply_Buffer_ReLU(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    float temp = 0.0f;

    __local float Asub[WGS][WGS];
    __local float Bsub[WGS][WGS];
     
    const int blocks = pDim/WGS;
    for (int t=0; t<blocks; t++) {
 
        const int tempRow = WGS*t + row;
        const int tempCol = WGS*t + col;
        Asub[row][col] = matrixA[tempCol + globalRow*pDim];
        Bsub[row][col] = matrixB[globalCol+ tempRow*nDim];
 
        barrier(CLK_LOCAL_MEM_FENCE);
 
        for (int k=0; k<WGS; k++) {
            temp += Asub[row][k] * Bsub[k][col];
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    matrixC[globalCol + globalRow*nDim] = fmax(temp+biases[globalRow],0);
    #if DEBUG_FORWARD == true
        printf("ReLU forward is %d %d, is %f \n", globalRow, globalCol, matrixC[globalCol + globalRow*nDim]);
    #endif
}

__kernel void Multiply_Deltas_Buffers_Identity(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    float temp = 0.0f;

    __local float Asub[WGS][WGS];
    __local float Bsub[WGS][WGS];
     
    const int blocks = pDim/WGS;
    for (int t=0; t<blocks; t++) {
 
        const int tempRow = WGS*t + row;
        const int tempCol = WGS*t + col;

        Asub[row][col] = matrixA[tempCol*mDim + globalRow];
        Bsub[row][col] = matrixB[globalCol+ tempRow*nDim];
 
        barrier(CLK_LOCAL_MEM_FENCE);
 
        for (int k=0; k<WGS; k++) {
            temp += Asub[row][k] * Bsub[k][col];
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    matrixC[globalCol + globalRow*nDim] = clamp((float)temp,-0.005f,0.005f);
    #if DEBUG_DELTAS == true
        printf("Identity deltas is %d %d, is %f \n", globalRow, globalCol, matrixC[globalCol + globalRow*nDim]);
    #endif
}


__kernel void Multiply_Deltas_Buffers_Sigmoid(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    __local float Asub[WGS][WGS];
    __local float Bsub[WGS][WGS];
 
    float temp = 0.0f;
    
    const int blocks = pDim/WGS;
    for (int t=0; t<blocks; t++) {
 
        const int tempRow = WGS*t + row;
        const int tempCol = WGS*t + col;

        Asub[row][col] = matrixA[tempCol*mDim + globalRow];
        Bsub[row][col] = matrixB[globalCol+ tempRow*nDim];
 
        barrier(CLK_LOCAL_MEM_FENCE);
 
        for (int k=0; k<WGS; k++) {
            temp += Asub[row][k] * Bsub[k][col];
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    matrixC[globalCol + globalRow*nDim] = clamp((float)(temp*matrixD[globalCol + globalRow*nDim] * (1.0f - matrixD[globalCol + globalRow*nDim])),-0.005f,0.005f);
    #if DEBUG_DELTAS == true
        printf("Sigmoid deltas is %d %d, is %f \n", globalRow, globalCol, matrixC[globalCol + globalRow*nDim]);
    #endif
}

__kernel void Multiply_Deltas_Buffers_Tanh(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
 
    __local float Asub[WGS][WGS];
    __local float Bsub[WGS][WGS];
 
    float temp = 0.0f;
    
    const int blocks = pDim/WGS;
    for (int t=0; t<blocks; t++) {
 
        const int tempRow = WGS*t + row;
        const int tempCol = WGS*t + col;
        Asub[row][col] = matrixA[tempCol*mDim + globalRow];
        Bsub[row][col] = matrixB[globalCol+ tempRow*nDim];
 
        barrier(CLK_LOCAL_MEM_FENCE);
 
        for (int k=0; k<WGS; k++) {
            temp += Asub[row][k] * Bsub[k][col];
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    matrixC[globalCol + globalRow*nDim] = clamp((float)(temp*(1 - pow(matrixD[globalCol + globalRow*nDim],2))),-0.005f,0.005f);
    #if DEBUG_DELTAS == true
        printf("Tanh deltas is %d %d, is %f \n", globalRow, globalCol, matrixC[globalCol + globalRow*nDim]);
    #endif
}

__kernel void Multiply_Deltas_Buffers_ReLU(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* matrixD)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    float temp = 0.0f;

    __local float Asub[WGS][WGS];
    __local float Bsub[WGS][WGS];
 
    
    const int blocks = pDim/WGS;
    for (int t=0; t<blocks; t++) {
 
        const int tempRow = WGS*t + row;
        const int tempCol = WGS*t + col;
        Asub[row][col] = matrixA[tempCol*mDim + globalRow];
        Bsub[row][col] = matrixB[globalCol+ tempRow*nDim];
 
        barrier(CLK_LOCAL_MEM_FENCE);
 
        for (int k=0; k<WGS; k++) {
            temp += Asub[row][k] * Bsub[k][col];
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    matrixC[globalCol + globalRow*nDim] = clamp((float)(temp*(matrixD[globalCol + globalRow*nDim] > 0.0? 1.0:0.0)),-0.005f,0.005f);
    #if DEBUG_DELTAS == true
        printf("ReLU deltas is %d %d, is %f \n", globalRow, globalCol, matrixC[globalCol + globalRow*nDim]);
    #endif
}

__kernel void Update_Weights_Buffers(global float* matrixA, global float* matrixB, global float* matrixC
,const int mDim, const int pDim, const int nDim, global float* biases, const float offset)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    float temp = 0.0f;

    __local float Asub[WGS][WGS];
    __local float Bsub[WGS][WGS];
     
    const int blocks = pDim/WGS;
    for (int t=0; t<blocks; t++) {
        
        const int tempRow = WGS*t + row;
        const int tempCol = WGS*t + col;
        Asub[row][col] = matrixA[tempCol + globalRow*pDim];
        Bsub[row][col] = matrixB[globalCol*pDim+ tempRow];
 
        barrier(CLK_LOCAL_MEM_FENCE);
 
        for (int k=0; k<WGS; k++) {
            temp += Asub[row][k] * Bsub[k][col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #if DEBUG_UPDATE == true
        printf("Update is %d %d, is %f \n", globalRow, globalCol, matrixC[globalCol + globalRow*nDim]);
    #endif

    matrixC[globalCol + globalRow*nDim] = matrixC[globalCol + globalRow*nDim] - offset*temp/(float)pDim;
    if (globalCol==0){
        float tempBias=0.0f;
        for (int p = 0 ; p < pDim; p++){
            tempBias+= matrixA[globalRow*pDim+p];
        }
        biases[globalRow] = biases[globalRow] - offset*tempBias / (float)pDim;
    }
} 