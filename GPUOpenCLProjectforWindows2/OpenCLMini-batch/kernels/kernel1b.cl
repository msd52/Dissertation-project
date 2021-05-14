__kernel void Multiply_Buffer_Identity(global float* restrict matrixA, global float* restrict matrixB, global float* restrict matrixC,
const int mDim, const int pDim, const int nDim, global float* restrict biases)
{   
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    float finalValue = 0.0;

    __attribute__((opencl_unroll_hint(16)))
    for (int p = 0 ; p < pDim ; p++){
        prefetch(&(matrixB[nDim*(p+4)+c]), 1);
        finalValue+=matrixA[pDim*r+p]*matrixB[nDim*p+c];
    }
    matrixC[r*nDim+c] = finalValue+biases[r];
    #if DEBUG_FORWARD == true
        if (r==0 && c==0){
            printf("in mul id kernel 1\n");
        }
        printf("Identity forward is %d %d, is %f \n", r, c, matrixC[r*nDim+c]);
    #endif
}

__kernel void Multiply_Buffer_Sigmoid(global float* restrict matrixA, global float* restrict matrixB, global float* restrict matrixC,
const int mDim, const int pDim, const int nDim, global float* restrict biases )
{
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    float finalValue = 0.0;

     __attribute__((opencl_unroll_hint(16)))
    for (int p = 0 ; p < pDim ; p++){
        prefetch(&(matrixB[nDim*(p+4)+c]), 1);
        finalValue+=matrixA[pDim*r+p]*matrixB[nDim*p+c];
    }
    matrixC[r*nDim+c] = 1.0 / (1.0 + exp(-finalValue-biases[r]));
    #if DEBUG_FORWARD
        if (r==0 && c==0){
            printf("in mul sigmoid kernel 1\n");
        }
        printf("Sigmoid forward %d %d, is %f \n", r, c, matrixC[r*nDim+c]);
    #endif
}

__kernel void Multiply_Buffer_Tanh(global float* restrict matrixA, global float* restrict matrixB, global float* restrict matrixC,
const int mDim, const int pDim, const int nDim, global float* restrict biases )
{
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    float finalValue = 0.0;

    __attribute__((opencl_unroll_hint(16)))
    for (int p = 0 ; p < pDim ; p++){
        prefetch(&(matrixB[nDim*(p+4)+c]), 1);
        finalValue+=matrixA[pDim*r+p]*matrixB[nDim*p+c];
    }
    matrixC[r*nDim+c] = tanh(finalValue+biases[r]);
    #if DEBUG_FORWARD
        if (r==0 && c==0){
            printf("in mul tanh kernel 1\n");
        }
        printf("Tanh forward %d %d, is %f \n", r, c, matrixC[r*nDim+c]);
    #endif
}

__kernel void Multiply_Buffer_ReLU(global float* restrict matrixA, global float* restrict matrixB, global float* restrict matrixC,
const int mDim, const int pDim, const int nDim, global float* restrict biases )
{
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    float finalValue = 0.0;

    __attribute__((opencl_unroll_hint(16)))
    for (int p = 0 ; p < pDim ; p++){
        prefetch(&(matrixB[nDim*(p+4)+c]), 1);
        finalValue+=matrixA[pDim*r+p]*matrixB[nDim*p+c];
    }
    matrixC[r*nDim+c] = fmax(finalValue+biases[r],0);
    #if DEBUG_FORWARD
        if (r==0 && c==0){
            printf("in mul relu kernel 1\n");
        }
        printf("ReLU forward %d %d, is %f \n", r, c, matrixC[r*nDim+c]);
    #endif
}

//New refers to current layer and old refers to previous layer (all while traversing the network backwards)
//So the weightMatrix has dimensions lOldxlNew. We have to compute lNew many deltas, so there are lNew many work items globally
__kernel void Multiply_Deltas_Buffers_Identity(global float* restrict weightsMatrix, global float* restrict deltasMatrixOld, global float* restrict deltasMatrixNew,
const int lNew, const int lOld, const int batchSize, global float* restrict outputs)
{
    const int x = get_global_id(0); //the row specification, from 0 to lNew-1
    const int y = get_global_id(1); //sample point id
    float A,B, temp = 0;
    
    __attribute__((opencl_unroll_hint(16)))
    for (int p = 0 ; p < lOld ; p++){
        prefetch(&deltasMatrixOld[(p+4)*batchSize+y], 1);
        A = weightsMatrix[p*lNew+x];
        B = deltasMatrixOld[p*batchSize+y];
        temp+=A*B;
    }

    deltasMatrixNew[x*batchSize + y] = clamp((float)temp,-0.005f,0.005f);
    #if DEBUG_DELTAS
        if (x==0 && y==0){
            printf("in delta id kernel 1\n");
            printf("dimensions are %d %d %d 1\n", lNew, lOld, batchSize);
        }
        //printf("Identity Deltas %d %d, is %f \n", r, c, matrixC[r*nDim+c]);
    #endif
}

__kernel void Multiply_Deltas_Buffers_Sigmoid(global float* restrict weightsMatrix, global float* restrict deltasMatrixOld, global float* restrict deltasMatrixNew,
const int lNew, const int lOld, const int batchSize, global float* restrict outputs)
{
    const int x = get_global_id(0); //the row specification, from 0 to lNew-1
    const int y = get_global_id(1); //sample point id
    float A,B, temp = 0;
    
    __attribute__((opencl_unroll_hint(16)))
    for (int p = 0 ; p < lOld ; p++){
        A = weightsMatrix[p*lNew+x];
        B = deltasMatrixOld[p*batchSize+y];
        temp+=A*B;
    }
    deltasMatrixNew[x*batchSize + y] = clamp((float)(temp * outputs[x*batchSize+y] * (1.0f - outputs[x*batchSize+y])) , -0.005f , 0.005f);
    #if DEBUG_DELTAS
        if (x==0 && y==0){
            printf("in delta sigmoid kernel 1\n");
            printf("dimensions are %d %d %d 1\n", lNew, lOld, batchSize);
        }
        //printf("Sigmoid Deltas %d %d, is %f \n", r, c, matrixC[r*nDim+c]);
    #endif
}

__kernel void Multiply_Deltas_Buffers_Tanh(global float* restrict weightsMatrix, global float* restrict deltasMatrixOld, global float* restrict deltasMatrixNew,
const int lNew, const int lOld, const int batchSize, global float* restrict outputs)
{
    const int x = get_global_id(0); //the row specification, from 0 to lNew-1
    const int y = get_global_id(1); //sample point id
    float A,B, temp = 0;
        
    __attribute__((opencl_unroll_hint(16)))    
    for (int p = 0 ; p < lOld ; p++){
        A = weightsMatrix[p*lNew+x];
        B = deltasMatrixOld[p*batchSize+y];
        temp+=A*B;
    }

    deltasMatrixNew[x*batchSize + y] = clamp((float)(temp * (1.0f - pow(outputs[x*batchSize+y],2))),-0.005f,0.005f);
    #if DEBUG_DELTAS
        if (x==0 && y==0){
            printf("in delta tanh kernel 1\n");
            printf("dimensions are %d %d %d 1\n", lNew, lOld, batchSize);
        }
        //printf("Tanh Deltas %d %d, is %f \n", r, c, matrixC[r*nDim+c]);
    #endif
}

__kernel void Multiply_Deltas_Buffers_ReLU(global float* restrict weightsMatrix, global float* restrict deltasMatrixOld, global float* restrict deltasMatrixNew,
const int lNew, const int lOld, const int batchSize, global float* restrict outputs)
{
    const int x = get_global_id(0); //the row specification, from 0 to lNew-1
    const int y = get_global_id(1); //sample point id
    float A,B, temp = 0;
    
    __attribute__((opencl_unroll_hint(16)))
    for (int p = 0 ; p < lOld ; p++){
        prefetch(&deltasMatrixOld[(p+4)*batchSize+y], 1);
        A = weightsMatrix[p*lNew+x];
        B = deltasMatrixOld[p*batchSize+y];
        temp+=A*B;
    }

    deltasMatrixNew[x*batchSize + y] = clamp((float)(temp * (outputs[x*batchSize + y] > 0.0? 1.0:0.0)),-0.005f,0.005f);
    #if DEBUG_DELTAS
        if (x==0 && y==0){
            printf("in delta relu kernel 1\n");
            printf("dimensions are %d %d %d 1\n", lNew, lOld, batchSize);
        }
        //printf("ReLU Deltas %d %d, is %f \n", r, c, matrixC[r*nDim+c]);
    #endif
}

//Performs matrixC = matrixA x matrixB.T
__kernel void Update_Weights_Buffers(global float* restrict deltasMatrix, global float* restrict outputsMatrix, global float* restrict weightsMatrix
,const int deltasDim, const int batchSize, const int outputsDim, global float* restrict biases, const float learning_rate)
{
    const int x = get_global_id(0); //the row specification, from 0 to deltasDim-1
    const int y = get_global_id(1); //the column specification, from 0 to outputsDim-1
    float A,B, temp=0.0f;

    __attribute__((opencl_unroll_hint(16)))
    for (int p = 0 ; p < batchSize ; p++){
        prefetch(&deltasMatrix[x*batchSize+p+4], 1);
        A = deltasMatrix[x*batchSize+p];
        B = outputsMatrix[y*batchSize+p];
        temp+=A*B;
    }

    temp =  learning_rate*temp / (float)batchSize;

    weightsMatrix[x*outputsDim + y] = weightsMatrix[x*outputsDim + y] - temp;
    #if DEBUG_UPDATE
        if (x==0&&y==0){
            printf("in weight update kernel 1");
        }
        printf("Update %d %d, is %f \n", r, c, matrixC[r*nDim+c]);
    #endif

    if (y==0){

        float tempBias=0.0f;
        __attribute__((opencl_unroll_hint(16)))
        for (int p = 0 ; p < batchSize ; p++){
            prefetch(&deltasMatrix[x*batchSize+p+4], 1);
            tempBias+= deltasMatrix[x*batchSize+p];
        }
        biases[x] = biases[x] - learning_rate*tempBias / (float)batchSize;
    }
}