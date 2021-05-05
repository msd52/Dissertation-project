/*****************************************************************************
 * Copyright (c) 2013-2016 Intel Corporation
 * All rights reserved.
 *
 * WARRANTY DISCLAIMER
 *
 * THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
 * MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Intel Corporation is the author of the Materials, and requests that all
 * problem reports or change requests be submitted to it directly
 *****************************************************************************/

//constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define TS 16
#define WPT 8
#define RTS 2
#define WPTM 4
#define WPTN 4


//2D register tiling
/*__kernel void Matrix_Multiply_Kernel_4(global float* matrixA, global float* matrixB, global float* matrixC,
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
    float acc[WPTM][WPTN];
 
    // Initialise the accumulation registers
    for (int wm=0; wm<WPTM; wm++) {
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
    const int numTiles = pDim/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        for (int w=0; w<WPT; w++) {
            Asub[row][col+w*RTS] = matrixA[globalRow*pDim+(tiledCol + w*RTS)];
            Bsub[row][col+w*RTS] = matrixB[(globalCol + w*RTS) + tiledRow*pDim];
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
        matrixC[(globalCol + w*RTS) + globalRow*pDim] = acc[w];
    }
}*/

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

__kernel void Matrix_Multiply_Kernel_3(global float* matrixA, global float* matrixB, global float* matrixC,
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
}

__kernel void Matrix_Multiply_Kernel_2(global float* matrixA, global float* matrixB, global float* matrixC,
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
}

__kernel void scalar_mul(__global const float *a,
        __global const float *b,
        __global float *result)
{
        int i = get_global_id(0);
        result[i] =a[i] * b[i];        
}


__kernel void Matrix_Multiply_Kernel_1(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim)
{
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    float finalValue = 0.0;
    //printf("start IBuffer %d %d \n", r, c);
    //printf("pls IBuffer be right %d %d should be %f %f and ndim is %d\n", r, c, matrixB[0], matrixB[1], nDim);
    for (int p = 0 ; p < pDim ; p++){
        finalValue+=matrixA[pDim*r+p]*matrixB[nDim*p+c];
        //printf("id is %d %d, values are %f %f \n ", r, c, matrixA[pDim*r+p], matrixB[nDim*p+c] );
    }
    matrixC[r*nDim+c] = finalValue;
    //printf("id IBuffer is %d %d, final value is %f \n \n \n ", r, c, finalValue);
    //printf("id IBuffer is %d %d, post function appied is %f \n \n \n ", r, c, finalValue);
}


__kernel void Multiply_Buffer_Identity(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim)
{   
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    float finalValue = 0.0;

    //printf("start IBuffer %d %d \n", r, c);
    //printf("pls IBuffer be right %d %d should be %f %f and ndim is %d\n", r, c, matrixB[0], matrixB[1], nDim);
    for (int p = 0 ; p < pDim ; p++){
        finalValue+=matrixA[pDim*r+p]*matrixB[nDim*p+c];
        //if (matrixB[nDim*p+c]>100.0f){
        //        printf("id is %d %d, pval is %d values are %f %f \n ", r, c, p, matrixA[pDim*r+p], matrixB[nDim*p+c] );
        //}
    }
    matrixC[r*nDim+c] = finalValue;
    //printf("id IBuffer is %d %d, final value is %f \n \n \n ", r, c, finalValue);
}

__kernel void Multiply_Buffer_Sigmoid(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim)
{
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    float finalValue = 0.0;
    //printf("start %d %d \n", r, c);
    //printf("pls be right %d %d should be %f %f and ndim is %d\n", r, c, matrixB[0], matrixB[1], nDim);
    for (int p = 0 ; p < pDim ; p++){
        finalValue+=matrixA[pDim*r+p]*matrixB[nDim*p+c];
        //printf("id is %d %d, values are %f %f \n ", r, c, matrixA[pDim*r+p], matrixB[nDim*p+c] );
    }
    matrixC[r*nDim+c] = 1.0 / (1.0 + exp(-finalValue));
    //printf("id is %d %d, final value is %f \n \n \n ", r, c, finalValue);
    //printf("id is %d %d, post function applied is %f \n \n \n ", r, c, matrixC[r*nDim+c]);
}

__kernel void Multiply_Buffer_Tanh(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim)
{
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    float finalValue = 0.0;
    //printf("start %d %d \n", r, c);
    //printf("pls be right %d %d should be %f %f and ndim is %d\n", r, c, matrixB[0], matrixB[1], nDim);
    for (int p = 0 ; p < pDim ; p++){
        finalValue+=matrixA[pDim*r+p]*matrixB[nDim*p+c];
        //printf("id is %d %d, values are %f %f \n ", r, c, matrixA[pDim*r+p], matrixB[nDim*p+c] );
    }
    matrixC[r*nDim+c] = tanh(finalValue);
    //printf("id is %d %d, final value is %f \n \n \n ", r, c, finalValue);
    //printf("id is %d %d, post function applied is %f \n \n \n ", r, c, matrixC[r*nDim+c]);
}

__kernel void Multiply_Buffer_ReLU(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim)
{
    const int r = get_global_id(0);
    const int c = get_global_id(1);

    float finalValue = 0.0;
    //printf("start %d %d \n", r, c);
    //printf("pls be right %d %d should be %f %f and ndim is %d\n", r, c, matrixB[0], matrixB[1], nDim);
    for (int p = 0 ; p < pDim ; p++){
        finalValue+=matrixA[pDim*r+p]*matrixB[nDim*p+c];
        //printf("id is %d %d, values are %f %f \n ", r, c, matrixA[pDim*r+p], matrixB[nDim*p+c] );
    }
    matrixC[r*nDim+c] = fmax(finalValue,0);
    //printf("id is %d %d, final value is %f \n \n \n ", r, c, finalValue);
    //printf("id is %d %d, post function applied is %f \n \n \n ", r, c, matrixC[r*nDim+c]);
}

//New refers to current layer and old refers to previous layer (all while traversing the network backwards)
//So the weightMatrix has dimensions lOldxlNew. We have to compute lNew many deltas, so there are lNew many work items globally
__kernel void Multiply_Deltas_Buffers_Identity(global float* weightsMatrix, global float* deltasMatrixOld, global float* deltasMatrixNew,
const int lNew, const int lOld, const int batchSize, global float* outputs)
{
    //printf("In Multiply_Deltas_Buffers_Identity");
    const int x = get_global_id(0); //the row specification, from 0 to lNew-1
    const int y = get_global_id(1); //sample point id
    float A,B, temp = 0;
    
    //printf("start %d %d \n", x, y);

    for (int p = 0 ; p < lOld ; p++){
        A = weightsMatrix[p*lNew+x];
        B = deltasMatrixOld[p*batchSize+y];
        //printf("id is %d %d, values are %f %f \n ", x, y, A, B );
        temp+=A*B;
    }
    //printf("id is %d %d, final value is %f \n \n \n ", x, y, temp);

    deltasMatrixNew[x*batchSize + y] = clamp((float)temp,-0.005f,0.005f);
}

__kernel void Multiply_Deltas_Buffers_Sigmoid(global float* weightsMatrix, global float* deltasMatrixOld, global float* deltasMatrixNew,
const int lNew, const int lOld, const int batchSize, global float* outputs)
{
    //printf("In Multiply_Deltas_Buffers_Sigmoid");
    const int x = get_global_id(0); //the row specification, from 0 to lNew-1
    const int y = get_global_id(1); //sample point id
    float A,B, temp = 0;
    
    //printf("start %d %d \n", x, y);

    for (int p = 0 ; p < lOld ; p++){
        A = weightsMatrix[p*lNew+x];
        B = deltasMatrixOld[p*batchSize+y];
        printf("id is %d %d, values are %f %f \n ", x, y, A, B );
        temp+=A*B;
    }
   // printf("id is %d %d, final value is %f \n \n \n ", x, y, temp);

    deltasMatrixNew[x*batchSize + y] = clamp((float)temp * outputs[x*batchSize+y] * (1.0f - outputs[x*batchSize+y]) , -0.005f , 0.005f);
}

__kernel void Multiply_Deltas_Buffers_Tanh(global float* weightsMatrix, global float* deltasMatrixOld, global float* deltasMatrixNew,
const int lNew, const int lOld, const int batchSize, global float* outputs)
{
    printf("In Multiply_Deltas_Buffers_Tanh");
    const int x = get_global_id(0); //the row specification, from 0 to lNew-1
    const int y = get_global_id(1); //sample point id
    float A,B, temp = 0;
    
    printf("start %d %d \n", x, y);

    for (int p = 0 ; p < lOld ; p++){
        A = weightsMatrix[p*lNew+x];
        B = deltasMatrixOld[p*batchSize+y];
        printf("id is %d %d, values are %f %f \n ", x, y, A, B );
        temp+=A*B;
    }
    printf("id is %d %d, final value is %f \n \n \n ", x, y, temp);

    deltasMatrixNew[x*batchSize + y] = clamp((float)(temp * (1.0f - pow(outputs[x*batchSize+y],2))),-0.005f,0.005f);
}

__kernel void Multiply_Deltas_Buffers_ReLU(global float* weightsMatrix, global float* deltasMatrixOld, global float* deltasMatrixNew,
const int lNew, const int lOld, const int batchSize, global float* outputs)
{
    //printf("In Multiply_Deltas_Buffers_ReLU");
    const int x = get_global_id(0); //the row specification, from 0 to lNew-1
    const int y = get_global_id(1); //sample point id

    float A,B, temp = 0;
    
    //printf("start %d %d \n", x, y);

    for (int p = 0 ; p < lOld ; p++){
        A = weightsMatrix[p*lNew+x];
        B = deltasMatrixOld[p*batchSize+y];
        //printf("id is %d %d, values are %f %f \n ", x, y, A, B );
        temp+=A*B;
    }
    //printf("id is %d %d, final value is %f \n \n \n ", x, y, temp);

    deltasMatrixNew[x*batchSize + y] = clamp((float)(temp * (outputs[x*batchSize + y] > 0.0? 1.0:0.0)),-0.005f,0.005f);
}

__kernel void Update_Weights_Buffers(global float* deltasMatrix, global float* outputsMatrix, global float* weightsMatrix
,const int deltasDim, const int batchSize, const int outputsDim,  const float learning_rate)
{

    //printf("Update_Weights_Buffers");
    const int x = get_global_id(0); //the row specification, from 0 to deltasDim-1
    const int y = get_global_id(1); //the column specification, from 0 to outputsDim-1
    float A,B, temp=0;
    //printf("start %d %d \n", x, y);
    
    for (int p = 0 ; p < batchSize ; p++){
        A = deltasMatrix[x*batchSize+p];
        B = outputsMatrix[y*batchSize+p];
        //printf("id is %d %d, values are %f %f \n ", x, y, A, B );
        temp+=A*B;
    }
    //temp = clamp(learning_rate*temp / (float)batchSize,-0.005f,0.005f);
    temp =  learning_rate*temp / (float)batchSize;
    //printf("id is %d %d, pre update weight  is %f \n", x, y, temp);
    weightsMatrix[x*outputsDim + y] = weightsMatrix[x*outputsDim + y] - temp;
    //printf("id is %d %d, update value is %f\n", x, y, temp);
    //printf("id is %d %d, post update weight is %f", x, y, weightsMatrix[x*outputsDim+y]);
}