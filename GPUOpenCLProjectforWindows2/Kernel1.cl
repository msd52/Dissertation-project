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
#define DEBUG_FORWARD false
#define DEBUG_DELTAS false
#define DEBUG_UPDATE false


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


__kernel void Multiply_Buffer_Identity(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases)
{   
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    float finalValue = 0.0;

    for (int p = 0 ; p < pDim ; p++){
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

__kernel void Multiply_Buffer_Sigmoid(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases )
{
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    float finalValue = 0.0;

    for (int p = 0 ; p < pDim ; p++){
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

__kernel void Multiply_Buffer_Tanh(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases )
{
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    float finalValue = 0.0;

    for (int p = 0 ; p < pDim ; p++){
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

__kernel void Multiply_Buffer_ReLU(global float* matrixA, global float* matrixB, global float* matrixC,
const int mDim, const int pDim, const int nDim, global float* biases )
{
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    float finalValue = 0.0;

    for (int p = 0 ; p < pDim ; p++){
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
__kernel void Multiply_Deltas_Buffers_Identity(global float* weightsMatrix, global float* deltasMatrixOld, global float* deltasMatrixNew,
const int lNew, const int lOld, const int batchSize, global float* outputs)
{
    const int x = get_global_id(0); //the row specification, from 0 to lNew-1
    const int y = get_global_id(1); //sample point id
    float A,B, temp = 0;
    
    for (int p = 0 ; p < lOld ; p++){
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

__kernel void Multiply_Deltas_Buffers_Sigmoid(global float* weightsMatrix, global float* deltasMatrixOld, global float* deltasMatrixNew,
const int lNew, const int lOld, const int batchSize, global float* outputs)
{
    const int x = get_global_id(0); //the row specification, from 0 to lNew-1
    const int y = get_global_id(1); //sample point id
    float A,B, temp = 0;
    
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

__kernel void Multiply_Deltas_Buffers_Tanh(global float* weightsMatrix, global float* deltasMatrixOld, global float* deltasMatrixNew,
const int lNew, const int lOld, const int batchSize, global float* outputs)
{
    const int x = get_global_id(0); //the row specification, from 0 to lNew-1
    const int y = get_global_id(1); //sample point id
    float A,B, temp = 0;
    
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

__kernel void Multiply_Deltas_Buffers_ReLU(global float* weightsMatrix, global float* deltasMatrixOld, global float* deltasMatrixNew,
const int lNew, const int lOld, const int batchSize, global float* outputs)
{
    const int x = get_global_id(0); //the row specification, from 0 to lNew-1
    const int y = get_global_id(1); //sample point id
    float A,B, temp = 0;
    
    for (int p = 0 ; p < lOld ; p++){
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
__kernel void Update_Weights_Buffers(global float* deltasMatrix, global float* outputsMatrix, global float* weightsMatrix
,const int deltasDim, const int batchSize, const int outputsDim, global float* biases, const float learning_rate)
{
    const int x = get_global_id(0); //the row specification, from 0 to deltasDim-1
    const int y = get_global_id(1); //the column specification, from 0 to outputsDim-1
    float A,B, temp=0.0f;

    for (int p = 0 ; p < batchSize ; p++){
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
        for (int p = 0 ; p < batchSize ; p++){
            tempBias+= deltasMatrix[x*batchSize+p];
        }
        biases[x] = biases[x] - learning_rate*tempBias / (float)batchSize;
    }
}