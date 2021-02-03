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

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void Add(read_only image2d_t imageA, read_only image2d_t imageB, write_only image2d_t imageC)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    uint A = read_imageui(imageA, sampler, (int2)(x, y)).x;
    uint B = read_imageui(imageB, sampler, (int2)(x, y)).x;

    write_imageui(imageC, (int2)(x, y), A + B);
}

__kernel void Multiply_Buffer(global int* matrixA, global int* matrixB, global int* matrixC,
const int pDim, const int mDim, const int nDim)
{
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    int finalValue = 0;
        printf("start %d %d \n", r, c);

    for (int p = 0 ; p < pDim ; p++){
        finalValue+=matrixA[pDim*r+p]*matrixB[nDim*p+c];
         printf("id is %d %d, values are %d %d \n ", r, c, matrixA[pDim*r+p], matrixB[nDim*p+c] );
    }
    matrixC[r*nDim+c] = finalValue;
    printf("id is %d %d, final value is %d \n \n \n ", r, c, finalValue);
}

__kernel void Multiply_2(read_only image2d_t matrixA, read_only image2d_t matrixB, write_only image2d_t matrixC,
const int pDim)
{
    const int x = get_global_id(0); //the row specification
    const int y = get_global_id(1); //the column specification
    int A = 0, B = 0, temp = 0;
    
    printf("start %d %d \n", x, y);

    for (int p = 0 ; p < pDim ; p++){
        A = read_imageui(matrixA, sampler, (int2)(p, x)).x;
        B = read_imageui(matrixB, sampler, (int2)(y, p)).x;
        printf("id is %d %d, values are %d %d \n ", x, y, A, B );
        temp+=A*B;
    }
    printf("id is %d %d, final value is %d \n \n \n ", x, y, temp);

    write_imageui(matrixC, (int2)(y, x), temp);
}

//New refers to most recent layer and old refers to second most recent layer (all while traversing the network backwards)
//So the weightMatrix has dimensions lOldxlNew. We have to compute lNew many deltas, so there are lNew many work items globally
__kernel void Multiply_Deltas(read_only image2d_t weightsMatrix, read_only image2d_t deltasMatrixOld, write_only image2d_t deltasMatrixNew,
const int lOld)
{
    printf("In MultiplyDeltas");
    const int x = get_global_id(0); //the row specification, from 0 to lNew-1
    const int y = get_global_id(1);
    int A = 0, B = 0, temp = 0;
    
    printf("start %d %d \n", x, y);

    for (int p = 0 ; p < lOld ; p++){
        A = read_imageui(weightsMatrix, sampler, (int2)(x, p)).x;
        B = read_imageui(deltasMatrixOld, sampler, (int2)(y, p)).x;
        printf("id is %d %d, values are %d %d \n ", x, y, A, B );
        temp+=A*B;
    }
    printf("id is %d %d, final value is %d \n \n \n ", x, y, temp);

    write_imageui(deltasMatrixNew, (int2)(y,x), temp);
}

__kernel void Update_Weights(read_only image2d_t deltasMatrix, read_only image2d_t outputsMatrix, global float* plss)
{
    printf("In Update_Weights");
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int A = 0, B = 0;
    
    printf("start %d %d \n", x, y);

    A = read_imageui(deltasMatrix, sampler, (int2)(0, x)).x;
    B = read_imageui(outputsMatrix, sampler, (int2)(0, y)).x;
    printf("id is %d %d, values are %d %d \n ", x, y, A, B );
    int temp=A*B;

    printf("id is %d %d, final value is %d \n \n \n ", x, y, temp);


    //write_imageui(weightsMatrix, (int2)(y,x), temp);
}