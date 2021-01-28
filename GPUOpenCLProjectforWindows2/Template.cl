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

__kernel void Multiply_1(const int mDim, const int nDim, const int pDim, 
global float* matrixA, global float* matrixB, global float* matrixC)
{
    const int rowInd = get_global_id(0);
    const int colInd = get_global_id(1);
    int finalValue = 0;

    for (int p = 0 ; p < pDim ; p++){
        finalValue+=matrixA[pDim*r+p]*matrixB[nDim*p+c];
    }
    C[r*nDim+c] = finalValue;
}

__kernel void Multiply_2(const int mDim, const int nDim, const int pDim, 
global float* matrixA, global float* matrixB, global float* matrixC)
{
    const int rowInd = get_global_id(0);
    const int colInd = get_global_id(1);
    int finalValue = 0;

    for (int p = 0 ; p < pDim ; p++){
        finalValue+=matrixA[pDim*r+p]*matrixB[nDim*p+c];
    }
    C[r*nDim+c] = finalValue;
}

{ \n" \
" int k,j; \n" \
" int i = get_global_id(0); \n" \
" float tmp; \n" \
" if( (i < Ndim) ) \n" \
" { \n" \
" for(j=0;j<Mdim;j++){ \n" \
" tmp = 0.0; \n" \
" for(k=0;k<Pdim;k++) \n" \
" tmp += A[i*Ndim+k] * B[k*Pdim+j]; \n" \
" C[i*Ndim+j] = tmp; \n" \
" } \n" \
" } \n" \
} 
