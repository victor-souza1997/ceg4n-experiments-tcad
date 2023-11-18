
#ifndef QUANTIZED_H
#define QUANTIZED_H
// This file is computer-generated by onnx2c 
// (TODO: add creating command line here)
// (TODO: print creation date here )

// ONNX model:
// produced by pytorch, version 1.11.0
// ONNX IR version: 9
// Model documentation: 
/*

*/

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#define MAX(X,Y) ( X > Y ? X : Y)
#define MIN(X,Y) ( X < Y ? X : Y)
#define CLIP(X,L) ( MAX(MIN(X,L), -L) )

static const float quantized_tensor_Gemm_0_weight[6][7] = 
{
  {1.9650176763534545898f, 0.56143361330032348633f, -0.84215044975280761719f, 0.56143361330032348633f, 1.1228672266006469727f, -0.84215044975280761719f, 0.0000000000000000000f},
  {-1.9650176763534545898f, 0.0000000000000000000f, 0.84215044975280761719f, -0.28071680665016174316f, -0.28071680665016174316f, 0.28071680665016174316f, -0.28071680665016174316f},
  {0.0000000000000000000f, -0.28071680665016174316f, -0.28071680665016174316f, 0.0000000000000000000f, 0.0000000000000000000f, 0.28071680665016174316f, 0.0000000000000000000f},
  {0.84215044975280761719f, 0.28071680665016174316f, 0.28071680665016174316f, 0.0000000000000000000f, 0.56143361330032348633f, -1.4035840034484863281f, -0.28071680665016174316f},
  {0.0000000000000000000f, 0.28071680665016174316f, 0.0000000000000000000f, 0.0000000000000000000f, -0.28071680665016174316f, -0.28071680665016174316f, -0.28071680665016174316f},
  {-1.1228672266006469727f, -0.28071680665016174316f, 0.56143361330032348633f, 0.28071680665016174316f, -0.56143361330032348633f, 0.56143361330032348633f, 0.28071680665016174316f}
};
static const float quantized_tensor_Gemm_0_bias[6] = 
{-1.1228672266006469727f, 1.6843008995056152344f, 0.0000000000000000000f, 0.0000000000000000000f, -0.28071680665016174316f, 0.84215044975280761719f};
static const float quantized_tensor_Gemm_1_weight[3][6] = 
{
  {0.0000000000000000000f, 0.0000000000000000000f, 0.0000000000000000000f, 1.5507127046585083008f, 0.0000000000000000000f, 0.0000000000000000000f},
  {1.5507127046585083008f, -3.1014254093170166016f, 0.0000000000000000000f, 0.0000000000000000000f, 0.0000000000000000000f, -1.5507127046585083008f},
  {-1.5507127046585083008f, 1.5507127046585083008f, 0.0000000000000000000f, -1.5507127046585083008f, 0.0000000000000000000f, 1.5507127046585083008f}
};
static const float quantized_tensor_Gemm_1_bias[3] = 
{0.0000000000000000000f, 0.0000000000000000000f, 0.0000000000000000000f};
float quantized_tensor_onnx__Gemm_5[1][7];
float quantized_tensor_onnx__Gemm_7[1][6];

float quantized_tensor_input[1][6];


static inline void quantized_node_Flatten_0( const float quantized_tensor_onnx__Flatten_0[1][7], float quantized_tensor_onnx__Gemm_5[1][7] )
{
	/* Flatten*/
	float *input = (float*)quantized_tensor_onnx__Flatten_0;
	float *output = (float*)quantized_tensor_onnx__Gemm_5;
	for( uint32_t i=0; i<7; i++ )
		output[i] = input[i];

}

static inline void quantized_node_Gemm_1( const float quantized_tensor_onnx__Gemm_5[1][7], const float quantized_tensor_Gemm_0_weight[6][7], const float quantized_tensor_Gemm_0_bias[6], float quantized_tensor_input[1][6] )
{
	/* Gemm */
	/* alpha   = 1.0000000000000000000
	   beta    = 1.0000000000000000000
	   transA  = 0
	   transB  = 1
	 */
	const int M = 1;
	const int K = 7;
	const int N = 6;
	float (*A)[7]  = (float(*)[7])quantized_tensor_onnx__Gemm_5;
	float (*Y)[6]  = (float(*)[6])quantized_tensor_input;
	float alpha = 1.0000000000000000000;
	float beta = 1.0000000000000000000;
	float (*C)[6]  = (float(*)[6])quantized_tensor_Gemm_0_bias;
	for( uint32_t r=0; r<M; r++ )
		for( uint32_t c=0; c<N; c++ ) {
			float ABrc = 0;
			for( uint32_t i=0; i<K; i++ ) {
				float B = quantized_tensor_Gemm_0_weight[c][i];
				ABrc += A[r][i] * B;
			}
			float tmp = ABrc * alpha;
			tmp += C[0][c] * beta;
			Y[r][c] = tmp;
	}
}

static inline void quantized_node_Relu_2( const float quantized_tensor_input[1][6], float quantized_tensor_onnx__Gemm_7[1][6] )
{
	/*Relu*/
	float *X = (float*)quantized_tensor_input;
	float *Y = (float*)quantized_tensor_onnx__Gemm_7;
	for( uint32_t i=0; i<6; i++ )
		Y[i] = X[i] > 0 ? X[i] : 0;

}

static inline void quantized_node_Gemm_3( const float quantized_tensor_onnx__Gemm_7[1][6], const float quantized_tensor_Gemm_1_weight[3][6], const float quantized_tensor_Gemm_1_bias[3], float quantized_tensor_8[1][3] )
{
	/* Gemm */
	/* alpha   = 1.0000000000000000000
	   beta    = 1.0000000000000000000
	   transA  = 0
	   transB  = 1
	 */
	const int M = 1;
	const int K = 6;
	const int N = 3;
	float (*A)[6]  = (float(*)[6])quantized_tensor_onnx__Gemm_7;
	float (*Y)[3]  = (float(*)[3])quantized_tensor_8;
	float alpha = 1.0000000000000000000;
	float beta = 1.0000000000000000000;
	float (*C)[3]  = (float(*)[3])quantized_tensor_Gemm_1_bias;
	for( uint32_t r=0; r<M; r++ )
		for( uint32_t c=0; c<N; c++ ) {
			float ABrc = 0;
			for( uint32_t i=0; i<K; i++ ) {
				float B = quantized_tensor_Gemm_1_weight[c][i];
				ABrc += A[r][i] * B;
			}
			float tmp = ABrc * alpha;
			tmp += C[0][c] * beta;
			Y[r][c] = tmp;
	}
}


void quantized(const float quantized_tensor_onnx__Flatten_0[1][7], float quantized_tensor_8[1][3]) {
	quantized_node_Flatten_0( quantized_tensor_onnx__Flatten_0, quantized_tensor_onnx__Gemm_5);
	quantized_node_Gemm_1( quantized_tensor_onnx__Gemm_5, quantized_tensor_Gemm_0_weight, quantized_tensor_Gemm_0_bias, quantized_tensor_input);
	quantized_node_Relu_2( quantized_tensor_input, quantized_tensor_onnx__Gemm_7);
	quantized_node_Gemm_3( quantized_tensor_onnx__Gemm_7, quantized_tensor_Gemm_1_weight, quantized_tensor_Gemm_1_bias, quantized_tensor_8);
}

#endif // QUANTIZED_H

