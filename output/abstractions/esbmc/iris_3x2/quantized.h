
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

static const float quantized_tensor_Gemm_0_weight[3][4] = 
{
  {0.0000000000000000000f, 0.45287403464317321777f, -0.67931103706359863281f, -0.11321850866079330444f},
  {0.0000000000000000000f, 1.0189665555953979492f, -1.3586220741271972656f, -1.5850591659545898438f},
  {0.33965551853179931641f, -0.11321850866079330444f, -0.33965551853179931641f, 0.56609255075454711914f}
};
static const float quantized_tensor_Gemm_0_bias[3] = 
{0.33965551853179931641f, 1.6982775926589965820f, 0.11321850866079330444f};
static const float quantized_tensor_Gemm_1_weight[3][3] = 
{
  {0.0000000000000000000f, -0.79938423633575439453f, 0.0000000000000000000f},
  {0.79938423633575439453f, 2.3981528282165527344f, 0.0000000000000000000f},
  {0.0000000000000000000f, 0.0000000000000000000f, 0.79938423633575439453f}
};
static const float quantized_tensor_Gemm_1_bias[3] = 
{0.0000000000000000000f, 0.0000000000000000000f, -0.79938423633575439453f};
static const float quantized_tensor_Gemm_2_weight[3][3] = 
{
  {0.0000000000000000000f, 1.3603926897048950195f, 0.68019634485244750977f},
  {0.0000000000000000000f, 0.68019634485244750977f, -0.68019634485244750977f},
  {-0.68019634485244750977f, -2.7207853794097900391f, 0.0000000000000000000f}
};
static const float quantized_tensor_Gemm_2_bias[3] = 
{-2.0405890941619873047f, 0.68019634485244750977f, 2.0405890941619873047f};
float quantized_tensor_onnx__Gemm_7[1][4];
float quantized_tensor_onnx__Gemm_9[1][3];
float quantized_tensor_onnx__Gemm_11[1][3];

float quantized_tensor_input[1][3];
float quantized_tensor_input_3[1][3];


static inline void quantized_node_Flatten_0( const float quantized_tensor_onnx__Flatten_0[1][4], float quantized_tensor_onnx__Gemm_7[1][4] )
{
	/* Flatten*/
	float *input = (float*)quantized_tensor_onnx__Flatten_0;
	float *output = (float*)quantized_tensor_onnx__Gemm_7;
	for( uint32_t i=0; i<4; i++ )
		output[i] = input[i];

}

static inline void quantized_node_Gemm_1( const float quantized_tensor_onnx__Gemm_7[1][4], const float quantized_tensor_Gemm_0_weight[3][4], const float quantized_tensor_Gemm_0_bias[3], float quantized_tensor_input[1][3] )
{
	/* Gemm */
	/* alpha   = 1.0000000000000000000
	   beta    = 1.0000000000000000000
	   transA  = 0
	   transB  = 1
	 */
	const int M = 1;
	const int K = 4;
	const int N = 3;
	float (*A)[4]  = (float(*)[4])quantized_tensor_onnx__Gemm_7;
	float (*Y)[3]  = (float(*)[3])quantized_tensor_input;
	float alpha = 1.0000000000000000000;
	float beta = 1.0000000000000000000;
	float (*C)[3]  = (float(*)[3])quantized_tensor_Gemm_0_bias;
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

static inline void quantized_node_Relu_2( const float quantized_tensor_input[1][3], float quantized_tensor_onnx__Gemm_9[1][3] )
{
	/*Relu*/
	float *X = (float*)quantized_tensor_input;
	float *Y = (float*)quantized_tensor_onnx__Gemm_9;
	for( uint32_t i=0; i<3; i++ )
		Y[i] = X[i] > 0 ? X[i] : 0;

}

static inline void quantized_node_Gemm_3( const float quantized_tensor_onnx__Gemm_9[1][3], const float quantized_tensor_Gemm_1_weight[3][3], const float quantized_tensor_Gemm_1_bias[3], float quantized_tensor_input_3[1][3] )
{
	/* Gemm */
	/* alpha   = 1.0000000000000000000
	   beta    = 1.0000000000000000000
	   transA  = 0
	   transB  = 1
	 */
	const int M = 1;
	const int K = 3;
	const int N = 3;
	float (*A)[3]  = (float(*)[3])quantized_tensor_onnx__Gemm_9;
	float (*Y)[3]  = (float(*)[3])quantized_tensor_input_3;
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

static inline void quantized_node_Relu_4( const float quantized_tensor_input_3[1][3], float quantized_tensor_onnx__Gemm_11[1][3] )
{
	/*Relu*/
	float *X = (float*)quantized_tensor_input_3;
	float *Y = (float*)quantized_tensor_onnx__Gemm_11;
	for( uint32_t i=0; i<3; i++ )
		Y[i] = X[i] > 0 ? X[i] : 0;

}

static inline void quantized_node_Gemm_5( const float quantized_tensor_onnx__Gemm_11[1][3], const float quantized_tensor_Gemm_2_weight[3][3], const float quantized_tensor_Gemm_2_bias[3], float quantized_tensor_12[1][3] )
{
	/* Gemm */
	/* alpha   = 1.0000000000000000000
	   beta    = 1.0000000000000000000
	   transA  = 0
	   transB  = 1
	 */
	const int M = 1;
	const int K = 3;
	const int N = 3;
	float (*A)[3]  = (float(*)[3])quantized_tensor_onnx__Gemm_11;
	float (*Y)[3]  = (float(*)[3])quantized_tensor_12;
	float alpha = 1.0000000000000000000;
	float beta = 1.0000000000000000000;
	float (*C)[3]  = (float(*)[3])quantized_tensor_Gemm_2_bias;
	for( uint32_t r=0; r<M; r++ )
		for( uint32_t c=0; c<N; c++ ) {
			float ABrc = 0;
			for( uint32_t i=0; i<K; i++ ) {
				float B = quantized_tensor_Gemm_2_weight[c][i];
				ABrc += A[r][i] * B;
			}
			float tmp = ABrc * alpha;
			tmp += C[0][c] * beta;
			Y[r][c] = tmp;
	}
}


void quantized(const float quantized_tensor_onnx__Flatten_0[1][4], float quantized_tensor_12[1][3]) {
	quantized_node_Flatten_0( quantized_tensor_onnx__Flatten_0, quantized_tensor_onnx__Gemm_7);
	quantized_node_Gemm_1( quantized_tensor_onnx__Gemm_7, quantized_tensor_Gemm_0_weight, quantized_tensor_Gemm_0_bias, quantized_tensor_input);
	quantized_node_Relu_2( quantized_tensor_input, quantized_tensor_onnx__Gemm_9);
	quantized_node_Gemm_3( quantized_tensor_onnx__Gemm_9, quantized_tensor_Gemm_1_weight, quantized_tensor_Gemm_1_bias, quantized_tensor_input_3);
	quantized_node_Relu_4( quantized_tensor_input_3, quantized_tensor_onnx__Gemm_11);
	quantized_node_Gemm_5( quantized_tensor_onnx__Gemm_11, quantized_tensor_Gemm_2_weight, quantized_tensor_Gemm_2_bias, quantized_tensor_12);
}

#endif // QUANTIZED_H

