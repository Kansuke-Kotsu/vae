??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.02unknown8ξ	
~
conv2d/kernelVarHandleOp*
shared_nameconv2d/kernel*
_output_shapes
: *
dtype0*
shape: 
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
: 
n
conv2d/biasVarHandleOp*
_output_shapes
: *
shared_nameconv2d/bias*
dtype0*
shape: 
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
: 
?
conv2d_1/kernelVarHandleOp*
dtype0*
shape: @*
_output_shapes
: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
shared_nameconv2d_1/bias*
_output_shapes
: *
dtype0*
shape:@
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
u
dense/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense/kernel*
shape:	?
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?*
dtype0
l

dense/biasVarHandleOp*
shared_name
dense/bias*
dtype0*
shape:*
_output_shapes
: 
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shared_namedense_1/kernel*
shape:	?
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes
:	?
q
dense_1/biasVarHandleOp*
shared_namedense_1/bias*
_output_shapes
: *
dtype0*
shape:?
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
?
conv2d_transpose/kernelVarHandleOp*(
shared_nameconv2d_transpose/kernel*
_output_shapes
: *
shape:@ *
dtype0
?
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
:@ *
dtype0
?
conv2d_transpose/biasVarHandleOp*
dtype0*
shape:@*&
shared_nameconv2d_transpose/bias*
_output_shapes
: 
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:@*
dtype0
?
conv2d_transpose_1/kernelVarHandleOp**
shared_nameconv2d_transpose_1/kernel*
shape: @*
dtype0*
_output_shapes
: 
?
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*
dtype0*&
_output_shapes
: @
?
conv2d_transpose_1/biasVarHandleOp*
dtype0*
_output_shapes
: *(
shared_nameconv2d_transpose_1/bias*
shape: 

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
dtype0*
_output_shapes
: 
?
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
shape: **
shared_nameconv2d_transpose_2/kernel*
dtype0
?
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_2/biasVarHandleOp*
shape:*
dtype0*
_output_shapes
: *(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
dtype0*
_output_shapes
:

NoOpNoOp
?-
ConstConst"/device:CPU:0*?-
value?-B?, B?,
9
encoder
decoder
	keras_api

signatures
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
	layer_with_weights-2
	layer-4

regularization_losses
trainable_variables
	variables
	keras_api
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
regularization_losses
trainable_variables
	variables
	keras_api
 
 
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
R
(regularization_losses
)trainable_variables
*	variables
+	keras_api
h

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
 
*
0
1
"2
#3
,4
-5
*
0
1
"2
#3
,4
-5
?

regularization_losses
2layer_regularization_losses
3non_trainable_variables
trainable_variables

4layers
5metrics
	variables
R
6regularization_losses
7trainable_variables
8	variables
9	keras_api
h

:kernel
;bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
R
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
h

Dkernel
Ebias
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
h

Jkernel
Kbias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
h

Pkernel
Qbias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
 
8
:0
;1
D2
E3
J4
K5
P6
Q7
8
:0
;1
D2
E3
J4
K5
P6
Q7
?
regularization_losses
Vlayer_regularization_losses
Wnon_trainable_variables
trainable_variables

Xlayers
Ymetrics
	variables
 
 
 
?
regularization_losses
Zlayer_regularization_losses
[non_trainable_variables
trainable_variables

\layers
]metrics
	variables
a_
VARIABLE_VALUEconv2d/kernel>encoder/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv2d/bias<encoder/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
^layer_regularization_losses
_non_trainable_variables
trainable_variables

`layers
ametrics
 	variables
ca
VARIABLE_VALUEconv2d_1/kernel>encoder/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_1/bias<encoder/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
?
$regularization_losses
blayer_regularization_losses
cnon_trainable_variables
%trainable_variables

dlayers
emetrics
&	variables
 
 
 
?
(regularization_losses
flayer_regularization_losses
gnon_trainable_variables
)trainable_variables

hlayers
imetrics
*	variables
`^
VARIABLE_VALUEdense/kernel>encoder/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE
dense/bias<encoder/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
?
.regularization_losses
jlayer_regularization_losses
knon_trainable_variables
/trainable_variables

llayers
mmetrics
0	variables
 
 

0
1
2
	3
 
 
 
 
?
6regularization_losses
nlayer_regularization_losses
onon_trainable_variables
7trainable_variables

players
qmetrics
8	variables
b`
VARIABLE_VALUEdense_1/kernel>decoder/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEdense_1/bias<decoder/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
?
<regularization_losses
rlayer_regularization_losses
snon_trainable_variables
=trainable_variables

tlayers
umetrics
>	variables
 
 
 
?
@regularization_losses
vlayer_regularization_losses
wnon_trainable_variables
Atrainable_variables

xlayers
ymetrics
B	variables
ki
VARIABLE_VALUEconv2d_transpose/kernel>decoder/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEconv2d_transpose/bias<decoder/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1

D0
E1
?
Fregularization_losses
zlayer_regularization_losses
{non_trainable_variables
Gtrainable_variables

|layers
}metrics
H	variables
mk
VARIABLE_VALUEconv2d_transpose_1/kernel>decoder/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEconv2d_transpose_1/bias<decoder/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
?
Lregularization_losses
~layer_regularization_losses
non_trainable_variables
Mtrainable_variables
?layers
?metrics
N	variables
mk
VARIABLE_VALUEconv2d_transpose_2/kernel>decoder/layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEconv2d_transpose_2/bias<decoder/layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1

P0
Q1
?
Rregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
Strainable_variables
?layers
?metrics
T	variables
 
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 *
_output_shapes
: *
dtype0
O
saver_filenamePlaceholder*
_output_shapes
: *
shape: *
dtype0
?
StatefulPartitionedCallStatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOpConst*)
f$R"
 __inference__traced_save_3060222*
Tout
2*
Tin
2*.
_gradient_op_typePartitionedCall-3060223*
_output_shapes
: **
config_proto

CPU

GPU 2J 8
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/bias*.
_gradient_op_typePartitionedCall-3060278*,
f'R%
#__inference__traced_restore_3060277*
Tout
2*
_output_shapes
: *
Tin
2**
config_proto

CPU

GPU 2J 8??
?
?
B__inference_dense_layer_call_and_return_conditional_losses_3059277

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?

?
.__inference_sequential_1_layer_call_fn_3060093

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8**
config_proto

CPU

GPU 2J 8*
Tout
2*
Tin
2	*R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059650*A
_output_shapes/
-:+???????????????????????????*.
_gradient_op_typePartitionedCall-3059651?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_3059825

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: ?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:????????? *
strides
*
paddingVALID*
T0?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: @*
dtype0?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:?????????@?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????@*
T0f
flatten/Reshape/shapeConst*
valueB"???? 	  *
dtype0*
_output_shapes
:?
flatten/ReshapeReshapeconv2d_1/Relu:activations:0flatten/Reshape/shape:output:0*(
_output_shapes
:??????????*
T0?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
IdentityIdentitydense/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp: : : : : :& "
 
_user_specified_nameinputs: 
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059617

inputs*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_23
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_25
1conv2d_transpose_2_statefulpartitionedcall_args_15
1conv2d_transpose_2_statefulpartitionedcall_args_2
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputs&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*(
_output_shapes
:??????????*
Tout
2**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3059524*.
_gradient_op_typePartitionedCall-3059530?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*/
_output_shapes
:????????? **
config_proto

CPU

GPU 2J 8*
Tout
2*M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_3059555*
Tin
2*.
_gradient_op_typePartitionedCall-3059561?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_3059401*.
_gradient_op_typePartitionedCall-3059407**
config_proto

CPU

GPU 2J 8?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tout
2*X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_3059449*
Tin
2*A
_output_shapes/
-:+??????????????????????????? **
config_proto

CPU

GPU 2J 8*.
_gradient_op_typePartitionedCall-3059455?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:01conv2d_transpose_2_statefulpartitionedcall_args_11conv2d_transpose_2_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-3059502*
Tout
2*
Tin
2*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_3059496?
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
?
?
*__inference_conv2d_1_layer_call_fn_3059236

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3059225**
config_proto

CPU

GPU 2J 8*.
_gradient_op_typePartitionedCall-3059231?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?	
?
D__inference_dense_1_layer_call_and_return_conditional_losses_3060132

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?

`
D__inference_reshape_layer_call_and_return_conditional_losses_3060153

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:_
strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0_
strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: Q
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
dtype0*
value	B :*
_output_shapes
: Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:????????? `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_2_layer_call_fn_3059507

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*.
_gradient_op_typePartitionedCall-3059502*A
_output_shapes/
-:+???????????????????????????*X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_3059496*
Tout
2*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?;
?
#__inference__traced_restore_3060277
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias%
!assignvariableop_6_dense_1_kernel#
assignvariableop_7_dense_1_bias.
*assignvariableop_8_conv2d_transpose_kernel,
(assignvariableop_9_conv2d_transpose_bias1
-assignvariableop_10_conv2d_transpose_1_kernel/
+assignvariableop_11_conv2d_transpose_1_bias1
-assignvariableop_12_conv2d_transpose_2_kernel/
+assignvariableop_13_conv2d_transpose_2_bias
identity_15??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?B>encoder/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB<encoder/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB>encoder/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB<encoder/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB>encoder/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB<encoder/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB>decoder/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB<decoder/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB>decoder/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB<decoder/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB>decoder/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB<decoder/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB>decoder/layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB<decoder/layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:*
dtype0?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*/
value&B$B B B B B B B B B B B B B B *
dtype0?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes
2*L
_output_shapes:
8::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:z
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:~
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0*
_output_shapes
 *
dtype0N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0}
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0*
_output_shapes
 *
dtype0N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_conv2d_transpose_kernelIdentity_8:output:0*
_output_shapes
 *
dtype0N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp(assignvariableop_9_conv2d_transpose_biasIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_conv2d_transpose_1_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_conv2d_transpose_1_biasIdentity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0?
AssignVariableOp_12AssignVariableOp-assignvariableop_12_conv2d_transpose_2_kernelIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp+assignvariableop_13_conv2d_transpose_2_biasIdentity_13:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B ?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : 
?
?
(__inference_conv2d_layer_call_fn_3059211

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*A
_output_shapes/
-:+??????????????????????????? *
Tout
2*.
_gradient_op_typePartitionedCall-3059206*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_3059200*
Tin
2**
config_proto

CPU

GPU 2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+??????????????????????????? *
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
E
)__inference_flatten_layer_call_fn_3060104

inputs
identity?
PartitionedCallPartitionedCallinputs*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3059254*
Tin
2*(
_output_shapes
:??????????*
Tout
2**
config_proto

CPU

GPU 2J 8*.
_gradient_op_typePartitionedCall-3059260a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3059225

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
paddingVALID*
strides
*A
_output_shapes/
-:+???????????????????????????@?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????@*
T0j
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+???????????????????????????@*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+???????????????????????????@*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059650

inputs*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_23
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_25
1conv2d_transpose_2_statefulpartitionedcall_args_15
1conv2d_transpose_2_statefulpartitionedcall_args_2
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputs&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3059524*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8*.
_gradient_op_typePartitionedCall-3059530*(
_output_shapes
:???????????
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*.
_gradient_op_typePartitionedCall-3059561*/
_output_shapes
:????????? *
Tin
2*M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_3059555*
Tout
2?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+???????????????????????????@*.
_gradient_op_typePartitionedCall-3059407*
Tout
2*V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_3059401?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+??????????????????????????? *
Tout
2*X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_3059449*
Tin
2*.
_gradient_op_typePartitionedCall-3059455?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:01conv2d_transpose_2_statefulpartitionedcall_args_11conv2d_transpose_2_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-3059502*X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_3059496*
Tin
2*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*
Tout
2?
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
? 
?
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_3059449

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:_
strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0_
strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
shrink_axis_mask*
_output_shapes
: *
Index0_
strided_slice_2/stackConst*
valueB:*
_output_shapes
:*
dtype0a
strided_slice_2/stack_1Const*
_output_shapes
:*
valueB:*
dtype0a
strided_slice_2/stack_2Const*
valueB:*
_output_shapes
:*
dtype0?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: G
mul/yConst*
dtype0*
_output_shapes
: *
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
_output_shapes
: *
T0I
mul_1/yConst*
_output_shapes
: *
value	B :*
dtype0Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
value	B : *
dtype0y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
valueB: *
_output_shapes
:*
dtype0a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
paddingSAME*
strides
*A
_output_shapes/
-:+??????????????????????????? *
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: ?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+??????????????????????????? *
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? "
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_3059799

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: *
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: ?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? f
conv2d/ReluReluconv2d/BiasAdd:output:0*/
_output_shapes
:????????? *
T0?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????@*
strides
*
T0*
paddingVALID?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????@*
T0j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@f
flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"???? 	  ?
flatten/ReshapeReshapeconv2d_1/Relu:activations:0flatten/Reshape/shape:output:0*(
_output_shapes
:??????????*
T0?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentitydense/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp: : : : : :& "
 
_user_specified_nameinputs: 
?'
?
 __inference__traced_save_3060222
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_5806b920300144d18cd9a22ee1b17292/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
_output_shapes
: *
NL

num_shardsConst*
_output_shapes
: *
value	B :*
dtype0f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*?
value?B?B>encoder/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB<encoder/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB>encoder/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB<encoder/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB>encoder/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB<encoder/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB>decoder/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB<decoder/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB>decoder/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB<decoder/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB>decoder/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB<decoder/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB>decoder/layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB<decoder/layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*/
value&B$B B B B B B B B B B B B B B *
dtype0?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop"/device:CPU:0*
dtypes
2*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
value	B :*
_output_shapes
: ?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B ?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:	?::	?:?:@ :@: @: : :: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : 
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059578
input_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_23
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_25
1conv2d_transpose_2_statefulpartitionedcall_args_15
1conv2d_transpose_2_statefulpartitionedcall_args_2
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_2&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tout
2*
Tin
2*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3059524*.
_gradient_op_typePartitionedCall-3059530*(
_output_shapes
:???????????
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:????????? *
Tin
2*.
_gradient_op_typePartitionedCall-3059561*M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_3059555?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tout
2*V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_3059401*.
_gradient_op_typePartitionedCall-3059407*
Tin
2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+???????????????????????????@?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*A
_output_shapes/
-:+??????????????????????????? *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_3059449**
config_proto

CPU

GPU 2J 8*.
_gradient_op_typePartitionedCall-3059455?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:01conv2d_transpose_2_statefulpartitionedcall_args_11conv2d_transpose_2_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_3059496*A
_output_shapes/
-:+???????????????????????????*.
_gradient_op_typePartitionedCall-3059502?
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_2: : : : : : : : 
?	
?
,__inference_sequential_layer_call_fn_3059836

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_3059328*
Tin
	2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*.
_gradient_op_typePartitionedCall-3059329*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_3059295
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*/
_output_shapes
:????????? *.
_gradient_op_typePartitionedCall-3059206*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_3059200**
config_proto

CPU

GPU 2J 8?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3059225*.
_gradient_op_typePartitionedCall-3059231*/
_output_shapes
:?????????@?
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3059254*
Tin
2*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-3059260*
Tout
2?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*.
_gradient_op_typePartitionedCall-3059283*'
_output_shapes
:?????????*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3059277?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : 
?	
?
D__inference_dense_1_layer_call_and_return_conditional_losses_3059524

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_3059496

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
valueB:*
dtype0a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
valueB:*
_output_shapes
:*
dtype0a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
shrink_axis_mask*
_output_shapes
: *
T0G
mul/yConst*
_output_shapes
: *
value	B :*
dtype0U
mulMulstrided_slice_1:output:0mul/y:output:0*
_output_shapes
: *
T0I
mul_1/yConst*
dtype0*
value	B :*
_output_shapes
: Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
value	B :*
_output_shapes
: *
dtype0y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
_output_shapes
:*
T0*
N_
strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB:*
_output_shapes
:*
dtype0a
strided_slice_3/stack_2Const*
valueB:*
_output_shapes
:*
dtype0?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: ?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+????????????????????????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059597
input_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_23
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_25
1conv2d_transpose_2_statefulpartitionedcall_args_15
1conv2d_transpose_2_statefulpartitionedcall_args_2
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_2&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-3059530**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3059524?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tout
2*M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_3059555*.
_gradient_op_typePartitionedCall-3059561*
Tin
2*/
_output_shapes
:????????? **
config_proto

CPU

GPU 2J 8?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*A
_output_shapes/
-:+???????????????????????????@*.
_gradient_op_typePartitionedCall-3059407*
Tin
2**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_3059401*
Tout
2?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-3059455*
Tout
2*
Tin
2**
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_3059449*A
_output_shapes/
-:+??????????????????????????? ?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:01conv2d_transpose_2_statefulpartitionedcall_args_11conv2d_transpose_2_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-3059502*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8*
Tout
2*X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_3059496*
Tin
2?
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall: : : : : : : :' #
!
_user_specified_name	input_2: 
?
E
)__inference_reshape_layer_call_fn_3060158

inputs
identity?
PartitionedCallPartitionedCallinputs*/
_output_shapes
:????????? *
Tin
2*M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_3059555**
config_proto

CPU

GPU 2J 8*
Tout
2*.
_gradient_op_typePartitionedCall-3059561h
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:????????? *
T0"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_layer_call_fn_3059412

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+???????????????????????????@*.
_gradient_op_typePartitionedCall-3059407*
Tout
2*V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_3059401?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
4__inference_conv2d_transpose_1_layer_call_fn_3059460

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+??????????????????????????? *.
_gradient_op_typePartitionedCall-3059455*X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_3059449*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? "
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
C__inference_conv2d_layer_call_and_return_conditional_losses_3059200

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: ?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
T0*
strides
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: ?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+??????????????????????????? *
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+??????????????????????????? *
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
??
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059957

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?z
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_1/ReluReludense_1/BiasAdd:output:0*(
_output_shapes
:??????????*
T0W
reshape/ShapeShapedense_1/Relu:activations:0*
_output_shapes
:*
T0e
reshape/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: g
reshape/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
_output_shapes
: *
Index0Y
reshape/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :Y
reshape/Reshape/shape/2Const*
dtype0*
value	B :*
_output_shapes
: Y
reshape/Reshape/shape/3Const*
value	B : *
dtype0*
_output_shapes
: ?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
_output_shapes
:*
T0*
N?
reshape/ReshapeReshapedense_1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? ^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0p
&conv2d_transpose/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: p
&conv2d_transpose/strided_slice_1/stackConst*
valueB:*
_output_shapes
:*
dtype0r
(conv2d_transpose/strided_slice_1/stack_1Const*
valueB:*
_output_shapes
:*
dtype0r
(conv2d_transpose/strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
_output_shapes
: *
shrink_axis_mask*
T0p
&conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
valueB:*
dtype0r
(conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0X
conv2d_transpose/mul/yConst*
_output_shapes
: *
value	B :*
dtype0?
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
_output_shapes
: *
T0Z
conv2d_transpose/mul_1/yConst*
_output_shapes
: *
value	B :*
dtype0?
conv2d_transpose/mul_1Mul)conv2d_transpose/strided_slice_2:output:0!conv2d_transpose/mul_1/y:output:0*
_output_shapes
: *
T0Z
conv2d_transpose/stack/3Const*
value	B :@*
dtype0*
_output_shapes
: ?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/mul:z:0conv2d_transpose/mul_1:z:0!conv2d_transpose/stack/3:output:0*
_output_shapes
:*
T0*
Np
&conv2d_transpose/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:r
(conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_3StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_3/stack:output:01conv2d_transpose/strided_slice_3/stack_1:output:01conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
_output_shapes
: *
shrink_axis_mask*
T0?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@ ?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:?????????@?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@z
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
_output_shapes
:*
T0p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:r
(conv2d_transpose_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0r
(conv2d_transpose_1/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0r
(conv2d_transpose_1/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:t
*conv2d_transpose_1/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:t
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0Z
conv2d_transpose_1/mul/yConst*
value	B :*
_output_shapes
: *
dtype0?
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
_output_shapes
: *
T0\
conv2d_transpose_1/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: ?
conv2d_transpose_1/mul_1Mul+conv2d_transpose_1/strided_slice_2:output:0#conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: \
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/mul:z:0conv2d_transpose_1/mul_1:z:0#conv2d_transpose_1/stack/3:output:0*
N*
_output_shapes
:*
T0r
(conv2d_transpose_1/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:t
*conv2d_transpose_1/strided_slice_3/stack_1Const*
dtype0*
valueB:*
_output_shapes
:t
*conv2d_transpose_1/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
"conv2d_transpose_1/strided_slice_3StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_3/stack:output:03conv2d_transpose_1/strided_slice_3/stack_1:output:03conv2d_transpose_1/strided_slice_3/stack_2:output:0*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: ?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: @*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
paddingSAME*
T0*/
_output_shapes
:????????? *
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: ?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ~
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? m
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
_output_shapes
:*
T0p
&conv2d_transpose_2/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:r
(conv2d_transpose_2/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0r
(conv2d_transpose_2/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
valueB:*
_output_shapes
:*
dtype0t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/Shape:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
_output_shapes
: *
T0*
shrink_axis_maskr
(conv2d_transpose_2/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:t
*conv2d_transpose_2/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:t
*conv2d_transpose_2/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
"conv2d_transpose_2/strided_slice_2StridedSlice!conv2d_transpose_2/Shape:output:01conv2d_transpose_2/strided_slice_2/stack:output:03conv2d_transpose_2/strided_slice_2/stack_1:output:03conv2d_transpose_2/strided_slice_2/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskZ
conv2d_transpose_2/mul/yConst*
dtype0*
_output_shapes
: *
value	B :?
conv2d_transpose_2/mulMul+conv2d_transpose_2/strided_slice_1:output:0!conv2d_transpose_2/mul/y:output:0*
_output_shapes
: *
T0\
conv2d_transpose_2/mul_1/yConst*
dtype0*
value	B :*
_output_shapes
: ?
conv2d_transpose_2/mul_1Mul+conv2d_transpose_2/strided_slice_2:output:0#conv2d_transpose_2/mul_1/y:output:0*
T0*
_output_shapes
: \
conv2d_transpose_2/stack/3Const*
dtype0*
value	B :*
_output_shapes
: ?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0conv2d_transpose_2/mul:z:0conv2d_transpose_2/mul_1:z:0#conv2d_transpose_2/stack/3:output:0*
N*
_output_shapes
:*
T0r
(conv2d_transpose_2/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB: t
*conv2d_transpose_2/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:t
*conv2d_transpose_2/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:?
"conv2d_transpose_2/strided_slice_3StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_3/stack:output:03conv2d_transpose_2/strided_slice_3/stack_1:output:03conv2d_transpose_2/strided_slice_3/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: ?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: *
dtype0?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
paddingSAME*/
_output_shapes
:?????????*
strides
*
T0?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????*
T0?
IdentityIdentity#conv2d_transpose_2/BiasAdd:output:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*/
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : : : : : 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_3059356

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2*/
_output_shapes
:????????? **
config_proto

CPU

GPU 2J 8*.
_gradient_op_typePartitionedCall-3059206*
Tin
2*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_3059200?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*/
_output_shapes
:?????????@*.
_gradient_op_typePartitionedCall-3059231**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3059225*
Tout
2*
Tin
2?
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tout
2*(
_output_shapes
:??????????*
Tin
2*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3059254*.
_gradient_op_typePartitionedCall-3059260?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3059277**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????*
Tout
2*.
_gradient_op_typePartitionedCall-3059283*
Tin
2?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
?
?
'__inference_dense_layer_call_fn_3060121

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2*.
_gradient_op_typePartitionedCall-3059283*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3059277*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_3059328

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_3059200*
Tin
2*
Tout
2*.
_gradient_op_typePartitionedCall-3059206*/
_output_shapes
:????????? ?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-3059231*
Tout
2*
Tin
2*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3059225**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:?????????@?
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-3059260**
config_proto

CPU

GPU 2J 8*
Tout
2*
Tin
2*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3059254?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3059277*'
_output_shapes
:?????????*
Tout
2*.
_gradient_op_typePartitionedCall-3059283*
Tin
2?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: : : 
??
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3060067

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?z
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????W
reshape/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:g
reshape/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:g
reshape/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
T0*
shrink_axis_mask*
_output_shapes
: *
Index0Y
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
value	B :*
dtype0*
_output_shapes
: Y
reshape/Reshape/shape/3Const*
dtype0*
value	B : *
_output_shapes
: ?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
_output_shapes
:*
N*
T0?
reshape/ReshapeReshapedense_1/Relu:activations:0reshape/Reshape/shape:output:0*/
_output_shapes
:????????? *
T0^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0p
&conv2d_transpose/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
&conv2d_transpose/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:r
(conv2d_transpose/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: p
&conv2d_transpose/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:r
(conv2d_transpose/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:r
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
shrink_axis_mask*
Index0*
_output_shapes
: *
T0X
conv2d_transpose/mul/yConst*
dtype0*
_output_shapes
: *
value	B :?
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
_output_shapes
: *
T0Z
conv2d_transpose/mul_1/yConst*
value	B :*
_output_shapes
: *
dtype0?
conv2d_transpose/mul_1Mul)conv2d_transpose/strided_slice_2:output:0!conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/mul:z:0conv2d_transpose/mul_1:z:0!conv2d_transpose/stack/3:output:0*
N*
_output_shapes
:*
T0p
&conv2d_transpose/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB: r
(conv2d_transpose/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:r
(conv2d_transpose/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:?
 conv2d_transpose/strided_slice_3StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_3/stack:output:01conv2d_transpose/strided_slice_3/stack_1:output:01conv2d_transpose/strided_slice_3/stack_2:output:0*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@ ?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
strides
*
paddingSAME*/
_output_shapes
:?????????@*
T0?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????@*
T0z
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*/
_output_shapes
:?????????@*
T0k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:r
(conv2d_transpose_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:r
(conv2d_transpose_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0r
(conv2d_transpose_1/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
_output_shapes
: *
shrink_axis_mask*
T0r
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
valueB:*
dtype0t
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
valueB:*
dtype0t
*conv2d_transpose_1/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:?
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: Z
conv2d_transpose_1/mul/yConst*
dtype0*
value	B :*
_output_shapes
: ?
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
_output_shapes
: *
T0\
conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
value	B :*
dtype0?
conv2d_transpose_1/mul_1Mul+conv2d_transpose_1/strided_slice_2:output:0#conv2d_transpose_1/mul_1/y:output:0*
_output_shapes
: *
T0\
conv2d_transpose_1/stack/3Const*
dtype0*
value	B : *
_output_shapes
: ?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/mul:z:0conv2d_transpose_1/mul_1:z:0#conv2d_transpose_1/stack/3:output:0*
T0*
_output_shapes
:*
Nr
(conv2d_transpose_1/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:t
*conv2d_transpose_1/strided_slice_3/stack_1Const*
dtype0*
valueB:*
_output_shapes
:t
*conv2d_transpose_1/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
"conv2d_transpose_1/strided_slice_3StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_3/stack:output:03conv2d_transpose_1/strided_slice_3/stack_1:output:03conv2d_transpose_1/strided_slice_3/stack_2:output:0*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: ?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
strides
*
paddingSAME*/
_output_shapes
:????????? *
T0?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: ?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:????????? *
T0~
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? m
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:r
(conv2d_transpose_2/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0r
(conv2d_transpose_2/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
shrink_axis_mask*
_output_shapes
: *
T0r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/Shape:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0r
(conv2d_transpose_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:?
"conv2d_transpose_2/strided_slice_2StridedSlice!conv2d_transpose_2/Shape:output:01conv2d_transpose_2/strided_slice_2/stack:output:03conv2d_transpose_2/strided_slice_2/stack_1:output:03conv2d_transpose_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: Z
conv2d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_2/mulMul+conv2d_transpose_2/strided_slice_1:output:0!conv2d_transpose_2/mul/y:output:0*
_output_shapes
: *
T0\
conv2d_transpose_2/mul_1/yConst*
_output_shapes
: *
value	B :*
dtype0?
conv2d_transpose_2/mul_1Mul+conv2d_transpose_2/strided_slice_2:output:0#conv2d_transpose_2/mul_1/y:output:0*
T0*
_output_shapes
: \
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
value	B :*
dtype0?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0conv2d_transpose_2/mul:z:0conv2d_transpose_2/mul_1:z:0#conv2d_transpose_2/stack/3:output:0*
T0*
N*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_3/stackConst*
valueB: *
_output_shapes
:*
dtype0t
*conv2d_transpose_2/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:t
*conv2d_transpose_2/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
"conv2d_transpose_2/strided_slice_3StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_3/stack:output:03conv2d_transpose_2/strided_slice_3/stack_1:output:03conv2d_transpose_2/strided_slice_3/stack_2:output:0*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: ?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
paddingSAME*
strides
*
T0*/
_output_shapes
:??????????
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
IdentityIdentity#conv2d_transpose_2/BiasAdd:output:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*/
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : : : : : 
?

?
.__inference_sequential_1_layer_call_fn_3059662
input_2"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tout
2*
Tin
2	*.
_gradient_op_typePartitionedCall-3059651*R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059650*A
_output_shapes/
-:+???????????????????????????**
config_proto

CPU

GPU 2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_2: : : : : : : : 
?	
?
,__inference_sequential_layer_call_fn_3059338
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*.
_gradient_op_typePartitionedCall-3059329*
Tin
	2*'
_output_shapes
:?????????*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_3059328*
Tout
2**
config_proto

CPU

GPU 2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : 
?
?
B__inference_dense_layer_call_and_return_conditional_losses_3060114

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
,__inference_sequential_layer_call_fn_3059366
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*'
_output_shapes
:?????????*.
_gradient_op_typePartitionedCall-3059357*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_3059356*
Tout
2*
Tin
	2**
config_proto

CPU

GPU 2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_3059311
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_3059200**
config_proto

CPU

GPU 2J 8*.
_gradient_op_typePartitionedCall-3059206*/
_output_shapes
:????????? ?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3059225*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2*/
_output_shapes
:?????????@*.
_gradient_op_typePartitionedCall-3059231?
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tout
2*
Tin
2*.
_gradient_op_typePartitionedCall-3059260*(
_output_shapes
:??????????*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3059254?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tout
2*.
_gradient_op_typePartitionedCall-3059283**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3059277*
Tin
2?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_1: 
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_3060099

inputs
identity^
Reshape/shapeConst*
valueB"???? 	  *
_output_shapes
:*
dtype0e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:??????????*
T0Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
? 
?
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_3059401

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
T0*
_output_shapes
: *
Index0_
strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
_output_shapes
:*
dtype0a
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0G
mul/yConst*
dtype0*
value	B :*
_output_shapes
: U
mulMulstrided_slice_1:output:0mul/y:output:0*
_output_shapes
: *
T0I
mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
dtype0*
value	B :@*
_output_shapes
: y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
_output_shapes
:*
T0_
strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: ?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@ ?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????@*
T0j
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+???????????????????????????@*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
)__inference_dense_1_layer_call_fn_3060139

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-3059530*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3059524*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:???????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?

`
D__inference_reshape_layer_call_and_return_conditional_losses_3059555

inputs
identity;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: Q
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
value	B :*
dtype0Q
Reshape/shape/3Const*
_output_shapes
: *
value	B : *
dtype0?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
_output_shapes
:*
T0l
ReshapeReshapeinputsReshape/shape:output:0*/
_output_shapes
:????????? *
T0`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?

?
.__inference_sequential_1_layer_call_fn_3059629
input_2"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	**
config_proto

CPU

GPU 2J 8*
Tout
2*R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059617*A
_output_shapes/
-:+???????????????????????????*.
_gradient_op_typePartitionedCall-3059618?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_2: : : 
?	
?
,__inference_sequential_layer_call_fn_3059847

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tout
2*
Tin
	2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_3059356*.
_gradient_op_typePartitionedCall-3059357?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
?

?
.__inference_sequential_1_layer_call_fn_3060080

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tout
2*.
_gradient_op_typePartitionedCall-3059618**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+???????????????????????????*R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059617*
Tin
2	?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : : : : : 
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_3059254

inputs
identity^
Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"???? 	  e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:??????????*
T0Y
IdentityIdentityReshape:output:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
??
?
__inference_sample_3059773
eps7
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource
identity??4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?+sequential_1/dense_1/BiasAdd/ReadVariableOp?*sequential_1/dense_1/MatMul/ReadVariableOp?
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?*
dtype0?
sequential_1/dense_1/MatMulMatMuleps2sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
_output_shapes
:	?*
T0r
sequential_1/dense_1/ReluRelu%sequential_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	?k
sequential_1/reshape/ShapeConst*
valueB"      *
_output_shapes
:*
dtype0r
(sequential_1/reshape/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:t
*sequential_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_1/reshape/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
"sequential_1/reshape/strided_sliceStridedSlice#sequential_1/reshape/Shape:output:01sequential_1/reshape/strided_slice/stack:output:03sequential_1/reshape/strided_slice/stack_1:output:03sequential_1/reshape/strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskf
$sequential_1/reshape/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0f
$sequential_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :f
$sequential_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ?
"sequential_1/reshape/Reshape/shapePack+sequential_1/reshape/strided_slice:output:0-sequential_1/reshape/Reshape/shape/1:output:0-sequential_1/reshape/Reshape/shape/2:output:0-sequential_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
sequential_1/reshape/ReshapeReshape'sequential_1/dense_1/Relu:activations:0+sequential_1/reshape/Reshape/shape:output:0*&
_output_shapes
: *
T0|
#sequential_1/conv2d_transpose/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"             {
1sequential_1/conv2d_transpose/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:}
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0}
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0}
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/Shape:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
shrink_axis_mask*
Index0*
_output_shapes
: *
T0}
3sequential_1/conv2d_transpose/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
5sequential_1/conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
5sequential_1/conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-sequential_1/conv2d_transpose/strided_slice_2StridedSlice,sequential_1/conv2d_transpose/Shape:output:0<sequential_1/conv2d_transpose/strided_slice_2/stack:output:0>sequential_1/conv2d_transpose/strided_slice_2/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_2/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0e
#sequential_1/conv2d_transpose/mul/yConst*
dtype0*
value	B :*
_output_shapes
: ?
!sequential_1/conv2d_transpose/mulMul6sequential_1/conv2d_transpose/strided_slice_1:output:0,sequential_1/conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: g
%sequential_1/conv2d_transpose/mul_1/yConst*
value	B :*
_output_shapes
: *
dtype0?
#sequential_1/conv2d_transpose/mul_1Mul6sequential_1/conv2d_transpose/strided_slice_2:output:0.sequential_1/conv2d_transpose/mul_1/y:output:0*
_output_shapes
: *
T0g
%sequential_1/conv2d_transpose/stack/3Const*
value	B :@*
_output_shapes
: *
dtype0?
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0%sequential_1/conv2d_transpose/mul:z:0'sequential_1/conv2d_transpose/mul_1:z:0.sequential_1/conv2d_transpose/stack/3:output:0*
_output_shapes
:*
T0*
N}
3sequential_1/conv2d_transpose/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
5sequential_1/conv2d_transpose/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
5sequential_1/conv2d_transpose/strided_slice_3/stack_2Const*
valueB:*
_output_shapes
:*
dtype0?
-sequential_1/conv2d_transpose/strided_slice_3StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_3/stack:output:0>sequential_1/conv2d_transpose/strided_slice_3/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_3/stack_2:output:0*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:@ *
dtype0?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/reshape/Reshape:output:0*
paddingSAME*
T0*
strides
*&
_output_shapes
:@?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@?
"sequential_1/conv2d_transpose/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*&
_output_shapes
:@*
T0~
%sequential_1/conv2d_transpose_1/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   }
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: 
5sequential_1/conv2d_transpose_1/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:?
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:?
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
/sequential_1/conv2d_transpose_1/strided_slice_2StridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0>sequential_1/conv2d_transpose_1/strided_slice_2/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_2/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_2/stack_2:output:0*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0g
%sequential_1/conv2d_transpose_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :?
#sequential_1/conv2d_transpose_1/mulMul8sequential_1/conv2d_transpose_1/strided_slice_1:output:0.sequential_1/conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: i
'sequential_1/conv2d_transpose_1/mul_1/yConst*
dtype0*
_output_shapes
: *
value	B :?
%sequential_1/conv2d_transpose_1/mul_1Mul8sequential_1/conv2d_transpose_1/strided_slice_2:output:00sequential_1/conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: i
'sequential_1/conv2d_transpose_1/stack/3Const*
value	B : *
dtype0*
_output_shapes
: ?
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:0'sequential_1/conv2d_transpose_1/mul:z:0)sequential_1/conv2d_transpose_1/mul_1:z:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:
5sequential_1/conv2d_transpose_1/strided_slice_3/stackConst*
valueB: *
_output_shapes
:*
dtype0?
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
valueB:*
dtype0?
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_1/conv2d_transpose_1/strided_slice_3StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_3/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_3/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: ?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: @*
dtype0?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:00sequential_1/conv2d_transpose/Relu:activations:0*&
_output_shapes
: *
paddingSAME*
T0*
strides
?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
$sequential_1/conv2d_transpose_1/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*&
_output_shapes
: ~
%sequential_1/conv2d_transpose_2/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"             }
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
valueB:*
dtype0?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
valueB:*
_output_shapes
:*
dtype0?
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
5sequential_1/conv2d_transpose_2/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:?
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:?
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_1/conv2d_transpose_2/strided_slice_2StridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0>sequential_1/conv2d_transpose_2/strided_slice_2/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_2/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_2/stack_2:output:0*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0g
%sequential_1/conv2d_transpose_2/mul/yConst*
_output_shapes
: *
value	B :*
dtype0?
#sequential_1/conv2d_transpose_2/mulMul8sequential_1/conv2d_transpose_2/strided_slice_1:output:0.sequential_1/conv2d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: i
'sequential_1/conv2d_transpose_2/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: ?
%sequential_1/conv2d_transpose_2/mul_1Mul8sequential_1/conv2d_transpose_2/strided_slice_2:output:00sequential_1/conv2d_transpose_2/mul_1/y:output:0*
T0*
_output_shapes
: i
'sequential_1/conv2d_transpose_2/stack/3Const*
dtype0*
value	B :*
_output_shapes
: ?
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:0'sequential_1/conv2d_transpose_2/mul:z:0)sequential_1/conv2d_transpose_2/mul_1:z:00sequential_1/conv2d_transpose_2/stack/3:output:0*
T0*
_output_shapes
:*
N
5sequential_1/conv2d_transpose_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:?
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
/sequential_1/conv2d_transpose_2/strided_slice_3StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_3/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_3/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_3/stack_2:output:0*
shrink_axis_mask*
Index0*
_output_shapes
: *
T0?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: *
dtype0?
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:02sequential_1/conv2d_transpose_1/Relu:activations:0*
T0*
paddingSAME*
strides
*&
_output_shapes
:?
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:u
SigmoidSigmoid0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*&
_output_shapes
:?
IdentityIdentitySigmoid:y:05^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*
T0*&
_output_shapes
:"
identityIdentity:output:0*=
_input_shapes,
*:::::::::2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2?
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp:# 

_user_specified_nameeps: : : : : : : : "wJ
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:??
?
encoder
decoder
	keras_api

signatures
?sample"?
_tf_keras_model?{"class_name": "CVAE", "name": "cvae", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "CVAE"}}
? 
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
	layer_with_weights-2
	layer-4

regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": [null, 28, 28, 1]}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": [null, 28, 28, 1]}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?+
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?(
_tf_keras_sequential?({"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1568, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": [null, 2]}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [7, 7, 32]}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1568, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": [null, 2]}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [7, 7, 32]}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}}}
"
_generic_user_object
"
signature_map
?
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "input_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 28, 28, 1], "config": {"batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "sparse": false, "name": "input_1"}}
?

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
?

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
?
(regularization_losses
)trainable_variables
*	variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}}
 "
trackable_list_wrapper
J
0
1
"2
#3
,4
-5"
trackable_list_wrapper
J
0
1
"2
#3
,4
-5"
trackable_list_wrapper
?

regularization_losses
2layer_regularization_losses
3non_trainable_variables
trainable_variables

4layers
5metrics
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
6regularization_losses
7trainable_variables
8	variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "input_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 2], "config": {"batch_input_shape": [null, 2], "dtype": "float32", "sparse": false, "name": "input_2"}}
?

:kernel
;bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1568, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}}
?
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [7, 7, 32]}}
?

Dkernel
Ebias
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
?

Jkernel
Kbias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
?

Pkernel
Qbias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
 "
trackable_list_wrapper
X
:0
;1
D2
E3
J4
K5
P6
Q7"
trackable_list_wrapper
X
:0
;1
D2
E3
J4
K5
P6
Q7"
trackable_list_wrapper
?
regularization_losses
Vlayer_regularization_losses
Wnon_trainable_variables
trainable_variables

Xlayers
Ymetrics
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Zlayer_regularization_losses
[non_trainable_variables
trainable_variables

\layers
]metrics
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':% 2conv2d/kernel
: 2conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
^layer_regularization_losses
_non_trainable_variables
trainable_variables

`layers
ametrics
 	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_1/kernel
:@2conv2d_1/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
$regularization_losses
blayer_regularization_losses
cnon_trainable_variables
%trainable_variables

dlayers
emetrics
&	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
(regularization_losses
flayer_regularization_losses
gnon_trainable_variables
)trainable_variables

hlayers
imetrics
*	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
.regularization_losses
jlayer_regularization_losses
knon_trainable_variables
/trainable_variables

llayers
mmetrics
0	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
6regularization_losses
nlayer_regularization_losses
onon_trainable_variables
7trainable_variables

players
qmetrics
8	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_1/kernel
:?2dense_1/bias
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
<regularization_losses
rlayer_regularization_losses
snon_trainable_variables
=trainable_variables

tlayers
umetrics
>	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
@regularization_losses
vlayer_regularization_losses
wnon_trainable_variables
Atrainable_variables

xlayers
ymetrics
B	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/@ 2conv2d_transpose/kernel
#:!@2conv2d_transpose/bias
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
?
Fregularization_losses
zlayer_regularization_losses
{non_trainable_variables
Gtrainable_variables

|layers
}metrics
H	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1 @2conv2d_transpose_1/kernel
%:# 2conv2d_transpose_1/bias
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
Lregularization_losses
~layer_regularization_losses
non_trainable_variables
Mtrainable_variables
?layers
?metrics
N	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1 2conv2d_transpose_2/kernel
%:#2conv2d_transpose_2/bias
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
Rregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
Strainable_variables
?layers
?metrics
T	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
__inference_sample_3059773?
???
FullArgSpec
args?
jself
jeps
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_sequential_layer_call_fn_3059366
,__inference_sequential_layer_call_fn_3059836
,__inference_sequential_layer_call_fn_3059338
,__inference_sequential_layer_call_fn_3059847?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_layer_call_and_return_conditional_losses_3059825
G__inference_sequential_layer_call_and_return_conditional_losses_3059799
G__inference_sequential_layer_call_and_return_conditional_losses_3059311
G__inference_sequential_layer_call_and_return_conditional_losses_3059295?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_sequential_1_layer_call_fn_3060093
.__inference_sequential_1_layer_call_fn_3059629
.__inference_sequential_1_layer_call_fn_3060080
.__inference_sequential_1_layer_call_fn_3059662?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3060067
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059957
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059597
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059578?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
(__inference_conv2d_layer_call_fn_3059211?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
C__inference_conv2d_layer_call_and_return_conditional_losses_3059200?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
*__inference_conv2d_1_layer_call_fn_3059236?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3059225?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
)__inference_flatten_layer_call_fn_3060104?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_layer_call_and_return_conditional_losses_3060099?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_layer_call_fn_3060121?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_layer_call_and_return_conditional_losses_3060114?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
)__inference_dense_1_layer_call_fn_3060139?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_1_layer_call_and_return_conditional_losses_3060132?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_layer_call_fn_3060158?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_layer_call_and_return_conditional_losses_3060153?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_conv2d_transpose_layer_call_fn_3059412?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_3059401?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
4__inference_conv2d_transpose_1_layer_call_fn_3059460?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_3059449?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
4__inference_conv2d_transpose_2_layer_call_fn_3059507?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_3059496?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059578?:;DEJKPQ8?5
.?+
!?
input_2?????????
p

 
? "??<
5?2
0+???????????????????????????
? {
'__inference_dense_layer_call_fn_3060121P,-0?-
&?#
!?
inputs??????????
? "???????????
)__inference_flatten_layer_call_fn_3060104T7?4
-?*
(?%
inputs?????????@
? "????????????
I__inference_sequential_1_layer_call_and_return_conditional_losses_3060067r:;DEJKPQ7?4
-?*
 ?
inputs?????????
p 

 
? "-?*
#? 
0?????????
? ?
,__inference_sequential_layer_call_fn_3059366d"#,-@?=
6?3
)?&
input_1?????????
p 

 
? "???????????
*__inference_conv2d_1_layer_call_fn_3059236?"#I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
.__inference_sequential_1_layer_call_fn_3060093w:;DEJKPQ7?4
-?*
 ?
inputs?????????
p 

 
? "2?/+????????????????????????????
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059597?:;DEJKPQ8?5
.?+
!?
input_2?????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_3059311q"#,-@?=
6?3
)?&
input_1?????????
p 

 
? "%?"
?
0?????????
? ?
4__inference_conv2d_transpose_2_layer_call_fn_3059507?PQI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
G__inference_sequential_layer_call_and_return_conditional_losses_3059825p"#,-??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
4__inference_conv2d_transpose_1_layer_call_fn_3059460?JKI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
D__inference_dense_1_layer_call_and_return_conditional_losses_3060132]:;/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? ?
2__inference_conv2d_transpose_layer_call_fn_3059412?DEI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
.__inference_sequential_1_layer_call_fn_3060080w:;DEJKPQ7?4
-?*
 ?
inputs?????????
p

 
? "2?/+????????????????????????????
G__inference_sequential_layer_call_and_return_conditional_losses_3059295q"#,-@?=
6?3
)?&
input_1?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_3059799p"#,-??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_layer_call_fn_3059338d"#,-@?=
6?3
)?&
input_1?????????
p

 
? "???????????
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_3059449?JKI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
B__inference_dense_layer_call_and_return_conditional_losses_3060114],-0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
,__inference_sequential_layer_call_fn_3059847c"#,-??<
5?2
(?%
inputs?????????
p 

 
? "???????????
I__inference_sequential_1_layer_call_and_return_conditional_losses_3059957r:;DEJKPQ7?4
-?*
 ?
inputs?????????
p

 
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_layer_call_fn_3059211?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? ?
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_3059401?DEI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3059225?"#I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? }
)__inference_dense_1_layer_call_fn_3060139P:;/?,
%?"
 ?
inputs?????????
? "????????????
C__inference_conv2d_layer_call_and_return_conditional_losses_3059200?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_3059496?PQI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
.__inference_sequential_1_layer_call_fn_3059629x:;DEJKPQ8?5
.?+
!?
input_2?????????
p

 
? "2?/+????????????????????????????
D__inference_reshape_layer_call_and_return_conditional_losses_3060153a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0????????? 
? f
__inference_sample_3059773H:;DEJKPQ#? 
?
?
eps
? "??
.__inference_sequential_1_layer_call_fn_3059662x:;DEJKPQ8?5
.?+
!?
input_2?????????
p 

 
? "2?/+????????????????????????????
,__inference_sequential_layer_call_fn_3059836c"#,-??<
5?2
(?%
inputs?????????
p

 
? "???????????
D__inference_flatten_layer_call_and_return_conditional_losses_3060099a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
)__inference_reshape_layer_call_fn_3060158T0?-
&?#
!?
inputs??????????
? " ?????????? 