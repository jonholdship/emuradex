╔ѕ
╠Б
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
dtypetypeѕ
Й
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
executor_typestring ѕ
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.3.02v2.3.0-0-gb36436b0878кЄ
w
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	г*
shared_namedense1/kernel
p
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*
_output_shapes
:	г*
dtype0
o
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:г*
shared_namedense1/bias
h
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes	
:г*
dtype0
Љ
batch_normalization_60/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:г*-
shared_namebatch_normalization_60/gamma
і
0batch_normalization_60/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_60/gamma*
_output_shapes	
:г*
dtype0
Ј
batch_normalization_60/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:г*,
shared_namebatch_normalization_60/beta
ѕ
/batch_normalization_60/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_60/beta*
_output_shapes	
:г*
dtype0
Ю
"batch_normalization_60/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:г*3
shared_name$"batch_normalization_60/moving_mean
ќ
6batch_normalization_60/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_60/moving_mean*
_output_shapes	
:г*
dtype0
Ц
&batch_normalization_60/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:г*7
shared_name(&batch_normalization_60/moving_variance
ъ
:batch_normalization_60/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_60/moving_variance*
_output_shapes	
:г*
dtype0
x
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
г╚*
shared_namedense2/kernel
q
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel* 
_output_shapes
:
г╚*
dtype0
o
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*
shared_namedense2/bias
h
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes	
:╚*
dtype0
Љ
batch_normalization_61/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*-
shared_namebatch_normalization_61/gamma
і
0batch_normalization_61/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_61/gamma*
_output_shapes	
:╚*
dtype0
Ј
batch_normalization_61/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*,
shared_namebatch_normalization_61/beta
ѕ
/batch_normalization_61/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_61/beta*
_output_shapes	
:╚*
dtype0
Ю
"batch_normalization_61/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*3
shared_name$"batch_normalization_61/moving_mean
ќ
6batch_normalization_61/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_61/moving_mean*
_output_shapes	
:╚*
dtype0
Ц
&batch_normalization_61/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*7
shared_name(&batch_normalization_61/moving_variance
ъ
:batch_normalization_61/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_61/moving_variance*
_output_shapes	
:╚*
dtype0
w
dense3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚d*
shared_namedense3/kernel
p
!dense3/kernel/Read/ReadVariableOpReadVariableOpdense3/kernel*
_output_shapes
:	╚d*
dtype0
n
dense3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense3/bias
g
dense3/bias/Read/ReadVariableOpReadVariableOpdense3/bias*
_output_shapes
:d*
dtype0
љ
batch_normalization_62/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_namebatch_normalization_62/gamma
Ѕ
0batch_normalization_62/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_62/gamma*
_output_shapes
:d*
dtype0
ј
batch_normalization_62/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*,
shared_namebatch_normalization_62/beta
Є
/batch_normalization_62/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_62/beta*
_output_shapes
:d*
dtype0
ю
"batch_normalization_62/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"batch_normalization_62/moving_mean
Ћ
6batch_normalization_62/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_62/moving_mean*
_output_shapes
:d*
dtype0
ц
&batch_normalization_62/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*7
shared_name(&batch_normalization_62/moving_variance
Ю
:batch_normalization_62/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_62/moving_variance*
_output_shapes
:d*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:d*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
љ
batch_normalization_63/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_63/gamma
Ѕ
0batch_normalization_63/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_63/gamma*
_output_shapes
:*
dtype0
ј
batch_normalization_63/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_63/beta
Є
/batch_normalization_63/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_63/beta*
_output_shapes
:*
dtype0
ю
"batch_normalization_63/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_63/moving_mean
Ћ
6batch_normalization_63/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_63/moving_mean*
_output_shapes
:*
dtype0
ц
&batch_normalization_63/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_63/moving_variance
Ю
:batch_normalization_63/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_63/moving_variance*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
Ё
Adam/dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	г*%
shared_nameAdam/dense1/kernel/m
~
(Adam/dense1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/m*
_output_shapes
:	г*
dtype0
}
Adam/dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:г*#
shared_nameAdam/dense1/bias/m
v
&Adam/dense1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/m*
_output_shapes	
:г*
dtype0
Ъ
#Adam/batch_normalization_60/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:г*4
shared_name%#Adam/batch_normalization_60/gamma/m
ў
7Adam/batch_normalization_60/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_60/gamma/m*
_output_shapes	
:г*
dtype0
Ю
"Adam/batch_normalization_60/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:г*3
shared_name$"Adam/batch_normalization_60/beta/m
ќ
6Adam/batch_normalization_60/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_60/beta/m*
_output_shapes	
:г*
dtype0
є
Adam/dense2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
г╚*%
shared_nameAdam/dense2/kernel/m

(Adam/dense2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/m* 
_output_shapes
:
г╚*
dtype0
}
Adam/dense2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*#
shared_nameAdam/dense2/bias/m
v
&Adam/dense2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/m*
_output_shapes	
:╚*
dtype0
Ъ
#Adam/batch_normalization_61/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*4
shared_name%#Adam/batch_normalization_61/gamma/m
ў
7Adam/batch_normalization_61/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_61/gamma/m*
_output_shapes	
:╚*
dtype0
Ю
"Adam/batch_normalization_61/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*3
shared_name$"Adam/batch_normalization_61/beta/m
ќ
6Adam/batch_normalization_61/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_61/beta/m*
_output_shapes	
:╚*
dtype0
Ё
Adam/dense3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚d*%
shared_nameAdam/dense3/kernel/m
~
(Adam/dense3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense3/kernel/m*
_output_shapes
:	╚d*
dtype0
|
Adam/dense3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameAdam/dense3/bias/m
u
&Adam/dense3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense3/bias/m*
_output_shapes
:d*
dtype0
ъ
#Adam/batch_normalization_62/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#Adam/batch_normalization_62/gamma/m
Ќ
7Adam/batch_normalization_62/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_62/gamma/m*
_output_shapes
:d*
dtype0
ю
"Adam/batch_normalization_62/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"Adam/batch_normalization_62/beta/m
Ћ
6Adam/batch_normalization_62/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_62/beta/m*
_output_shapes
:d*
dtype0
ё
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

:d*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0
ъ
#Adam/batch_normalization_63/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_63/gamma/m
Ќ
7Adam/batch_normalization_63/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_63/gamma/m*
_output_shapes
:*
dtype0
ю
"Adam/batch_normalization_63/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_63/beta/m
Ћ
6Adam/batch_normalization_63/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_63/beta/m*
_output_shapes
:*
dtype0
Ё
Adam/dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	г*%
shared_nameAdam/dense1/kernel/v
~
(Adam/dense1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/v*
_output_shapes
:	г*
dtype0
}
Adam/dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:г*#
shared_nameAdam/dense1/bias/v
v
&Adam/dense1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/v*
_output_shapes	
:г*
dtype0
Ъ
#Adam/batch_normalization_60/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:г*4
shared_name%#Adam/batch_normalization_60/gamma/v
ў
7Adam/batch_normalization_60/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_60/gamma/v*
_output_shapes	
:г*
dtype0
Ю
"Adam/batch_normalization_60/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:г*3
shared_name$"Adam/batch_normalization_60/beta/v
ќ
6Adam/batch_normalization_60/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_60/beta/v*
_output_shapes	
:г*
dtype0
є
Adam/dense2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
г╚*%
shared_nameAdam/dense2/kernel/v

(Adam/dense2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/v* 
_output_shapes
:
г╚*
dtype0
}
Adam/dense2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*#
shared_nameAdam/dense2/bias/v
v
&Adam/dense2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/v*
_output_shapes	
:╚*
dtype0
Ъ
#Adam/batch_normalization_61/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*4
shared_name%#Adam/batch_normalization_61/gamma/v
ў
7Adam/batch_normalization_61/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_61/gamma/v*
_output_shapes	
:╚*
dtype0
Ю
"Adam/batch_normalization_61/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*3
shared_name$"Adam/batch_normalization_61/beta/v
ќ
6Adam/batch_normalization_61/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_61/beta/v*
_output_shapes	
:╚*
dtype0
Ё
Adam/dense3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚d*%
shared_nameAdam/dense3/kernel/v
~
(Adam/dense3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense3/kernel/v*
_output_shapes
:	╚d*
dtype0
|
Adam/dense3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameAdam/dense3/bias/v
u
&Adam/dense3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense3/bias/v*
_output_shapes
:d*
dtype0
ъ
#Adam/batch_normalization_62/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#Adam/batch_normalization_62/gamma/v
Ќ
7Adam/batch_normalization_62/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_62/gamma/v*
_output_shapes
:d*
dtype0
ю
"Adam/batch_normalization_62/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"Adam/batch_normalization_62/beta/v
Ћ
6Adam/batch_normalization_62/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_62/beta/v*
_output_shapes
:d*
dtype0
ё
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

:d*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0
ъ
#Adam/batch_normalization_63/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_63/gamma/v
Ќ
7Adam/batch_normalization_63/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_63/gamma/v*
_output_shapes
:*
dtype0
ю
"Adam/batch_normalization_63/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_63/beta/v
Ћ
6Adam/batch_normalization_63/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_63/beta/v*
_output_shapes
:*
dtype0

NoOpNoOp
щe
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*┤e
valueфeBДe Bаe
Л
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
Ќ
axis
	gamma
beta
moving_mean
 moving_variance
!trainable_variables
"regularization_losses
#	variables
$	keras_api
h

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
R
+trainable_variables
,regularization_losses
-	variables
.	keras_api
Ќ
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4trainable_variables
5regularization_losses
6	variables
7	keras_api
h

8kernel
9bias
:trainable_variables
;regularization_losses
<	variables
=	keras_api
R
>trainable_variables
?regularization_losses
@	variables
A	keras_api
Ќ
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
h

Kkernel
Lbias
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
Ќ
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
ђ
Ziter

[beta_1

\beta_2
	]decay
^learning_ratemдmДmеmЕ%mф&mФ0mг1mГ8m«9m»Cm░Dm▒Km▓Lm│Rm┤SmхvХvиvИv╣%v║&v╗0v╝1vй8vЙ9v┐Cv└Dv┴Kv┬Lv├Rv─Sv┼
v
0
1
2
3
%4
&5
06
17
88
99
C10
D11
K12
L13
R14
S15
 
Х
0
1
2
3
4
 5
%6
&7
08
19
210
311
812
913
C14
D15
E16
F17
K18
L19
R20
S21
T22
U23
Г
_layer_metrics
`non_trainable_variables

alayers
bmetrics
trainable_variables
regularization_losses
clayer_regularization_losses
	variables
 
YW
VARIABLE_VALUEdense1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
dlayer_metrics
enon_trainable_variables

flayers
gmetrics
trainable_variables
regularization_losses
hlayer_regularization_losses
	variables
 
 
 
Г
ilayer_metrics
jnon_trainable_variables

klayers
lmetrics
trainable_variables
regularization_losses
mlayer_regularization_losses
	variables
 
ge
VARIABLE_VALUEbatch_normalization_60/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_60/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_60/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_60/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
2
 3
Г
nlayer_metrics
onon_trainable_variables

players
qmetrics
!trainable_variables
"regularization_losses
rlayer_regularization_losses
#	variables
YW
VARIABLE_VALUEdense2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
Г
slayer_metrics
tnon_trainable_variables

ulayers
vmetrics
'trainable_variables
(regularization_losses
wlayer_regularization_losses
)	variables
 
 
 
Г
xlayer_metrics
ynon_trainable_variables

zlayers
{metrics
+trainable_variables
,regularization_losses
|layer_regularization_losses
-	variables
 
ge
VARIABLE_VALUEbatch_normalization_61/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_61/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_61/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_61/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
22
33
»
}layer_metrics
~non_trainable_variables

layers
ђmetrics
4trainable_variables
5regularization_losses
 Ђlayer_regularization_losses
6	variables
YW
VARIABLE_VALUEdense3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
▓
ѓlayer_metrics
Ѓnon_trainable_variables
ёlayers
Ёmetrics
:trainable_variables
;regularization_losses
 єlayer_regularization_losses
<	variables
 
 
 
▓
Єlayer_metrics
ѕnon_trainable_variables
Ѕlayers
іmetrics
>trainable_variables
?regularization_losses
 Іlayer_regularization_losses
@	variables
 
ge
VARIABLE_VALUEbatch_normalization_62/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_62/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_62/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_62/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
 

C0
D1
E2
F3
▓
їlayer_metrics
Їnon_trainable_variables
јlayers
Јmetrics
Gtrainable_variables
Hregularization_losses
 љlayer_regularization_losses
I	variables
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
 

K0
L1
▓
Љlayer_metrics
њnon_trainable_variables
Њlayers
ћmetrics
Mtrainable_variables
Nregularization_losses
 Ћlayer_regularization_losses
O	variables
 
ge
VARIABLE_VALUEbatch_normalization_63/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_63/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_63/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_63/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

R0
S1
 

R0
S1
T2
U3
▓
ќlayer_metrics
Ќnon_trainable_variables
ўlayers
Ўmetrics
Vtrainable_variables
Wregularization_losses
 џlayer_regularization_losses
X	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
 1
22
33
E4
F5
T6
U7
N
0
1
2
3
4
5
6
7
	8

9
10

Џ0
ю1
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

0
 1
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

20
31
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

E0
F1
 
 
 
 
 
 
 
 
 

T0
U1
 
 
 
8

Юtotal

ъcount
Ъ	variables
а	keras_api
I

Аtotal

бcount
Б
_fn_kwargs
ц	variables
Ц	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ю0
ъ1

Ъ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

А0
б1

ц	variables
|z
VARIABLE_VALUEAdam/dense1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#Adam/batch_normalization_60/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"Adam/batch_normalization_60/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#Adam/batch_normalization_61/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"Adam/batch_normalization_61/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#Adam/batch_normalization_62/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"Adam/batch_normalization_62/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#Adam/batch_normalization_63/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"Adam/batch_normalization_63/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#Adam/batch_normalization_60/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"Adam/batch_normalization_60/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#Adam/batch_normalization_61/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"Adam/batch_normalization_61/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#Adam/batch_normalization_62/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"Adam/batch_normalization_62/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#Adam/batch_normalization_63/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"Adam/batch_normalization_63/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_dense1_inputPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
э
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense1_inputdense1/kerneldense1/bias&batch_normalization_60/moving_variancebatch_normalization_60/gamma"batch_normalization_60/moving_meanbatch_normalization_60/betadense2/kerneldense2/bias&batch_normalization_61/moving_variancebatch_normalization_61/gamma"batch_normalization_61/moving_meanbatch_normalization_61/betadense3/kerneldense3/bias&batch_normalization_62/moving_variancebatch_normalization_62/gamma"batch_normalization_62/moving_meanbatch_normalization_62/betaoutput/kerneloutput/bias&batch_normalization_63/moving_variancebatch_normalization_63/gamma"batch_normalization_63/moving_meanbatch_normalization_63/beta*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *.
f)R'
%__inference_signature_wrapper_1000183
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
§
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp0batch_normalization_60/gamma/Read/ReadVariableOp/batch_normalization_60/beta/Read/ReadVariableOp6batch_normalization_60/moving_mean/Read/ReadVariableOp:batch_normalization_60/moving_variance/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOp0batch_normalization_61/gamma/Read/ReadVariableOp/batch_normalization_61/beta/Read/ReadVariableOp6batch_normalization_61/moving_mean/Read/ReadVariableOp:batch_normalization_61/moving_variance/Read/ReadVariableOp!dense3/kernel/Read/ReadVariableOpdense3/bias/Read/ReadVariableOp0batch_normalization_62/gamma/Read/ReadVariableOp/batch_normalization_62/beta/Read/ReadVariableOp6batch_normalization_62/moving_mean/Read/ReadVariableOp:batch_normalization_62/moving_variance/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOp0batch_normalization_63/gamma/Read/ReadVariableOp/batch_normalization_63/beta/Read/ReadVariableOp6batch_normalization_63/moving_mean/Read/ReadVariableOp:batch_normalization_63/moving_variance/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/dense1/kernel/m/Read/ReadVariableOp&Adam/dense1/bias/m/Read/ReadVariableOp7Adam/batch_normalization_60/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_60/beta/m/Read/ReadVariableOp(Adam/dense2/kernel/m/Read/ReadVariableOp&Adam/dense2/bias/m/Read/ReadVariableOp7Adam/batch_normalization_61/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_61/beta/m/Read/ReadVariableOp(Adam/dense3/kernel/m/Read/ReadVariableOp&Adam/dense3/bias/m/Read/ReadVariableOp7Adam/batch_normalization_62/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_62/beta/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp7Adam/batch_normalization_63/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_63/beta/m/Read/ReadVariableOp(Adam/dense1/kernel/v/Read/ReadVariableOp&Adam/dense1/bias/v/Read/ReadVariableOp7Adam/batch_normalization_60/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_60/beta/v/Read/ReadVariableOp(Adam/dense2/kernel/v/Read/ReadVariableOp&Adam/dense2/bias/v/Read/ReadVariableOp7Adam/batch_normalization_61/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_61/beta/v/Read/ReadVariableOp(Adam/dense3/kernel/v/Read/ReadVariableOp&Adam/dense3/bias/v/Read/ReadVariableOp7Adam/batch_normalization_62/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_62/beta/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOp7Adam/batch_normalization_63/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_63/beta/v/Read/ReadVariableOpConst*N
TinG
E2C	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *)
f$R"
 __inference__traced_save_1001279
С
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense1/kerneldense1/biasbatch_normalization_60/gammabatch_normalization_60/beta"batch_normalization_60/moving_mean&batch_normalization_60/moving_variancedense2/kerneldense2/biasbatch_normalization_61/gammabatch_normalization_61/beta"batch_normalization_61/moving_mean&batch_normalization_61/moving_variancedense3/kerneldense3/biasbatch_normalization_62/gammabatch_normalization_62/beta"batch_normalization_62/moving_mean&batch_normalization_62/moving_varianceoutput/kerneloutput/biasbatch_normalization_63/gammabatch_normalization_63/beta"batch_normalization_63/moving_mean&batch_normalization_63/moving_variance	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense1/kernel/mAdam/dense1/bias/m#Adam/batch_normalization_60/gamma/m"Adam/batch_normalization_60/beta/mAdam/dense2/kernel/mAdam/dense2/bias/m#Adam/batch_normalization_61/gamma/m"Adam/batch_normalization_61/beta/mAdam/dense3/kernel/mAdam/dense3/bias/m#Adam/batch_normalization_62/gamma/m"Adam/batch_normalization_62/beta/mAdam/output/kernel/mAdam/output/bias/m#Adam/batch_normalization_63/gamma/m"Adam/batch_normalization_63/beta/mAdam/dense1/kernel/vAdam/dense1/bias/v#Adam/batch_normalization_60/gamma/v"Adam/batch_normalization_60/beta/vAdam/dense2/kernel/vAdam/dense2/bias/v#Adam/batch_normalization_61/gamma/v"Adam/batch_normalization_61/beta/vAdam/dense3/kernel/vAdam/dense3/bias/v#Adam/batch_normalization_62/gamma/v"Adam/batch_normalization_62/beta/vAdam/output/kernel/vAdam/output/bias/v#Adam/batch_normalization_63/gamma/v"Adam/batch_normalization_63/beta/v*M
TinF
D2B*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *,
f'R%
#__inference__traced_restore_1001484ЛС
ї
f
G__inference_dropout_45_layer_call_and_return_conditional_losses_1000604

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         г2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         г*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         г2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         г2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         г2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*'
_input_shapes
:         г:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
џ
H
,__inference_dropout_47_layer_call_fn_1000877

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_47_layer_call_and_return_conditional_losses_9997132
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
┴
Ф
8__inference_batch_normalization_60_layer_call_fn_1000701

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_60_layer_call_and_return_conditional_losses_9990502
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         г::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
ъ
H
,__inference_dropout_45_layer_call_fn_1000619

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_45_layer_call_and_return_conditional_losses_9995292
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*'
_input_shapes
:         г:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
њ
Ћ
R__inference_batch_normalization_61_layer_call_and_return_conditional_losses_999190

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:╚*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:╚2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:╚2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:╚*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╚2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ╚2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:╚*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:╚2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:╚*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╚2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╚2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ╚:::::P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
я
}
(__inference_dense3_layer_call_fn_1000850

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense3_layer_call_and_return_conditional_losses_9996802
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
╗
Ф
8__inference_batch_normalization_62_layer_call_fn_1000946

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_62_layer_call_and_return_conditional_losses_9992972
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         d::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
П<
Є	
@__inference_dnn_layer_call_and_return_conditional_losses_1000069

inputs
dense1_1000009
dense1_1000011"
batch_normalization_60_1000015"
batch_normalization_60_1000017"
batch_normalization_60_1000019"
batch_normalization_60_1000021
dense2_1000024
dense2_1000026"
batch_normalization_61_1000030"
batch_normalization_61_1000032"
batch_normalization_61_1000034"
batch_normalization_61_1000036
dense3_1000039
dense3_1000041"
batch_normalization_62_1000045"
batch_normalization_62_1000047"
batch_normalization_62_1000049"
batch_normalization_62_1000051
output_1000054
output_1000056"
batch_normalization_63_1000059"
batch_normalization_63_1000061"
batch_normalization_63_1000063"
batch_normalization_63_1000065
identityѕб.batch_normalization_60/StatefulPartitionedCallб.batch_normalization_61/StatefulPartitionedCallб.batch_normalization_62/StatefulPartitionedCallб.batch_normalization_63/StatefulPartitionedCallбdense1/StatefulPartitionedCallбdense2/StatefulPartitionedCallбdense3/StatefulPartitionedCallбoutput/StatefulPartitionedCallљ
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_1000009dense1_1000011*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_9994962 
dense1/StatefulPartitionedCall 
dropout_45/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_45_layer_call_and_return_conditional_losses_9995292
dropout_45/PartitionedCall┴
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCall#dropout_45/PartitionedCall:output:0batch_normalization_60_1000015batch_normalization_60_1000017batch_normalization_60_1000019batch_normalization_60_1000021*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_60_layer_call_and_return_conditional_losses_99905020
.batch_normalization_60/StatefulPartitionedCall┴
dense2/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0dense2_1000024dense2_1000026*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_9995882 
dense2/StatefulPartitionedCall 
dropout_46/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_46_layer_call_and_return_conditional_losses_9996212
dropout_46/PartitionedCall┴
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall#dropout_46/PartitionedCall:output:0batch_normalization_61_1000030batch_normalization_61_1000032batch_normalization_61_1000034batch_normalization_61_1000036*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_61_layer_call_and_return_conditional_losses_99919020
.batch_normalization_61/StatefulPartitionedCall└
dense3/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0dense3_1000039dense3_1000041*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense3_layer_call_and_return_conditional_losses_9996802 
dense3/StatefulPartitionedCall■
dropout_47/PartitionedCallPartitionedCall'dense3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_47_layer_call_and_return_conditional_losses_9997132
dropout_47/PartitionedCall└
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall#dropout_47/PartitionedCall:output:0batch_normalization_62_1000045batch_normalization_62_1000047batch_normalization_62_1000049batch_normalization_62_1000051*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_62_layer_call_and_return_conditional_losses_99933020
.batch_normalization_62/StatefulPartitionedCall└
output/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_62/StatefulPartitionedCall:output:0output_1000054output_1000056*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9997722 
output/StatefulPartitionedCall─
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall'output/StatefulPartitionedCall:output:0batch_normalization_63_1000059batch_normalization_63_1000061batch_normalization_63_1000063batch_normalization_63_1000065*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_63_layer_call_and_return_conditional_losses_99947020
.batch_normalization_63/StatefulPartitionedCallМ
IdentityIdentity7batch_normalization_63/StatefulPartitionedCall:output:0/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*є
_input_shapesu
s:         ::::::::::::::::::::::::2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
я
}
(__inference_dense1_layer_call_fn_1000592

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_9994962
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╬
e
G__inference_dropout_46_layer_call_and_return_conditional_losses_1000738

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ╚2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ╚2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
┐
Ф
8__inference_batch_normalization_60_layer_call_fn_1000688

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_60_layer_call_and_return_conditional_losses_9990172
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         г::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
▄
}
(__inference_output_layer_call_fn_1000979

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9997722
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
ф
e
,__inference_dropout_45_layer_call_fn_1000614

inputs
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_45_layer_call_and_return_conditional_losses_9995242
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*'
_input_shapes
:         г22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
Я
}
(__inference_dense2_layer_call_fn_1000721

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_9995882
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         г::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
Ф
Ф
C__inference_dense3_layer_call_and_return_conditional_losses_1000841

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚:::P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
К)
╬
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_1000655

inputs
assignmovingavg_1000630
assignmovingavg_1_1000636)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	г*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	г2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         г2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	г*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:г*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:г*
squeeze_dims
 2
moments/Squeeze_1Ъ
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/1000630*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg/decayЋ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1000630*
_output_shapes	
:г*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/1000630*
_output_shapes	
:г2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/1000630*
_output_shapes	
:г2
AssignMovingAvg/mulЃ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1000630AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/1000630*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЦ
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/1000636*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg_1/decayЏ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1000636*
_output_shapes	
:г*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¤
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1000636*
_output_shapes	
:г2
AssignMovingAvg_1/subк
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1000636*
_output_shapes	
:г2
AssignMovingAvg_1/mulЈ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1000636AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/1000636*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:г2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:г2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:г*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:г2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         г2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:г2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:г*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:г2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         г2
batchnorm/add_1Х
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         г::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
╬
e
G__inference_dropout_45_layer_call_and_return_conditional_losses_1000609

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         г2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         г2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         г:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
Ё
ќ
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_1001035

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕњ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѕ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         2
batchnorm/mul_1ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Ё
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Ѓ
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         :::::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ф
e
,__inference_dropout_46_layer_call_fn_1000743

inputs
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_46_layer_call_and_return_conditional_losses_9996162
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╚22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
ъ
H
,__inference_dropout_46_layer_call_fn_1000748

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_46_layer_call_and_return_conditional_losses_9996212
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
╔
d
F__inference_dropout_47_layer_call_and_return_conditional_losses_999713

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
й
Ф
8__inference_batch_normalization_63_layer_call_fn_1001061

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_63_layer_call_and_return_conditional_losses_9994702
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
«
Ф
C__inference_dense1_layer_call_and_return_conditional_losses_1000583

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	г*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         г2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:г*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         г2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         г2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
іA
П	
?__inference_dnn_layer_call_and_return_conditional_losses_999953

inputs
dense1_999893
dense1_999895!
batch_normalization_60_999899!
batch_normalization_60_999901!
batch_normalization_60_999903!
batch_normalization_60_999905
dense2_999908
dense2_999910!
batch_normalization_61_999914!
batch_normalization_61_999916!
batch_normalization_61_999918!
batch_normalization_61_999920
dense3_999923
dense3_999925!
batch_normalization_62_999929!
batch_normalization_62_999931!
batch_normalization_62_999933!
batch_normalization_62_999935
output_999938
output_999940!
batch_normalization_63_999943!
batch_normalization_63_999945!
batch_normalization_63_999947!
batch_normalization_63_999949
identityѕб.batch_normalization_60/StatefulPartitionedCallб.batch_normalization_61/StatefulPartitionedCallб.batch_normalization_62/StatefulPartitionedCallб.batch_normalization_63/StatefulPartitionedCallбdense1/StatefulPartitionedCallбdense2/StatefulPartitionedCallбdense3/StatefulPartitionedCallб"dropout_45/StatefulPartitionedCallб"dropout_46/StatefulPartitionedCallб"dropout_47/StatefulPartitionedCallбoutput/StatefulPartitionedCallј
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_999893dense1_999895*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_9994962 
dense1/StatefulPartitionedCallЌ
"dropout_45/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_45_layer_call_and_return_conditional_losses_9995242$
"dropout_45/StatefulPartitionedCall├
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCall+dropout_45/StatefulPartitionedCall:output:0batch_normalization_60_999899batch_normalization_60_999901batch_normalization_60_999903batch_normalization_60_999905*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_60_layer_call_and_return_conditional_losses_99901720
.batch_normalization_60/StatefulPartitionedCall┐
dense2/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0dense2_999908dense2_999910*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_9995882 
dense2/StatefulPartitionedCall╝
"dropout_46/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0#^dropout_45/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_46_layer_call_and_return_conditional_losses_9996162$
"dropout_46/StatefulPartitionedCall├
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall+dropout_46/StatefulPartitionedCall:output:0batch_normalization_61_999914batch_normalization_61_999916batch_normalization_61_999918batch_normalization_61_999920*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_61_layer_call_and_return_conditional_losses_99915720
.batch_normalization_61/StatefulPartitionedCallЙ
dense3/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0dense3_999923dense3_999925*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense3_layer_call_and_return_conditional_losses_9996802 
dense3/StatefulPartitionedCall╗
"dropout_47/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0#^dropout_46/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_47_layer_call_and_return_conditional_losses_9997082$
"dropout_47/StatefulPartitionedCall┬
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall+dropout_47/StatefulPartitionedCall:output:0batch_normalization_62_999929batch_normalization_62_999931batch_normalization_62_999933batch_normalization_62_999935*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_62_layer_call_and_return_conditional_losses_99929720
.batch_normalization_62/StatefulPartitionedCallЙ
output/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_62/StatefulPartitionedCall:output:0output_999938output_999940*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9997722 
output/StatefulPartitionedCallЙ
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall'output/StatefulPartitionedCall:output:0batch_normalization_63_999943batch_normalization_63_999945batch_normalization_63_999947batch_normalization_63_999949*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_63_layer_call_and_return_conditional_losses_99943720
.batch_normalization_63/StatefulPartitionedCall┬
IdentityIdentity7batch_normalization_63/StatefulPartitionedCall:output:0/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense3/StatefulPartitionedCall#^dropout_45/StatefulPartitionedCall#^dropout_46/StatefulPartitionedCall#^dropout_47/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*є
_input_shapesu
s:         ::::::::::::::::::::::::2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2H
"dropout_45/StatefulPartitionedCall"dropout_45/StatefulPartitionedCall2H
"dropout_46/StatefulPartitionedCall"dropout_46/StatefulPartitionedCall2H
"dropout_47/StatefulPartitionedCall"dropout_47/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
І
e
F__inference_dropout_46_layer_call_and_return_conditional_losses_999616

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ╚2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╚2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ╚2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
Д
ф
B__inference_output_layer_call_and_return_conditional_losses_999772

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d:::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
┼
О
%__inference_dnn_layer_call_fn_1000120
dense1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCalldense1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_dnn_layer_call_and_return_conditional_losses_10000692
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*є
_input_shapesu
s:         ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:         
&
_user_specified_namedense1_input
Ё
ќ
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_1000933

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕњ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѕ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         d2
batchnorm/mul_1ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp_1Ё
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:d2
batchnorm/mul_2ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp_2Ѓ
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         d2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         d:::::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
д
О
%__inference_signature_wrapper_1000183
dense1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityѕбStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCalldense1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__wrapped_model_9989212
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*є
_input_shapesu
s:         ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:         
&
_user_specified_namedense1_input
ѕФ
М
@__inference_dnn_layer_call_and_return_conditional_losses_1000367

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource2
.batch_normalization_60_assignmovingavg_10002094
0batch_normalization_60_assignmovingavg_1_1000215@
<batch_normalization_60_batchnorm_mul_readvariableop_resource<
8batch_normalization_60_batchnorm_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource2
.batch_normalization_61_assignmovingavg_10002564
0batch_normalization_61_assignmovingavg_1_1000262@
<batch_normalization_61_batchnorm_mul_readvariableop_resource<
8batch_normalization_61_batchnorm_readvariableop_resource)
%dense3_matmul_readvariableop_resource*
&dense3_biasadd_readvariableop_resource2
.batch_normalization_62_assignmovingavg_10003034
0batch_normalization_62_assignmovingavg_1_1000309@
<batch_normalization_62_batchnorm_mul_readvariableop_resource<
8batch_normalization_62_batchnorm_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource2
.batch_normalization_63_assignmovingavg_10003424
0batch_normalization_63_assignmovingavg_1_1000348@
<batch_normalization_63_batchnorm_mul_readvariableop_resource<
8batch_normalization_63_batchnorm_readvariableop_resource
identityѕб:batch_normalization_60/AssignMovingAvg/AssignSubVariableOpб<batch_normalization_60/AssignMovingAvg_1/AssignSubVariableOpб:batch_normalization_61/AssignMovingAvg/AssignSubVariableOpб<batch_normalization_61/AssignMovingAvg_1/AssignSubVariableOpб:batch_normalization_62/AssignMovingAvg/AssignSubVariableOpб<batch_normalization_62/AssignMovingAvg_1/AssignSubVariableOpб:batch_normalization_63/AssignMovingAvg/AssignSubVariableOpб<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOpБ
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	г*
dtype02
dense1/MatMul/ReadVariableOpЅ
dense1/MatMulMatMulinputs$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         г2
dense1/MatMulб
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:г*
dtype02
dense1/BiasAdd/ReadVariableOpъ
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         г2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:         г2
dense1/Reluy
dropout_45/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_45/dropout/Constе
dropout_45/dropout/MulMuldense1/Relu:activations:0!dropout_45/dropout/Const:output:0*
T0*(
_output_shapes
:         г2
dropout_45/dropout/Mul}
dropout_45/dropout/ShapeShapedense1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_45/dropout/Shapeо
/dropout_45/dropout/random_uniform/RandomUniformRandomUniform!dropout_45/dropout/Shape:output:0*
T0*(
_output_shapes
:         г*
dtype021
/dropout_45/dropout/random_uniform/RandomUniformІ
!dropout_45/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_45/dropout/GreaterEqual/yв
dropout_45/dropout/GreaterEqualGreaterEqual8dropout_45/dropout/random_uniform/RandomUniform:output:0*dropout_45/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         г2!
dropout_45/dropout/GreaterEqualА
dropout_45/dropout/CastCast#dropout_45/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         г2
dropout_45/dropout/CastД
dropout_45/dropout/Mul_1Muldropout_45/dropout/Mul:z:0dropout_45/dropout/Cast:y:0*
T0*(
_output_shapes
:         г2
dropout_45/dropout/Mul_1И
5batch_normalization_60/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_60/moments/mean/reduction_indicesв
#batch_normalization_60/moments/meanMeandropout_45/dropout/Mul_1:z:0>batch_normalization_60/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	г*
	keep_dims(2%
#batch_normalization_60/moments/mean┬
+batch_normalization_60/moments/StopGradientStopGradient,batch_normalization_60/moments/mean:output:0*
T0*
_output_shapes
:	г2-
+batch_normalization_60/moments/StopGradientђ
0batch_normalization_60/moments/SquaredDifferenceSquaredDifferencedropout_45/dropout/Mul_1:z:04batch_normalization_60/moments/StopGradient:output:0*
T0*(
_output_shapes
:         г22
0batch_normalization_60/moments/SquaredDifference└
9batch_normalization_60/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_60/moments/variance/reduction_indicesЈ
'batch_normalization_60/moments/varianceMean4batch_normalization_60/moments/SquaredDifference:z:0Bbatch_normalization_60/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	г*
	keep_dims(2)
'batch_normalization_60/moments/varianceк
&batch_normalization_60/moments/SqueezeSqueeze,batch_normalization_60/moments/mean:output:0*
T0*
_output_shapes	
:г*
squeeze_dims
 2(
&batch_normalization_60/moments/Squeeze╬
(batch_normalization_60/moments/Squeeze_1Squeeze0batch_normalization_60/moments/variance:output:0*
T0*
_output_shapes	
:г*
squeeze_dims
 2*
(batch_normalization_60/moments/Squeeze_1С
,batch_normalization_60/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_60/AssignMovingAvg/1000209*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2.
,batch_normalization_60/AssignMovingAvg/decay┌
5batch_normalization_60/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_60_assignmovingavg_1000209*
_output_shapes	
:г*
dtype027
5batch_normalization_60/AssignMovingAvg/ReadVariableOpИ
*batch_normalization_60/AssignMovingAvg/subSub=batch_normalization_60/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_60/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_60/AssignMovingAvg/1000209*
_output_shapes	
:г2,
*batch_normalization_60/AssignMovingAvg/sub»
*batch_normalization_60/AssignMovingAvg/mulMul.batch_normalization_60/AssignMovingAvg/sub:z:05batch_normalization_60/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_60/AssignMovingAvg/1000209*
_output_shapes	
:г2,
*batch_normalization_60/AssignMovingAvg/mulЇ
:batch_normalization_60/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_60_assignmovingavg_1000209.batch_normalization_60/AssignMovingAvg/mul:z:06^batch_normalization_60/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_60/AssignMovingAvg/1000209*
_output_shapes
 *
dtype02<
:batch_normalization_60/AssignMovingAvg/AssignSubVariableOpЖ
.batch_normalization_60/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_60/AssignMovingAvg_1/1000215*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.batch_normalization_60/AssignMovingAvg_1/decayЯ
7batch_normalization_60/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_60_assignmovingavg_1_1000215*
_output_shapes	
:г*
dtype029
7batch_normalization_60/AssignMovingAvg_1/ReadVariableOp┬
,batch_normalization_60/AssignMovingAvg_1/subSub?batch_normalization_60/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_60/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_60/AssignMovingAvg_1/1000215*
_output_shapes	
:г2.
,batch_normalization_60/AssignMovingAvg_1/sub╣
,batch_normalization_60/AssignMovingAvg_1/mulMul0batch_normalization_60/AssignMovingAvg_1/sub:z:07batch_normalization_60/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_60/AssignMovingAvg_1/1000215*
_output_shapes	
:г2.
,batch_normalization_60/AssignMovingAvg_1/mulЎ
<batch_normalization_60/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_60_assignmovingavg_1_10002150batch_normalization_60/AssignMovingAvg_1/mul:z:08^batch_normalization_60/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_60/AssignMovingAvg_1/1000215*
_output_shapes
 *
dtype02>
<batch_normalization_60/AssignMovingAvg_1/AssignSubVariableOpЋ
&batch_normalization_60/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&batch_normalization_60/batchnorm/add/y▀
$batch_normalization_60/batchnorm/addAddV21batch_normalization_60/moments/Squeeze_1:output:0/batch_normalization_60/batchnorm/add/y:output:0*
T0*
_output_shapes	
:г2&
$batch_normalization_60/batchnorm/addЕ
&batch_normalization_60/batchnorm/RsqrtRsqrt(batch_normalization_60/batchnorm/add:z:0*
T0*
_output_shapes	
:г2(
&batch_normalization_60/batchnorm/RsqrtС
3batch_normalization_60/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_60_batchnorm_mul_readvariableop_resource*
_output_shapes	
:г*
dtype025
3batch_normalization_60/batchnorm/mul/ReadVariableOpР
$batch_normalization_60/batchnorm/mulMul*batch_normalization_60/batchnorm/Rsqrt:y:0;batch_normalization_60/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:г2&
$batch_normalization_60/batchnorm/mulм
&batch_normalization_60/batchnorm/mul_1Muldropout_45/dropout/Mul_1:z:0(batch_normalization_60/batchnorm/mul:z:0*
T0*(
_output_shapes
:         г2(
&batch_normalization_60/batchnorm/mul_1п
&batch_normalization_60/batchnorm/mul_2Mul/batch_normalization_60/moments/Squeeze:output:0(batch_normalization_60/batchnorm/mul:z:0*
T0*
_output_shapes	
:г2(
&batch_normalization_60/batchnorm/mul_2п
/batch_normalization_60/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_60_batchnorm_readvariableop_resource*
_output_shapes	
:г*
dtype021
/batch_normalization_60/batchnorm/ReadVariableOpя
$batch_normalization_60/batchnorm/subSub7batch_normalization_60/batchnorm/ReadVariableOp:value:0*batch_normalization_60/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:г2&
$batch_normalization_60/batchnorm/subР
&batch_normalization_60/batchnorm/add_1AddV2*batch_normalization_60/batchnorm/mul_1:z:0(batch_normalization_60/batchnorm/sub:z:0*
T0*(
_output_shapes
:         г2(
&batch_normalization_60/batchnorm/add_1ц
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource* 
_output_shapes
:
г╚*
dtype02
dense2/MatMul/ReadVariableOpГ
dense2/MatMulMatMul*batch_normalization_60/batchnorm/add_1:z:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
dense2/MatMulб
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
dense2/BiasAdd/ReadVariableOpъ
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
dense2/BiasAddn
dense2/ReluReludense2/BiasAdd:output:0*
T0*(
_output_shapes
:         ╚2
dense2/Reluy
dropout_46/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_46/dropout/Constе
dropout_46/dropout/MulMuldense2/Relu:activations:0!dropout_46/dropout/Const:output:0*
T0*(
_output_shapes
:         ╚2
dropout_46/dropout/Mul}
dropout_46/dropout/ShapeShapedense2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_46/dropout/Shapeо
/dropout_46/dropout/random_uniform/RandomUniformRandomUniform!dropout_46/dropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype021
/dropout_46/dropout/random_uniform/RandomUniformІ
!dropout_46/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_46/dropout/GreaterEqual/yв
dropout_46/dropout/GreaterEqualGreaterEqual8dropout_46/dropout/random_uniform/RandomUniform:output:0*dropout_46/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚2!
dropout_46/dropout/GreaterEqualА
dropout_46/dropout/CastCast#dropout_46/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╚2
dropout_46/dropout/CastД
dropout_46/dropout/Mul_1Muldropout_46/dropout/Mul:z:0dropout_46/dropout/Cast:y:0*
T0*(
_output_shapes
:         ╚2
dropout_46/dropout/Mul_1И
5batch_normalization_61/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_61/moments/mean/reduction_indicesв
#batch_normalization_61/moments/meanMeandropout_46/dropout/Mul_1:z:0>batch_normalization_61/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	╚*
	keep_dims(2%
#batch_normalization_61/moments/mean┬
+batch_normalization_61/moments/StopGradientStopGradient,batch_normalization_61/moments/mean:output:0*
T0*
_output_shapes
:	╚2-
+batch_normalization_61/moments/StopGradientђ
0batch_normalization_61/moments/SquaredDifferenceSquaredDifferencedropout_46/dropout/Mul_1:z:04batch_normalization_61/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ╚22
0batch_normalization_61/moments/SquaredDifference└
9batch_normalization_61/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_61/moments/variance/reduction_indicesЈ
'batch_normalization_61/moments/varianceMean4batch_normalization_61/moments/SquaredDifference:z:0Bbatch_normalization_61/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	╚*
	keep_dims(2)
'batch_normalization_61/moments/varianceк
&batch_normalization_61/moments/SqueezeSqueeze,batch_normalization_61/moments/mean:output:0*
T0*
_output_shapes	
:╚*
squeeze_dims
 2(
&batch_normalization_61/moments/Squeeze╬
(batch_normalization_61/moments/Squeeze_1Squeeze0batch_normalization_61/moments/variance:output:0*
T0*
_output_shapes	
:╚*
squeeze_dims
 2*
(batch_normalization_61/moments/Squeeze_1С
,batch_normalization_61/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_61/AssignMovingAvg/1000256*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2.
,batch_normalization_61/AssignMovingAvg/decay┌
5batch_normalization_61/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_61_assignmovingavg_1000256*
_output_shapes	
:╚*
dtype027
5batch_normalization_61/AssignMovingAvg/ReadVariableOpИ
*batch_normalization_61/AssignMovingAvg/subSub=batch_normalization_61/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_61/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_61/AssignMovingAvg/1000256*
_output_shapes	
:╚2,
*batch_normalization_61/AssignMovingAvg/sub»
*batch_normalization_61/AssignMovingAvg/mulMul.batch_normalization_61/AssignMovingAvg/sub:z:05batch_normalization_61/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_61/AssignMovingAvg/1000256*
_output_shapes	
:╚2,
*batch_normalization_61/AssignMovingAvg/mulЇ
:batch_normalization_61/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_61_assignmovingavg_1000256.batch_normalization_61/AssignMovingAvg/mul:z:06^batch_normalization_61/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_61/AssignMovingAvg/1000256*
_output_shapes
 *
dtype02<
:batch_normalization_61/AssignMovingAvg/AssignSubVariableOpЖ
.batch_normalization_61/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_61/AssignMovingAvg_1/1000262*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.batch_normalization_61/AssignMovingAvg_1/decayЯ
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_61_assignmovingavg_1_1000262*
_output_shapes	
:╚*
dtype029
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp┬
,batch_normalization_61/AssignMovingAvg_1/subSub?batch_normalization_61/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_61/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_61/AssignMovingAvg_1/1000262*
_output_shapes	
:╚2.
,batch_normalization_61/AssignMovingAvg_1/sub╣
,batch_normalization_61/AssignMovingAvg_1/mulMul0batch_normalization_61/AssignMovingAvg_1/sub:z:07batch_normalization_61/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_61/AssignMovingAvg_1/1000262*
_output_shapes	
:╚2.
,batch_normalization_61/AssignMovingAvg_1/mulЎ
<batch_normalization_61/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_61_assignmovingavg_1_10002620batch_normalization_61/AssignMovingAvg_1/mul:z:08^batch_normalization_61/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_61/AssignMovingAvg_1/1000262*
_output_shapes
 *
dtype02>
<batch_normalization_61/AssignMovingAvg_1/AssignSubVariableOpЋ
&batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&batch_normalization_61/batchnorm/add/y▀
$batch_normalization_61/batchnorm/addAddV21batch_normalization_61/moments/Squeeze_1:output:0/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes	
:╚2&
$batch_normalization_61/batchnorm/addЕ
&batch_normalization_61/batchnorm/RsqrtRsqrt(batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes	
:╚2(
&batch_normalization_61/batchnorm/RsqrtС
3batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes	
:╚*
dtype025
3batch_normalization_61/batchnorm/mul/ReadVariableOpР
$batch_normalization_61/batchnorm/mulMul*batch_normalization_61/batchnorm/Rsqrt:y:0;batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╚2&
$batch_normalization_61/batchnorm/mulм
&batch_normalization_61/batchnorm/mul_1Muldropout_46/dropout/Mul_1:z:0(batch_normalization_61/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ╚2(
&batch_normalization_61/batchnorm/mul_1п
&batch_normalization_61/batchnorm/mul_2Mul/batch_normalization_61/moments/Squeeze:output:0(batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes	
:╚2(
&batch_normalization_61/batchnorm/mul_2п
/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes	
:╚*
dtype021
/batch_normalization_61/batchnorm/ReadVariableOpя
$batch_normalization_61/batchnorm/subSub7batch_normalization_61/batchnorm/ReadVariableOp:value:0*batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╚2&
$batch_normalization_61/batchnorm/subР
&batch_normalization_61/batchnorm/add_1AddV2*batch_normalization_61/batchnorm/mul_1:z:0(batch_normalization_61/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╚2(
&batch_normalization_61/batchnorm/add_1Б
dense3/MatMul/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource*
_output_shapes
:	╚d*
dtype02
dense3/MatMul/ReadVariableOpг
dense3/MatMulMatMul*batch_normalization_61/batchnorm/add_1:z:0$dense3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense3/MatMulА
dense3/BiasAdd/ReadVariableOpReadVariableOp&dense3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
dense3/BiasAdd/ReadVariableOpЮ
dense3/BiasAddBiasAdddense3/MatMul:product:0%dense3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense3/BiasAddm
dense3/ReluReludense3/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
dense3/Reluy
dropout_47/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_47/dropout/ConstД
dropout_47/dropout/MulMuldense3/Relu:activations:0!dropout_47/dropout/Const:output:0*
T0*'
_output_shapes
:         d2
dropout_47/dropout/Mul}
dropout_47/dropout/ShapeShapedense3/Relu:activations:0*
T0*
_output_shapes
:2
dropout_47/dropout/ShapeН
/dropout_47/dropout/random_uniform/RandomUniformRandomUniform!dropout_47/dropout/Shape:output:0*
T0*'
_output_shapes
:         d*
dtype021
/dropout_47/dropout/random_uniform/RandomUniformІ
!dropout_47/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_47/dropout/GreaterEqual/yЖ
dropout_47/dropout/GreaterEqualGreaterEqual8dropout_47/dropout/random_uniform/RandomUniform:output:0*dropout_47/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         d2!
dropout_47/dropout/GreaterEqualа
dropout_47/dropout/CastCast#dropout_47/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         d2
dropout_47/dropout/Castд
dropout_47/dropout/Mul_1Muldropout_47/dropout/Mul:z:0dropout_47/dropout/Cast:y:0*
T0*'
_output_shapes
:         d2
dropout_47/dropout/Mul_1И
5batch_normalization_62/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_62/moments/mean/reduction_indicesЖ
#batch_normalization_62/moments/meanMeandropout_47/dropout/Mul_1:z:0>batch_normalization_62/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2%
#batch_normalization_62/moments/mean┴
+batch_normalization_62/moments/StopGradientStopGradient,batch_normalization_62/moments/mean:output:0*
T0*
_output_shapes

:d2-
+batch_normalization_62/moments/StopGradient 
0batch_normalization_62/moments/SquaredDifferenceSquaredDifferencedropout_47/dropout/Mul_1:z:04batch_normalization_62/moments/StopGradient:output:0*
T0*'
_output_shapes
:         d22
0batch_normalization_62/moments/SquaredDifference└
9batch_normalization_62/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_62/moments/variance/reduction_indicesј
'batch_normalization_62/moments/varianceMean4batch_normalization_62/moments/SquaredDifference:z:0Bbatch_normalization_62/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2)
'batch_normalization_62/moments/variance┼
&batch_normalization_62/moments/SqueezeSqueeze,batch_normalization_62/moments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2(
&batch_normalization_62/moments/Squeeze═
(batch_normalization_62/moments/Squeeze_1Squeeze0batch_normalization_62/moments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2*
(batch_normalization_62/moments/Squeeze_1С
,batch_normalization_62/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_62/AssignMovingAvg/1000303*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2.
,batch_normalization_62/AssignMovingAvg/decay┘
5batch_normalization_62/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_62_assignmovingavg_1000303*
_output_shapes
:d*
dtype027
5batch_normalization_62/AssignMovingAvg/ReadVariableOpи
*batch_normalization_62/AssignMovingAvg/subSub=batch_normalization_62/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_62/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_62/AssignMovingAvg/1000303*
_output_shapes
:d2,
*batch_normalization_62/AssignMovingAvg/sub«
*batch_normalization_62/AssignMovingAvg/mulMul.batch_normalization_62/AssignMovingAvg/sub:z:05batch_normalization_62/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_62/AssignMovingAvg/1000303*
_output_shapes
:d2,
*batch_normalization_62/AssignMovingAvg/mulЇ
:batch_normalization_62/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_62_assignmovingavg_1000303.batch_normalization_62/AssignMovingAvg/mul:z:06^batch_normalization_62/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_62/AssignMovingAvg/1000303*
_output_shapes
 *
dtype02<
:batch_normalization_62/AssignMovingAvg/AssignSubVariableOpЖ
.batch_normalization_62/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_62/AssignMovingAvg_1/1000309*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.batch_normalization_62/AssignMovingAvg_1/decay▀
7batch_normalization_62/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_62_assignmovingavg_1_1000309*
_output_shapes
:d*
dtype029
7batch_normalization_62/AssignMovingAvg_1/ReadVariableOp┴
,batch_normalization_62/AssignMovingAvg_1/subSub?batch_normalization_62/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_62/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_62/AssignMovingAvg_1/1000309*
_output_shapes
:d2.
,batch_normalization_62/AssignMovingAvg_1/subИ
,batch_normalization_62/AssignMovingAvg_1/mulMul0batch_normalization_62/AssignMovingAvg_1/sub:z:07batch_normalization_62/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_62/AssignMovingAvg_1/1000309*
_output_shapes
:d2.
,batch_normalization_62/AssignMovingAvg_1/mulЎ
<batch_normalization_62/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_62_assignmovingavg_1_10003090batch_normalization_62/AssignMovingAvg_1/mul:z:08^batch_normalization_62/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_62/AssignMovingAvg_1/1000309*
_output_shapes
 *
dtype02>
<batch_normalization_62/AssignMovingAvg_1/AssignSubVariableOpЋ
&batch_normalization_62/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&batch_normalization_62/batchnorm/add/yя
$batch_normalization_62/batchnorm/addAddV21batch_normalization_62/moments/Squeeze_1:output:0/batch_normalization_62/batchnorm/add/y:output:0*
T0*
_output_shapes
:d2&
$batch_normalization_62/batchnorm/addе
&batch_normalization_62/batchnorm/RsqrtRsqrt(batch_normalization_62/batchnorm/add:z:0*
T0*
_output_shapes
:d2(
&batch_normalization_62/batchnorm/Rsqrtс
3batch_normalization_62/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_62_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype025
3batch_normalization_62/batchnorm/mul/ReadVariableOpр
$batch_normalization_62/batchnorm/mulMul*batch_normalization_62/batchnorm/Rsqrt:y:0;batch_normalization_62/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2&
$batch_normalization_62/batchnorm/mulЛ
&batch_normalization_62/batchnorm/mul_1Muldropout_47/dropout/Mul_1:z:0(batch_normalization_62/batchnorm/mul:z:0*
T0*'
_output_shapes
:         d2(
&batch_normalization_62/batchnorm/mul_1О
&batch_normalization_62/batchnorm/mul_2Mul/batch_normalization_62/moments/Squeeze:output:0(batch_normalization_62/batchnorm/mul:z:0*
T0*
_output_shapes
:d2(
&batch_normalization_62/batchnorm/mul_2О
/batch_normalization_62/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_62_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype021
/batch_normalization_62/batchnorm/ReadVariableOpП
$batch_normalization_62/batchnorm/subSub7batch_normalization_62/batchnorm/ReadVariableOp:value:0*batch_normalization_62/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2&
$batch_normalization_62/batchnorm/subр
&batch_normalization_62/batchnorm/add_1AddV2*batch_normalization_62/batchnorm/mul_1:z:0(batch_normalization_62/batchnorm/sub:z:0*
T0*'
_output_shapes
:         d2(
&batch_normalization_62/batchnorm/add_1б
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
output/MatMul/ReadVariableOpг
output/MatMulMatMul*batch_normalization_62/batchnorm/add_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/MatMulА
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЮ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/BiasAddm
output/ReluReluoutput/BiasAdd:output:0*
T0*'
_output_shapes
:         2
output/ReluИ
5batch_normalization_63/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_63/moments/mean/reduction_indicesу
#batch_normalization_63/moments/meanMeanoutput/Relu:activations:0>batch_normalization_63/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_63/moments/mean┴
+batch_normalization_63/moments/StopGradientStopGradient,batch_normalization_63/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_63/moments/StopGradientЧ
0batch_normalization_63/moments/SquaredDifferenceSquaredDifferenceoutput/Relu:activations:04batch_normalization_63/moments/StopGradient:output:0*
T0*'
_output_shapes
:         22
0batch_normalization_63/moments/SquaredDifference└
9batch_normalization_63/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_63/moments/variance/reduction_indicesј
'batch_normalization_63/moments/varianceMean4batch_normalization_63/moments/SquaredDifference:z:0Bbatch_normalization_63/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_63/moments/variance┼
&batch_normalization_63/moments/SqueezeSqueeze,batch_normalization_63/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_63/moments/Squeeze═
(batch_normalization_63/moments/Squeeze_1Squeeze0batch_normalization_63/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_63/moments/Squeeze_1С
,batch_normalization_63/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_63/AssignMovingAvg/1000342*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2.
,batch_normalization_63/AssignMovingAvg/decay┘
5batch_normalization_63/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_63_assignmovingavg_1000342*
_output_shapes
:*
dtype027
5batch_normalization_63/AssignMovingAvg/ReadVariableOpи
*batch_normalization_63/AssignMovingAvg/subSub=batch_normalization_63/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_63/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_63/AssignMovingAvg/1000342*
_output_shapes
:2,
*batch_normalization_63/AssignMovingAvg/sub«
*batch_normalization_63/AssignMovingAvg/mulMul.batch_normalization_63/AssignMovingAvg/sub:z:05batch_normalization_63/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_63/AssignMovingAvg/1000342*
_output_shapes
:2,
*batch_normalization_63/AssignMovingAvg/mulЇ
:batch_normalization_63/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_63_assignmovingavg_1000342.batch_normalization_63/AssignMovingAvg/mul:z:06^batch_normalization_63/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_63/AssignMovingAvg/1000342*
_output_shapes
 *
dtype02<
:batch_normalization_63/AssignMovingAvg/AssignSubVariableOpЖ
.batch_normalization_63/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_63/AssignMovingAvg_1/1000348*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.batch_normalization_63/AssignMovingAvg_1/decay▀
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_63_assignmovingavg_1_1000348*
_output_shapes
:*
dtype029
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOp┴
,batch_normalization_63/AssignMovingAvg_1/subSub?batch_normalization_63/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_63/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_63/AssignMovingAvg_1/1000348*
_output_shapes
:2.
,batch_normalization_63/AssignMovingAvg_1/subИ
,batch_normalization_63/AssignMovingAvg_1/mulMul0batch_normalization_63/AssignMovingAvg_1/sub:z:07batch_normalization_63/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_63/AssignMovingAvg_1/1000348*
_output_shapes
:2.
,batch_normalization_63/AssignMovingAvg_1/mulЎ
<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_63_assignmovingavg_1_10003480batch_normalization_63/AssignMovingAvg_1/mul:z:08^batch_normalization_63/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_63/AssignMovingAvg_1/1000348*
_output_shapes
 *
dtype02>
<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOpЋ
&batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&batch_normalization_63/batchnorm/add/yя
$batch_normalization_63/batchnorm/addAddV21batch_normalization_63/moments/Squeeze_1:output:0/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_63/batchnorm/addе
&batch_normalization_63/batchnorm/RsqrtRsqrt(batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_63/batchnorm/Rsqrtс
3batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_63/batchnorm/mul/ReadVariableOpр
$batch_normalization_63/batchnorm/mulMul*batch_normalization_63/batchnorm/Rsqrt:y:0;batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_63/batchnorm/mul╬
&batch_normalization_63/batchnorm/mul_1Muloutput/Relu:activations:0(batch_normalization_63/batchnorm/mul:z:0*
T0*'
_output_shapes
:         2(
&batch_normalization_63/batchnorm/mul_1О
&batch_normalization_63/batchnorm/mul_2Mul/batch_normalization_63/moments/Squeeze:output:0(batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_63/batchnorm/mul_2О
/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_63/batchnorm/ReadVariableOpП
$batch_normalization_63/batchnorm/subSub7batch_normalization_63/batchnorm/ReadVariableOp:value:0*batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_63/batchnorm/subр
&batch_normalization_63/batchnorm/add_1AddV2*batch_normalization_63/batchnorm/mul_1:z:0(batch_normalization_63/batchnorm/sub:z:0*
T0*'
_output_shapes
:         2(
&batch_normalization_63/batchnorm/add_1Ь
IdentityIdentity*batch_normalization_63/batchnorm/add_1:z:0;^batch_normalization_60/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_60/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_61/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_61/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_62/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_62/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_63/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*є
_input_shapesu
s:         ::::::::::::::::::::::::2x
:batch_normalization_60/AssignMovingAvg/AssignSubVariableOp:batch_normalization_60/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_60/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_60/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_61/AssignMovingAvg/AssignSubVariableOp:batch_normalization_61/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_61/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_61/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_62/AssignMovingAvg/AssignSubVariableOp:batch_normalization_62/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_62/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_62/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_63/AssignMovingAvg/AssignSubVariableOp:batch_normalization_63/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Њ
ќ
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_1000675

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:г*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:г2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:г2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:г*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:г2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         г2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:г*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:г2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:г*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:г2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         г2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         г:::::P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
░
ф
B__inference_dense2_layer_call_and_return_conditional_losses_999588

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
г╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ╚2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         г:::P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
Ѓ
f
G__inference_dropout_47_layer_call_and_return_conditional_losses_1000862

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
│
Л
%__inference_dnn_layer_call_fn_1000572

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_dnn_layer_call_and_return_conditional_losses_10000692
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*є
_input_shapesu
s:         ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ѓ
e
F__inference_dropout_47_layer_call_and_return_conditional_losses_999708

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
е
Ф
C__inference_output_layer_call_and_return_conditional_losses_1000970

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d:::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Г
ф
B__inference_dense1_layer_call_and_return_conditional_losses_999496

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	г*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         г2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:г*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         г2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         г2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╗
Ф
8__inference_batch_normalization_63_layer_call_fn_1001048

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_63_layer_call_and_return_conditional_losses_9994372
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
═
d
F__inference_dropout_45_layer_call_and_return_conditional_losses_999529

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         г2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         г2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         г:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
╩
e
G__inference_dropout_47_layer_call_and_return_conditional_losses_1000867

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
а)
╦
R__inference_batch_normalization_62_layer_call_and_return_conditional_losses_999297

inputs
assignmovingavg_999272
assignmovingavg_1_999278)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesЈ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:d2
moments/StopGradientц
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         d2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
moments/varianceђ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
moments/Squeezeѕ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
moments/Squeeze_1ъ
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/999272*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_999272*
_output_shapes
:d*
dtype02 
AssignMovingAvg/ReadVariableOp├
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/999272*
_output_shapes
:d2
AssignMovingAvg/sub║
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/999272*
_output_shapes
:d2
AssignMovingAvg/mulЂ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_999272AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/999272*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpц
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/999278*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_999278*
_output_shapes
:d*
dtype02"
 AssignMovingAvg_1/ReadVariableOp═
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/999278*
_output_shapes
:d2
AssignMovingAvg_1/sub─
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/999278*
_output_shapes
:d2
AssignMovingAvg_1/mulЇ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_999278AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/999278*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѓ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         d2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:d2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOpЂ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         d2
batchnorm/add_1х
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         d::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
К)
╬
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1000784

inputs
assignmovingavg_1000759
assignmovingavg_1_1000765)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	╚*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	╚2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ╚2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	╚*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:╚*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:╚*
squeeze_dims
 2
moments/Squeeze_1Ъ
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/1000759*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg/decayЋ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1000759*
_output_shapes	
:╚*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/1000759*
_output_shapes	
:╚2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/1000759*
_output_shapes	
:╚2
AssignMovingAvg/mulЃ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1000759AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/1000759*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЦ
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/1000765*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg_1/decayЏ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1000765*
_output_shapes	
:╚*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¤
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1000765*
_output_shapes	
:╚2
AssignMovingAvg_1/subк
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1000765*
_output_shapes	
:╚2
AssignMovingAvg_1/mulЈ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1000765AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/1000765*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:╚2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:╚2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:╚*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╚2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ╚2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:╚2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:╚*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╚2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╚2
batchnorm/add_1Х
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ╚::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
┴
Ф
8__inference_batch_normalization_61_layer_call_fn_1000830

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_61_layer_call_and_return_conditional_losses_9991902
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ╚::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
И)
╦
R__inference_batch_normalization_61_layer_call_and_return_conditional_losses_999157

inputs
assignmovingavg_999132
assignmovingavg_1_999138)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	╚*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	╚2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ╚2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	╚*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:╚*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:╚*
squeeze_dims
 2
moments/Squeeze_1ъ
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/999132*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg/decayћ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_999132*
_output_shapes	
:╚*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/999132*
_output_shapes	
:╚2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/999132*
_output_shapes	
:╚2
AssignMovingAvg/mulЂ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_999132AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/999132*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpц
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/999138*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg_1/decayџ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_999138*
_output_shapes	
:╚*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/999138*
_output_shapes	
:╚2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/999138*
_output_shapes	
:╚2
AssignMovingAvg_1/mulЇ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_999138AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/999138*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:╚2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:╚2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:╚*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╚2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ╚2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:╚2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:╚*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╚2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╚2
batchnorm/add_1Х
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ╚::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
њ
Ћ
R__inference_batch_normalization_60_layer_call_and_return_conditional_losses_999050

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:г*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:г2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:г2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:г*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:г2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         г2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:г*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:г2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:г*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:г2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         г2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         г:::::P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
јё
Н
 __inference__traced_save_1001279
file_prefix,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop;
7savev2_batch_normalization_60_gamma_read_readvariableop:
6savev2_batch_normalization_60_beta_read_readvariableopA
=savev2_batch_normalization_60_moving_mean_read_readvariableopE
Asavev2_batch_normalization_60_moving_variance_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop;
7savev2_batch_normalization_61_gamma_read_readvariableop:
6savev2_batch_normalization_61_beta_read_readvariableopA
=savev2_batch_normalization_61_moving_mean_read_readvariableopE
Asavev2_batch_normalization_61_moving_variance_read_readvariableop,
(savev2_dense3_kernel_read_readvariableop*
&savev2_dense3_bias_read_readvariableop;
7savev2_batch_normalization_62_gamma_read_readvariableop:
6savev2_batch_normalization_62_beta_read_readvariableopA
=savev2_batch_normalization_62_moving_mean_read_readvariableopE
Asavev2_batch_normalization_62_moving_variance_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop;
7savev2_batch_normalization_63_gamma_read_readvariableop:
6savev2_batch_normalization_63_beta_read_readvariableopA
=savev2_batch_normalization_63_moving_mean_read_readvariableopE
Asavev2_batch_normalization_63_moving_variance_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_dense1_kernel_m_read_readvariableop1
-savev2_adam_dense1_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_60_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_60_beta_m_read_readvariableop3
/savev2_adam_dense2_kernel_m_read_readvariableop1
-savev2_adam_dense2_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_61_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_61_beta_m_read_readvariableop3
/savev2_adam_dense3_kernel_m_read_readvariableop1
-savev2_adam_dense3_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_62_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_62_beta_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_63_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_63_beta_m_read_readvariableop3
/savev2_adam_dense1_kernel_v_read_readvariableop1
-savev2_adam_dense1_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_60_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_60_beta_v_read_readvariableop3
/savev2_adam_dense2_kernel_v_read_readvariableop1
-savev2_adam_dense2_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_61_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_61_beta_v_read_readvariableop3
/savev2_adam_dense3_kernel_v_read_readvariableop1
-savev2_adam_dense3_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_62_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_62_beta_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_63_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_63_beta_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_c24b350d8adb4b0abc30ee93a73541bf/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameџ$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*г#
valueб#BЪ#BB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЈ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*Ў
valueЈBїBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices▀
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop7savev2_batch_normalization_60_gamma_read_readvariableop6savev2_batch_normalization_60_beta_read_readvariableop=savev2_batch_normalization_60_moving_mean_read_readvariableopAsavev2_batch_normalization_60_moving_variance_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop7savev2_batch_normalization_61_gamma_read_readvariableop6savev2_batch_normalization_61_beta_read_readvariableop=savev2_batch_normalization_61_moving_mean_read_readvariableopAsavev2_batch_normalization_61_moving_variance_read_readvariableop(savev2_dense3_kernel_read_readvariableop&savev2_dense3_bias_read_readvariableop7savev2_batch_normalization_62_gamma_read_readvariableop6savev2_batch_normalization_62_beta_read_readvariableop=savev2_batch_normalization_62_moving_mean_read_readvariableopAsavev2_batch_normalization_62_moving_variance_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop7savev2_batch_normalization_63_gamma_read_readvariableop6savev2_batch_normalization_63_beta_read_readvariableop=savev2_batch_normalization_63_moving_mean_read_readvariableopAsavev2_batch_normalization_63_moving_variance_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_dense1_kernel_m_read_readvariableop-savev2_adam_dense1_bias_m_read_readvariableop>savev2_adam_batch_normalization_60_gamma_m_read_readvariableop=savev2_adam_batch_normalization_60_beta_m_read_readvariableop/savev2_adam_dense2_kernel_m_read_readvariableop-savev2_adam_dense2_bias_m_read_readvariableop>savev2_adam_batch_normalization_61_gamma_m_read_readvariableop=savev2_adam_batch_normalization_61_beta_m_read_readvariableop/savev2_adam_dense3_kernel_m_read_readvariableop-savev2_adam_dense3_bias_m_read_readvariableop>savev2_adam_batch_normalization_62_gamma_m_read_readvariableop=savev2_adam_batch_normalization_62_beta_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop>savev2_adam_batch_normalization_63_gamma_m_read_readvariableop=savev2_adam_batch_normalization_63_beta_m_read_readvariableop/savev2_adam_dense1_kernel_v_read_readvariableop-savev2_adam_dense1_bias_v_read_readvariableop>savev2_adam_batch_normalization_60_gamma_v_read_readvariableop=savev2_adam_batch_normalization_60_beta_v_read_readvariableop/savev2_adam_dense2_kernel_v_read_readvariableop-savev2_adam_dense2_bias_v_read_readvariableop>savev2_adam_batch_normalization_61_gamma_v_read_readvariableop=savev2_adam_batch_normalization_61_beta_v_read_readvariableop/savev2_adam_dense3_kernel_v_read_readvariableop-savev2_adam_dense3_bias_v_read_readvariableop>savev2_adam_batch_normalization_62_gamma_v_read_readvariableop=savev2_adam_batch_normalization_62_beta_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableop>savev2_adam_batch_normalization_63_gamma_v_read_readvariableop=savev2_adam_batch_normalization_63_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *P
dtypesF
D2B	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*═
_input_shapes╗
И: :	г:г:г:г:г:г:
г╚:╚:╚:╚:╚:╚:	╚d:d:d:d:d:d:d:::::: : : : : : : : : :	г:г:г:г:
г╚:╚:╚:╚:	╚d:d:d:d:d::::	г:г:г:г:
г╚:╚:╚:╚:	╚d:d:d:d:d:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	г:!

_output_shapes	
:г:!

_output_shapes	
:г:!

_output_shapes	
:г:!

_output_shapes	
:г:!

_output_shapes	
:г:&"
 
_output_shapes
:
г╚:!

_output_shapes	
:╚:!	

_output_shapes	
:╚:!


_output_shapes	
:╚:!

_output_shapes	
:╚:!

_output_shapes	
:╚:%!

_output_shapes
:	╚d: 

_output_shapes
:d: 

_output_shapes
:d: 

_output_shapes
:d: 

_output_shapes
:d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :%"!

_output_shapes
:	г:!#

_output_shapes	
:г:!$

_output_shapes	
:г:!%

_output_shapes	
:г:&&"
 
_output_shapes
:
г╚:!'

_output_shapes	
:╚:!(

_output_shapes	
:╚:!)

_output_shapes	
:╚:%*!

_output_shapes
:	╚d: +

_output_shapes
:d: ,

_output_shapes
:d: -

_output_shapes
:d:$. 

_output_shapes

:d: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
::%2!

_output_shapes
:	г:!3

_output_shapes	
:г:!4

_output_shapes	
:г:!5

_output_shapes	
:г:&6"
 
_output_shapes
:
г╚:!7

_output_shapes	
:╚:!8

_output_shapes	
:╚:!9

_output_shapes	
:╚:%:!

_output_shapes
:	╚d: ;

_output_shapes
:d: <

_output_shapes
:d: =

_output_shapes
:d:$> 

_output_shapes

:d: ?

_output_shapes
:: @

_output_shapes
:: A

_output_shapes
::B

_output_shapes
: 
▒
Ф
C__inference_dense2_layer_call_and_return_conditional_losses_1000712

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
г╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ╚2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         г:::P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
Њ
ќ
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1000804

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:╚*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:╚2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:╚2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:╚*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╚2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ╚2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:╚*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:╚2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:╚*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╚2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╚2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ╚:::::P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
ё
Ћ
R__inference_batch_normalization_63_layer_call_and_return_conditional_losses_999470

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕњ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѕ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         2
batchnorm/mul_1ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Ё
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Ѓ
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         :::::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
а)
╦
R__inference_batch_normalization_63_layer_call_and_return_conditional_losses_999437

inputs
assignmovingavg_999412
assignmovingavg_1_999418)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesЈ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradientц
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/varianceђ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeezeѕ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1ъ
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/999412*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_999412*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp├
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/999412*
_output_shapes
:2
AssignMovingAvg/sub║
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/999412*
_output_shapes
:2
AssignMovingAvg/mulЂ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_999412AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/999412*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpц
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/999418*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_999418*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp═
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/999418*
_output_shapes
:2
AssignMovingAvg_1/sub─
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/999418*
_output_shapes
:2
AssignMovingAvg_1/mulЇ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_999418AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/999418*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѓ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpЂ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         2
batchnorm/add_1х
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
яЋ
ж$
#__inference__traced_restore_1001484
file_prefix"
assignvariableop_dense1_kernel"
assignvariableop_1_dense1_bias3
/assignvariableop_2_batch_normalization_60_gamma2
.assignvariableop_3_batch_normalization_60_beta9
5assignvariableop_4_batch_normalization_60_moving_mean=
9assignvariableop_5_batch_normalization_60_moving_variance$
 assignvariableop_6_dense2_kernel"
assignvariableop_7_dense2_bias3
/assignvariableop_8_batch_normalization_61_gamma2
.assignvariableop_9_batch_normalization_61_beta:
6assignvariableop_10_batch_normalization_61_moving_mean>
:assignvariableop_11_batch_normalization_61_moving_variance%
!assignvariableop_12_dense3_kernel#
assignvariableop_13_dense3_bias4
0assignvariableop_14_batch_normalization_62_gamma3
/assignvariableop_15_batch_normalization_62_beta:
6assignvariableop_16_batch_normalization_62_moving_mean>
:assignvariableop_17_batch_normalization_62_moving_variance%
!assignvariableop_18_output_kernel#
assignvariableop_19_output_bias4
0assignvariableop_20_batch_normalization_63_gamma3
/assignvariableop_21_batch_normalization_63_beta:
6assignvariableop_22_batch_normalization_63_moving_mean>
:assignvariableop_23_batch_normalization_63_moving_variance!
assignvariableop_24_adam_iter#
assignvariableop_25_adam_beta_1#
assignvariableop_26_adam_beta_2"
assignvariableop_27_adam_decay*
&assignvariableop_28_adam_learning_rate
assignvariableop_29_total
assignvariableop_30_count
assignvariableop_31_total_1
assignvariableop_32_count_1,
(assignvariableop_33_adam_dense1_kernel_m*
&assignvariableop_34_adam_dense1_bias_m;
7assignvariableop_35_adam_batch_normalization_60_gamma_m:
6assignvariableop_36_adam_batch_normalization_60_beta_m,
(assignvariableop_37_adam_dense2_kernel_m*
&assignvariableop_38_adam_dense2_bias_m;
7assignvariableop_39_adam_batch_normalization_61_gamma_m:
6assignvariableop_40_adam_batch_normalization_61_beta_m,
(assignvariableop_41_adam_dense3_kernel_m*
&assignvariableop_42_adam_dense3_bias_m;
7assignvariableop_43_adam_batch_normalization_62_gamma_m:
6assignvariableop_44_adam_batch_normalization_62_beta_m,
(assignvariableop_45_adam_output_kernel_m*
&assignvariableop_46_adam_output_bias_m;
7assignvariableop_47_adam_batch_normalization_63_gamma_m:
6assignvariableop_48_adam_batch_normalization_63_beta_m,
(assignvariableop_49_adam_dense1_kernel_v*
&assignvariableop_50_adam_dense1_bias_v;
7assignvariableop_51_adam_batch_normalization_60_gamma_v:
6assignvariableop_52_adam_batch_normalization_60_beta_v,
(assignvariableop_53_adam_dense2_kernel_v*
&assignvariableop_54_adam_dense2_bias_v;
7assignvariableop_55_adam_batch_normalization_61_gamma_v:
6assignvariableop_56_adam_batch_normalization_61_beta_v,
(assignvariableop_57_adam_dense3_kernel_v*
&assignvariableop_58_adam_dense3_bias_v;
7assignvariableop_59_adam_batch_normalization_62_gamma_v:
6assignvariableop_60_adam_batch_normalization_62_beta_v,
(assignvariableop_61_adam_output_kernel_v*
&assignvariableop_62_adam_output_bias_v;
7assignvariableop_63_adam_batch_normalization_63_gamma_v:
6assignvariableop_64_adam_batch_normalization_63_beta_v
identity_66ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_63бAssignVariableOp_64бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9а$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*г#
valueб#BЪ#BB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЋ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*Ў
valueЈBїBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЭ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ъ
_output_shapesІ
ѕ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*P
dtypesF
D2B	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЮ
AssignVariableOpAssignVariableOpassignvariableop_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Б
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2┤
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_60_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3│
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_60_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4║
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_60_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Й
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_60_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ц
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Б
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8┤
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_61_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9│
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_61_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Й
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_61_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11┬
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_61_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Е
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Д
AssignVariableOp_13AssignVariableOpassignvariableop_13_dense3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14И
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_62_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15и
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_62_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Й
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_62_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17┬
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_62_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Е
AssignVariableOp_18AssignVariableOp!assignvariableop_18_output_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Д
AssignVariableOp_19AssignVariableOpassignvariableop_19_output_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20И
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_63_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21и
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_63_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Й
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_63_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23┬
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_63_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_24Ц
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Д
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Д
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27д
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_decayIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28«
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_learning_rateIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29А
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30А
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Б
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Б
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33░
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_dense1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34«
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_dense1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35┐
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_batch_normalization_60_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Й
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adam_batch_normalization_60_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37░
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_dense2_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38«
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_dense2_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39┐
AssignVariableOp_39AssignVariableOp7assignvariableop_39_adam_batch_normalization_61_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Й
AssignVariableOp_40AssignVariableOp6assignvariableop_40_adam_batch_normalization_61_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41░
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_dense3_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42«
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_dense3_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43┐
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_batch_normalization_62_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Й
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_batch_normalization_62_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45░
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_output_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46«
AssignVariableOp_46AssignVariableOp&assignvariableop_46_adam_output_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47┐
AssignVariableOp_47AssignVariableOp7assignvariableop_47_adam_batch_normalization_63_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Й
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adam_batch_normalization_63_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49░
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_dense1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50«
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_dense1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51┐
AssignVariableOp_51AssignVariableOp7assignvariableop_51_adam_batch_normalization_60_gamma_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Й
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_batch_normalization_60_beta_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53░
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_dense2_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54«
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_dense2_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55┐
AssignVariableOp_55AssignVariableOp7assignvariableop_55_adam_batch_normalization_61_gamma_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Й
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_batch_normalization_61_beta_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57░
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_dense3_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58«
AssignVariableOp_58AssignVariableOp&assignvariableop_58_adam_dense3_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59┐
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_62_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Й
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_62_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61░
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_output_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62«
AssignVariableOp_62AssignVariableOp&assignvariableop_62_adam_output_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63┐
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_batch_normalization_63_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Й
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_63_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_649
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЗ
Identity_65Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_65у
Identity_66IdentityIdentity_65:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_66"#
identity_66Identity_66:output:0*Џ
_input_shapesЅ
є: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╝
О
%__inference_dnn_layer_call_fn_1000004
dense1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCalldense1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__inference_dnn_layer_call_and_return_conditional_losses_9999532
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*є
_input_shapesu
s:         ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:         
&
_user_specified_namedense1_input
Й<
З
?__inference_dnn_layer_call_and_return_conditional_losses_999887
dense1_input
dense1_999827
dense1_999829!
batch_normalization_60_999833!
batch_normalization_60_999835!
batch_normalization_60_999837!
batch_normalization_60_999839
dense2_999842
dense2_999844!
batch_normalization_61_999848!
batch_normalization_61_999850!
batch_normalization_61_999852!
batch_normalization_61_999854
dense3_999857
dense3_999859!
batch_normalization_62_999863!
batch_normalization_62_999865!
batch_normalization_62_999867!
batch_normalization_62_999869
output_999872
output_999874!
batch_normalization_63_999877!
batch_normalization_63_999879!
batch_normalization_63_999881!
batch_normalization_63_999883
identityѕб.batch_normalization_60/StatefulPartitionedCallб.batch_normalization_61/StatefulPartitionedCallб.batch_normalization_62/StatefulPartitionedCallб.batch_normalization_63/StatefulPartitionedCallбdense1/StatefulPartitionedCallбdense2/StatefulPartitionedCallбdense3/StatefulPartitionedCallбoutput/StatefulPartitionedCallћ
dense1/StatefulPartitionedCallStatefulPartitionedCalldense1_inputdense1_999827dense1_999829*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_9994962 
dense1/StatefulPartitionedCall 
dropout_45/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_45_layer_call_and_return_conditional_losses_9995292
dropout_45/PartitionedCallй
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCall#dropout_45/PartitionedCall:output:0batch_normalization_60_999833batch_normalization_60_999835batch_normalization_60_999837batch_normalization_60_999839*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_60_layer_call_and_return_conditional_losses_99905020
.batch_normalization_60/StatefulPartitionedCall┐
dense2/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0dense2_999842dense2_999844*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_9995882 
dense2/StatefulPartitionedCall 
dropout_46/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_46_layer_call_and_return_conditional_losses_9996212
dropout_46/PartitionedCallй
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall#dropout_46/PartitionedCall:output:0batch_normalization_61_999848batch_normalization_61_999850batch_normalization_61_999852batch_normalization_61_999854*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_61_layer_call_and_return_conditional_losses_99919020
.batch_normalization_61/StatefulPartitionedCallЙ
dense3/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0dense3_999857dense3_999859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense3_layer_call_and_return_conditional_losses_9996802 
dense3/StatefulPartitionedCall■
dropout_47/PartitionedCallPartitionedCall'dense3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_47_layer_call_and_return_conditional_losses_9997132
dropout_47/PartitionedCall╝
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall#dropout_47/PartitionedCall:output:0batch_normalization_62_999863batch_normalization_62_999865batch_normalization_62_999867batch_normalization_62_999869*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_62_layer_call_and_return_conditional_losses_99933020
.batch_normalization_62/StatefulPartitionedCallЙ
output/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_62/StatefulPartitionedCall:output:0output_999872output_999874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9997722 
output/StatefulPartitionedCall└
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall'output/StatefulPartitionedCall:output:0batch_normalization_63_999877batch_normalization_63_999879batch_normalization_63_999881batch_normalization_63_999883*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_63_layer_call_and_return_conditional_losses_99947020
.batch_normalization_63/StatefulPartitionedCallМ
IdentityIdentity7batch_normalization_63/StatefulPartitionedCall:output:0/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*є
_input_shapesu
s:         ::::::::::::::::::::::::2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:U Q
'
_output_shapes
:         
&
_user_specified_namedense1_input
═
d
F__inference_dropout_46_layer_call_and_return_conditional_losses_999621

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ╚2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ╚2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
фѓ
ѓ
!__inference__wrapped_model_998921
dense1_input-
)dnn_dense1_matmul_readvariableop_resource.
*dnn_dense1_biasadd_readvariableop_resource@
<dnn_batch_normalization_60_batchnorm_readvariableop_resourceD
@dnn_batch_normalization_60_batchnorm_mul_readvariableop_resourceB
>dnn_batch_normalization_60_batchnorm_readvariableop_1_resourceB
>dnn_batch_normalization_60_batchnorm_readvariableop_2_resource-
)dnn_dense2_matmul_readvariableop_resource.
*dnn_dense2_biasadd_readvariableop_resource@
<dnn_batch_normalization_61_batchnorm_readvariableop_resourceD
@dnn_batch_normalization_61_batchnorm_mul_readvariableop_resourceB
>dnn_batch_normalization_61_batchnorm_readvariableop_1_resourceB
>dnn_batch_normalization_61_batchnorm_readvariableop_2_resource-
)dnn_dense3_matmul_readvariableop_resource.
*dnn_dense3_biasadd_readvariableop_resource@
<dnn_batch_normalization_62_batchnorm_readvariableop_resourceD
@dnn_batch_normalization_62_batchnorm_mul_readvariableop_resourceB
>dnn_batch_normalization_62_batchnorm_readvariableop_1_resourceB
>dnn_batch_normalization_62_batchnorm_readvariableop_2_resource-
)dnn_output_matmul_readvariableop_resource.
*dnn_output_biasadd_readvariableop_resource@
<dnn_batch_normalization_63_batchnorm_readvariableop_resourceD
@dnn_batch_normalization_63_batchnorm_mul_readvariableop_resourceB
>dnn_batch_normalization_63_batchnorm_readvariableop_1_resourceB
>dnn_batch_normalization_63_batchnorm_readvariableop_2_resource
identityѕ»
 dnn/dense1/MatMul/ReadVariableOpReadVariableOp)dnn_dense1_matmul_readvariableop_resource*
_output_shapes
:	г*
dtype02"
 dnn/dense1/MatMul/ReadVariableOpЏ
dnn/dense1/MatMulMatMuldense1_input(dnn/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         г2
dnn/dense1/MatMul«
!dnn/dense1/BiasAdd/ReadVariableOpReadVariableOp*dnn_dense1_biasadd_readvariableop_resource*
_output_shapes	
:г*
dtype02#
!dnn/dense1/BiasAdd/ReadVariableOp«
dnn/dense1/BiasAddBiasAdddnn/dense1/MatMul:product:0)dnn/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         г2
dnn/dense1/BiasAddz
dnn/dense1/ReluReludnn/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:         г2
dnn/dense1/Reluљ
dnn/dropout_45/IdentityIdentitydnn/dense1/Relu:activations:0*
T0*(
_output_shapes
:         г2
dnn/dropout_45/IdentityС
3dnn/batch_normalization_60/batchnorm/ReadVariableOpReadVariableOp<dnn_batch_normalization_60_batchnorm_readvariableop_resource*
_output_shapes	
:г*
dtype025
3dnn/batch_normalization_60/batchnorm/ReadVariableOpЮ
*dnn/batch_normalization_60/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2,
*dnn/batch_normalization_60/batchnorm/add/yш
(dnn/batch_normalization_60/batchnorm/addAddV2;dnn/batch_normalization_60/batchnorm/ReadVariableOp:value:03dnn/batch_normalization_60/batchnorm/add/y:output:0*
T0*
_output_shapes	
:г2*
(dnn/batch_normalization_60/batchnorm/addх
*dnn/batch_normalization_60/batchnorm/RsqrtRsqrt,dnn/batch_normalization_60/batchnorm/add:z:0*
T0*
_output_shapes	
:г2,
*dnn/batch_normalization_60/batchnorm/Rsqrt­
7dnn/batch_normalization_60/batchnorm/mul/ReadVariableOpReadVariableOp@dnn_batch_normalization_60_batchnorm_mul_readvariableop_resource*
_output_shapes	
:г*
dtype029
7dnn/batch_normalization_60/batchnorm/mul/ReadVariableOpЫ
(dnn/batch_normalization_60/batchnorm/mulMul.dnn/batch_normalization_60/batchnorm/Rsqrt:y:0?dnn/batch_normalization_60/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:г2*
(dnn/batch_normalization_60/batchnorm/mulР
*dnn/batch_normalization_60/batchnorm/mul_1Mul dnn/dropout_45/Identity:output:0,dnn/batch_normalization_60/batchnorm/mul:z:0*
T0*(
_output_shapes
:         г2,
*dnn/batch_normalization_60/batchnorm/mul_1Ж
5dnn/batch_normalization_60/batchnorm/ReadVariableOp_1ReadVariableOp>dnn_batch_normalization_60_batchnorm_readvariableop_1_resource*
_output_shapes	
:г*
dtype027
5dnn/batch_normalization_60/batchnorm/ReadVariableOp_1Ы
*dnn/batch_normalization_60/batchnorm/mul_2Mul=dnn/batch_normalization_60/batchnorm/ReadVariableOp_1:value:0,dnn/batch_normalization_60/batchnorm/mul:z:0*
T0*
_output_shapes	
:г2,
*dnn/batch_normalization_60/batchnorm/mul_2Ж
5dnn/batch_normalization_60/batchnorm/ReadVariableOp_2ReadVariableOp>dnn_batch_normalization_60_batchnorm_readvariableop_2_resource*
_output_shapes	
:г*
dtype027
5dnn/batch_normalization_60/batchnorm/ReadVariableOp_2­
(dnn/batch_normalization_60/batchnorm/subSub=dnn/batch_normalization_60/batchnorm/ReadVariableOp_2:value:0.dnn/batch_normalization_60/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:г2*
(dnn/batch_normalization_60/batchnorm/subЫ
*dnn/batch_normalization_60/batchnorm/add_1AddV2.dnn/batch_normalization_60/batchnorm/mul_1:z:0,dnn/batch_normalization_60/batchnorm/sub:z:0*
T0*(
_output_shapes
:         г2,
*dnn/batch_normalization_60/batchnorm/add_1░
 dnn/dense2/MatMul/ReadVariableOpReadVariableOp)dnn_dense2_matmul_readvariableop_resource* 
_output_shapes
:
г╚*
dtype02"
 dnn/dense2/MatMul/ReadVariableOpй
dnn/dense2/MatMulMatMul.dnn/batch_normalization_60/batchnorm/add_1:z:0(dnn/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
dnn/dense2/MatMul«
!dnn/dense2/BiasAdd/ReadVariableOpReadVariableOp*dnn_dense2_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02#
!dnn/dense2/BiasAdd/ReadVariableOp«
dnn/dense2/BiasAddBiasAdddnn/dense2/MatMul:product:0)dnn/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
dnn/dense2/BiasAddz
dnn/dense2/ReluReludnn/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:         ╚2
dnn/dense2/Reluљ
dnn/dropout_46/IdentityIdentitydnn/dense2/Relu:activations:0*
T0*(
_output_shapes
:         ╚2
dnn/dropout_46/IdentityС
3dnn/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOp<dnn_batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes	
:╚*
dtype025
3dnn/batch_normalization_61/batchnorm/ReadVariableOpЮ
*dnn/batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2,
*dnn/batch_normalization_61/batchnorm/add/yш
(dnn/batch_normalization_61/batchnorm/addAddV2;dnn/batch_normalization_61/batchnorm/ReadVariableOp:value:03dnn/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes	
:╚2*
(dnn/batch_normalization_61/batchnorm/addх
*dnn/batch_normalization_61/batchnorm/RsqrtRsqrt,dnn/batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes	
:╚2,
*dnn/batch_normalization_61/batchnorm/Rsqrt­
7dnn/batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOp@dnn_batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes	
:╚*
dtype029
7dnn/batch_normalization_61/batchnorm/mul/ReadVariableOpЫ
(dnn/batch_normalization_61/batchnorm/mulMul.dnn/batch_normalization_61/batchnorm/Rsqrt:y:0?dnn/batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╚2*
(dnn/batch_normalization_61/batchnorm/mulР
*dnn/batch_normalization_61/batchnorm/mul_1Mul dnn/dropout_46/Identity:output:0,dnn/batch_normalization_61/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ╚2,
*dnn/batch_normalization_61/batchnorm/mul_1Ж
5dnn/batch_normalization_61/batchnorm/ReadVariableOp_1ReadVariableOp>dnn_batch_normalization_61_batchnorm_readvariableop_1_resource*
_output_shapes	
:╚*
dtype027
5dnn/batch_normalization_61/batchnorm/ReadVariableOp_1Ы
*dnn/batch_normalization_61/batchnorm/mul_2Mul=dnn/batch_normalization_61/batchnorm/ReadVariableOp_1:value:0,dnn/batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes	
:╚2,
*dnn/batch_normalization_61/batchnorm/mul_2Ж
5dnn/batch_normalization_61/batchnorm/ReadVariableOp_2ReadVariableOp>dnn_batch_normalization_61_batchnorm_readvariableop_2_resource*
_output_shapes	
:╚*
dtype027
5dnn/batch_normalization_61/batchnorm/ReadVariableOp_2­
(dnn/batch_normalization_61/batchnorm/subSub=dnn/batch_normalization_61/batchnorm/ReadVariableOp_2:value:0.dnn/batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╚2*
(dnn/batch_normalization_61/batchnorm/subЫ
*dnn/batch_normalization_61/batchnorm/add_1AddV2.dnn/batch_normalization_61/batchnorm/mul_1:z:0,dnn/batch_normalization_61/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╚2,
*dnn/batch_normalization_61/batchnorm/add_1»
 dnn/dense3/MatMul/ReadVariableOpReadVariableOp)dnn_dense3_matmul_readvariableop_resource*
_output_shapes
:	╚d*
dtype02"
 dnn/dense3/MatMul/ReadVariableOp╝
dnn/dense3/MatMulMatMul.dnn/batch_normalization_61/batchnorm/add_1:z:0(dnn/dense3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dnn/dense3/MatMulГ
!dnn/dense3/BiasAdd/ReadVariableOpReadVariableOp*dnn_dense3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02#
!dnn/dense3/BiasAdd/ReadVariableOpГ
dnn/dense3/BiasAddBiasAdddnn/dense3/MatMul:product:0)dnn/dense3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dnn/dense3/BiasAddy
dnn/dense3/ReluReludnn/dense3/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
dnn/dense3/ReluЈ
dnn/dropout_47/IdentityIdentitydnn/dense3/Relu:activations:0*
T0*'
_output_shapes
:         d2
dnn/dropout_47/Identityс
3dnn/batch_normalization_62/batchnorm/ReadVariableOpReadVariableOp<dnn_batch_normalization_62_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype025
3dnn/batch_normalization_62/batchnorm/ReadVariableOpЮ
*dnn/batch_normalization_62/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2,
*dnn/batch_normalization_62/batchnorm/add/yЗ
(dnn/batch_normalization_62/batchnorm/addAddV2;dnn/batch_normalization_62/batchnorm/ReadVariableOp:value:03dnn/batch_normalization_62/batchnorm/add/y:output:0*
T0*
_output_shapes
:d2*
(dnn/batch_normalization_62/batchnorm/add┤
*dnn/batch_normalization_62/batchnorm/RsqrtRsqrt,dnn/batch_normalization_62/batchnorm/add:z:0*
T0*
_output_shapes
:d2,
*dnn/batch_normalization_62/batchnorm/Rsqrt№
7dnn/batch_normalization_62/batchnorm/mul/ReadVariableOpReadVariableOp@dnn_batch_normalization_62_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype029
7dnn/batch_normalization_62/batchnorm/mul/ReadVariableOpы
(dnn/batch_normalization_62/batchnorm/mulMul.dnn/batch_normalization_62/batchnorm/Rsqrt:y:0?dnn/batch_normalization_62/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2*
(dnn/batch_normalization_62/batchnorm/mulр
*dnn/batch_normalization_62/batchnorm/mul_1Mul dnn/dropout_47/Identity:output:0,dnn/batch_normalization_62/batchnorm/mul:z:0*
T0*'
_output_shapes
:         d2,
*dnn/batch_normalization_62/batchnorm/mul_1ж
5dnn/batch_normalization_62/batchnorm/ReadVariableOp_1ReadVariableOp>dnn_batch_normalization_62_batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype027
5dnn/batch_normalization_62/batchnorm/ReadVariableOp_1ы
*dnn/batch_normalization_62/batchnorm/mul_2Mul=dnn/batch_normalization_62/batchnorm/ReadVariableOp_1:value:0,dnn/batch_normalization_62/batchnorm/mul:z:0*
T0*
_output_shapes
:d2,
*dnn/batch_normalization_62/batchnorm/mul_2ж
5dnn/batch_normalization_62/batchnorm/ReadVariableOp_2ReadVariableOp>dnn_batch_normalization_62_batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype027
5dnn/batch_normalization_62/batchnorm/ReadVariableOp_2№
(dnn/batch_normalization_62/batchnorm/subSub=dnn/batch_normalization_62/batchnorm/ReadVariableOp_2:value:0.dnn/batch_normalization_62/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2*
(dnn/batch_normalization_62/batchnorm/subы
*dnn/batch_normalization_62/batchnorm/add_1AddV2.dnn/batch_normalization_62/batchnorm/mul_1:z:0,dnn/batch_normalization_62/batchnorm/sub:z:0*
T0*'
_output_shapes
:         d2,
*dnn/batch_normalization_62/batchnorm/add_1«
 dnn/output/MatMul/ReadVariableOpReadVariableOp)dnn_output_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02"
 dnn/output/MatMul/ReadVariableOp╝
dnn/output/MatMulMatMul.dnn/batch_normalization_62/batchnorm/add_1:z:0(dnn/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dnn/output/MatMulГ
!dnn/output/BiasAdd/ReadVariableOpReadVariableOp*dnn_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dnn/output/BiasAdd/ReadVariableOpГ
dnn/output/BiasAddBiasAdddnn/output/MatMul:product:0)dnn/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dnn/output/BiasAddy
dnn/output/ReluReludnn/output/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dnn/output/Reluс
3dnn/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOp<dnn_batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype025
3dnn/batch_normalization_63/batchnorm/ReadVariableOpЮ
*dnn/batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2,
*dnn/batch_normalization_63/batchnorm/add/yЗ
(dnn/batch_normalization_63/batchnorm/addAddV2;dnn/batch_normalization_63/batchnorm/ReadVariableOp:value:03dnn/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes
:2*
(dnn/batch_normalization_63/batchnorm/add┤
*dnn/batch_normalization_63/batchnorm/RsqrtRsqrt,dnn/batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes
:2,
*dnn/batch_normalization_63/batchnorm/Rsqrt№
7dnn/batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOp@dnn_batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype029
7dnn/batch_normalization_63/batchnorm/mul/ReadVariableOpы
(dnn/batch_normalization_63/batchnorm/mulMul.dnn/batch_normalization_63/batchnorm/Rsqrt:y:0?dnn/batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2*
(dnn/batch_normalization_63/batchnorm/mulя
*dnn/batch_normalization_63/batchnorm/mul_1Muldnn/output/Relu:activations:0,dnn/batch_normalization_63/batchnorm/mul:z:0*
T0*'
_output_shapes
:         2,
*dnn/batch_normalization_63/batchnorm/mul_1ж
5dnn/batch_normalization_63/batchnorm/ReadVariableOp_1ReadVariableOp>dnn_batch_normalization_63_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype027
5dnn/batch_normalization_63/batchnorm/ReadVariableOp_1ы
*dnn/batch_normalization_63/batchnorm/mul_2Mul=dnn/batch_normalization_63/batchnorm/ReadVariableOp_1:value:0,dnn/batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes
:2,
*dnn/batch_normalization_63/batchnorm/mul_2ж
5dnn/batch_normalization_63/batchnorm/ReadVariableOp_2ReadVariableOp>dnn_batch_normalization_63_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype027
5dnn/batch_normalization_63/batchnorm/ReadVariableOp_2№
(dnn/batch_normalization_63/batchnorm/subSub=dnn/batch_normalization_63/batchnorm/ReadVariableOp_2:value:0.dnn/batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2*
(dnn/batch_normalization_63/batchnorm/subы
*dnn/batch_normalization_63/batchnorm/add_1AddV2.dnn/batch_normalization_63/batchnorm/mul_1:z:0,dnn/batch_normalization_63/batchnorm/sub:z:0*
T0*'
_output_shapes
:         2,
*dnn/batch_normalization_63/batchnorm/add_1ѓ
IdentityIdentity.dnn/batch_normalization_63/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*є
_input_shapesu
s:         :::::::::::::::::::::::::U Q
'
_output_shapes
:         
&
_user_specified_namedense1_input
ф
Л
%__inference_dnn_layer_call_fn_1000519

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityѕбStatefulPartitionedCallЋ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__inference_dnn_layer_call_and_return_conditional_losses_9999532
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*є
_input_shapesu
s:         ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┐
Ф
8__inference_batch_normalization_61_layer_call_fn_1000817

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_61_layer_call_and_return_conditional_losses_9991572
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ╚::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
њz
╗
@__inference_dnn_layer_call_and_return_conditional_losses_1000466

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource<
8batch_normalization_60_batchnorm_readvariableop_resource@
<batch_normalization_60_batchnorm_mul_readvariableop_resource>
:batch_normalization_60_batchnorm_readvariableop_1_resource>
:batch_normalization_60_batchnorm_readvariableop_2_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource<
8batch_normalization_61_batchnorm_readvariableop_resource@
<batch_normalization_61_batchnorm_mul_readvariableop_resource>
:batch_normalization_61_batchnorm_readvariableop_1_resource>
:batch_normalization_61_batchnorm_readvariableop_2_resource)
%dense3_matmul_readvariableop_resource*
&dense3_biasadd_readvariableop_resource<
8batch_normalization_62_batchnorm_readvariableop_resource@
<batch_normalization_62_batchnorm_mul_readvariableop_resource>
:batch_normalization_62_batchnorm_readvariableop_1_resource>
:batch_normalization_62_batchnorm_readvariableop_2_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource<
8batch_normalization_63_batchnorm_readvariableop_resource@
<batch_normalization_63_batchnorm_mul_readvariableop_resource>
:batch_normalization_63_batchnorm_readvariableop_1_resource>
:batch_normalization_63_batchnorm_readvariableop_2_resource
identityѕБ
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	г*
dtype02
dense1/MatMul/ReadVariableOpЅ
dense1/MatMulMatMulinputs$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         г2
dense1/MatMulб
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:г*
dtype02
dense1/BiasAdd/ReadVariableOpъ
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         г2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:         г2
dense1/Reluё
dropout_45/IdentityIdentitydense1/Relu:activations:0*
T0*(
_output_shapes
:         г2
dropout_45/Identityп
/batch_normalization_60/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_60_batchnorm_readvariableop_resource*
_output_shapes	
:г*
dtype021
/batch_normalization_60/batchnorm/ReadVariableOpЋ
&batch_normalization_60/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&batch_normalization_60/batchnorm/add/yт
$batch_normalization_60/batchnorm/addAddV27batch_normalization_60/batchnorm/ReadVariableOp:value:0/batch_normalization_60/batchnorm/add/y:output:0*
T0*
_output_shapes	
:г2&
$batch_normalization_60/batchnorm/addЕ
&batch_normalization_60/batchnorm/RsqrtRsqrt(batch_normalization_60/batchnorm/add:z:0*
T0*
_output_shapes	
:г2(
&batch_normalization_60/batchnorm/RsqrtС
3batch_normalization_60/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_60_batchnorm_mul_readvariableop_resource*
_output_shapes	
:г*
dtype025
3batch_normalization_60/batchnorm/mul/ReadVariableOpР
$batch_normalization_60/batchnorm/mulMul*batch_normalization_60/batchnorm/Rsqrt:y:0;batch_normalization_60/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:г2&
$batch_normalization_60/batchnorm/mulм
&batch_normalization_60/batchnorm/mul_1Muldropout_45/Identity:output:0(batch_normalization_60/batchnorm/mul:z:0*
T0*(
_output_shapes
:         г2(
&batch_normalization_60/batchnorm/mul_1я
1batch_normalization_60/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_60_batchnorm_readvariableop_1_resource*
_output_shapes	
:г*
dtype023
1batch_normalization_60/batchnorm/ReadVariableOp_1Р
&batch_normalization_60/batchnorm/mul_2Mul9batch_normalization_60/batchnorm/ReadVariableOp_1:value:0(batch_normalization_60/batchnorm/mul:z:0*
T0*
_output_shapes	
:г2(
&batch_normalization_60/batchnorm/mul_2я
1batch_normalization_60/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_60_batchnorm_readvariableop_2_resource*
_output_shapes	
:г*
dtype023
1batch_normalization_60/batchnorm/ReadVariableOp_2Я
$batch_normalization_60/batchnorm/subSub9batch_normalization_60/batchnorm/ReadVariableOp_2:value:0*batch_normalization_60/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:г2&
$batch_normalization_60/batchnorm/subР
&batch_normalization_60/batchnorm/add_1AddV2*batch_normalization_60/batchnorm/mul_1:z:0(batch_normalization_60/batchnorm/sub:z:0*
T0*(
_output_shapes
:         г2(
&batch_normalization_60/batchnorm/add_1ц
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource* 
_output_shapes
:
г╚*
dtype02
dense2/MatMul/ReadVariableOpГ
dense2/MatMulMatMul*batch_normalization_60/batchnorm/add_1:z:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
dense2/MatMulб
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
dense2/BiasAdd/ReadVariableOpъ
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
dense2/BiasAddn
dense2/ReluReludense2/BiasAdd:output:0*
T0*(
_output_shapes
:         ╚2
dense2/Reluё
dropout_46/IdentityIdentitydense2/Relu:activations:0*
T0*(
_output_shapes
:         ╚2
dropout_46/Identityп
/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes	
:╚*
dtype021
/batch_normalization_61/batchnorm/ReadVariableOpЋ
&batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&batch_normalization_61/batchnorm/add/yт
$batch_normalization_61/batchnorm/addAddV27batch_normalization_61/batchnorm/ReadVariableOp:value:0/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes	
:╚2&
$batch_normalization_61/batchnorm/addЕ
&batch_normalization_61/batchnorm/RsqrtRsqrt(batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes	
:╚2(
&batch_normalization_61/batchnorm/RsqrtС
3batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes	
:╚*
dtype025
3batch_normalization_61/batchnorm/mul/ReadVariableOpР
$batch_normalization_61/batchnorm/mulMul*batch_normalization_61/batchnorm/Rsqrt:y:0;batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╚2&
$batch_normalization_61/batchnorm/mulм
&batch_normalization_61/batchnorm/mul_1Muldropout_46/Identity:output:0(batch_normalization_61/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ╚2(
&batch_normalization_61/batchnorm/mul_1я
1batch_normalization_61/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_61_batchnorm_readvariableop_1_resource*
_output_shapes	
:╚*
dtype023
1batch_normalization_61/batchnorm/ReadVariableOp_1Р
&batch_normalization_61/batchnorm/mul_2Mul9batch_normalization_61/batchnorm/ReadVariableOp_1:value:0(batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes	
:╚2(
&batch_normalization_61/batchnorm/mul_2я
1batch_normalization_61/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_61_batchnorm_readvariableop_2_resource*
_output_shapes	
:╚*
dtype023
1batch_normalization_61/batchnorm/ReadVariableOp_2Я
$batch_normalization_61/batchnorm/subSub9batch_normalization_61/batchnorm/ReadVariableOp_2:value:0*batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╚2&
$batch_normalization_61/batchnorm/subР
&batch_normalization_61/batchnorm/add_1AddV2*batch_normalization_61/batchnorm/mul_1:z:0(batch_normalization_61/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╚2(
&batch_normalization_61/batchnorm/add_1Б
dense3/MatMul/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource*
_output_shapes
:	╚d*
dtype02
dense3/MatMul/ReadVariableOpг
dense3/MatMulMatMul*batch_normalization_61/batchnorm/add_1:z:0$dense3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense3/MatMulА
dense3/BiasAdd/ReadVariableOpReadVariableOp&dense3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
dense3/BiasAdd/ReadVariableOpЮ
dense3/BiasAddBiasAdddense3/MatMul:product:0%dense3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense3/BiasAddm
dense3/ReluReludense3/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
dense3/ReluЃ
dropout_47/IdentityIdentitydense3/Relu:activations:0*
T0*'
_output_shapes
:         d2
dropout_47/IdentityО
/batch_normalization_62/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_62_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype021
/batch_normalization_62/batchnorm/ReadVariableOpЋ
&batch_normalization_62/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&batch_normalization_62/batchnorm/add/yС
$batch_normalization_62/batchnorm/addAddV27batch_normalization_62/batchnorm/ReadVariableOp:value:0/batch_normalization_62/batchnorm/add/y:output:0*
T0*
_output_shapes
:d2&
$batch_normalization_62/batchnorm/addе
&batch_normalization_62/batchnorm/RsqrtRsqrt(batch_normalization_62/batchnorm/add:z:0*
T0*
_output_shapes
:d2(
&batch_normalization_62/batchnorm/Rsqrtс
3batch_normalization_62/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_62_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype025
3batch_normalization_62/batchnorm/mul/ReadVariableOpр
$batch_normalization_62/batchnorm/mulMul*batch_normalization_62/batchnorm/Rsqrt:y:0;batch_normalization_62/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2&
$batch_normalization_62/batchnorm/mulЛ
&batch_normalization_62/batchnorm/mul_1Muldropout_47/Identity:output:0(batch_normalization_62/batchnorm/mul:z:0*
T0*'
_output_shapes
:         d2(
&batch_normalization_62/batchnorm/mul_1П
1batch_normalization_62/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_62_batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype023
1batch_normalization_62/batchnorm/ReadVariableOp_1р
&batch_normalization_62/batchnorm/mul_2Mul9batch_normalization_62/batchnorm/ReadVariableOp_1:value:0(batch_normalization_62/batchnorm/mul:z:0*
T0*
_output_shapes
:d2(
&batch_normalization_62/batchnorm/mul_2П
1batch_normalization_62/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_62_batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype023
1batch_normalization_62/batchnorm/ReadVariableOp_2▀
$batch_normalization_62/batchnorm/subSub9batch_normalization_62/batchnorm/ReadVariableOp_2:value:0*batch_normalization_62/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2&
$batch_normalization_62/batchnorm/subр
&batch_normalization_62/batchnorm/add_1AddV2*batch_normalization_62/batchnorm/mul_1:z:0(batch_normalization_62/batchnorm/sub:z:0*
T0*'
_output_shapes
:         d2(
&batch_normalization_62/batchnorm/add_1б
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
output/MatMul/ReadVariableOpг
output/MatMulMatMul*batch_normalization_62/batchnorm/add_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/MatMulА
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЮ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/BiasAddm
output/ReluReluoutput/BiasAdd:output:0*
T0*'
_output_shapes
:         2
output/ReluО
/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_63/batchnorm/ReadVariableOpЋ
&batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&batch_normalization_63/batchnorm/add/yС
$batch_normalization_63/batchnorm/addAddV27batch_normalization_63/batchnorm/ReadVariableOp:value:0/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_63/batchnorm/addе
&batch_normalization_63/batchnorm/RsqrtRsqrt(batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_63/batchnorm/Rsqrtс
3batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_63/batchnorm/mul/ReadVariableOpр
$batch_normalization_63/batchnorm/mulMul*batch_normalization_63/batchnorm/Rsqrt:y:0;batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_63/batchnorm/mul╬
&batch_normalization_63/batchnorm/mul_1Muloutput/Relu:activations:0(batch_normalization_63/batchnorm/mul:z:0*
T0*'
_output_shapes
:         2(
&batch_normalization_63/batchnorm/mul_1П
1batch_normalization_63/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_63_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_63/batchnorm/ReadVariableOp_1р
&batch_normalization_63/batchnorm/mul_2Mul9batch_normalization_63/batchnorm/ReadVariableOp_1:value:0(batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_63/batchnorm/mul_2П
1batch_normalization_63/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_63_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_63/batchnorm/ReadVariableOp_2▀
$batch_normalization_63/batchnorm/subSub9batch_normalization_63/batchnorm/ReadVariableOp_2:value:0*batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_63/batchnorm/subр
&batch_normalization_63/batchnorm/add_1AddV2*batch_normalization_63/batchnorm/mul_1:z:0(batch_normalization_63/batchnorm/sub:z:0*
T0*'
_output_shapes
:         2(
&batch_normalization_63/batchnorm/add_1~
IdentityIdentity*batch_normalization_63/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*є
_input_shapesu
s:         :::::::::::::::::::::::::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
І
e
F__inference_dropout_45_layer_call_and_return_conditional_losses_999524

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         г2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         г*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         г2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         г2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         г2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*'
_input_shapes
:         г:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
д
e
,__inference_dropout_47_layer_call_fn_1000872

inputs
identityѕбStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_47_layer_call_and_return_conditional_losses_9997082
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*&
_input_shapes
:         d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
И)
╦
R__inference_batch_normalization_60_layer_call_and_return_conditional_losses_999017

inputs
assignmovingavg_998992
assignmovingavg_1_998998)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	г*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	г2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         г2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	г*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:г*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:г*
squeeze_dims
 2
moments/Squeeze_1ъ
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/998992*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg/decayћ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_998992*
_output_shapes	
:г*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/998992*
_output_shapes	
:г2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/998992*
_output_shapes	
:г2
AssignMovingAvg/mulЂ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_998992AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/998992*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpц
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/998998*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg_1/decayџ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_998998*
_output_shapes	
:г*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/998998*
_output_shapes	
:г2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/998998*
_output_shapes	
:г2
AssignMovingAvg_1/mulЇ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_998998AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/998998*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:г2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:г2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:г*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:г2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         г2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:г2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:г*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:г2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         г2
batchnorm/add_1Х
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         г::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
й
Ф
8__inference_batch_normalization_62_layer_call_fn_1000959

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_62_layer_call_and_return_conditional_losses_9993302
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         d::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
ф
ф
B__inference_dense3_layer_call_and_return_conditional_losses_999680

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚:::P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
ё
Ћ
R__inference_batch_normalization_62_layer_call_and_return_conditional_losses_999330

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕњ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѕ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         d2
batchnorm/mul_1ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp_1Ё
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:d2
batchnorm/mul_2ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp_2Ѓ
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         d2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         d:::::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
ї
f
G__inference_dropout_46_layer_call_and_return_conditional_losses_1000733

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ╚2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╚2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ╚2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
»)
╬
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_1001015

inputs
assignmovingavg_1000990
assignmovingavg_1_1000996)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesЈ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradientц
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/varianceђ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeezeѕ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ъ
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/1000990*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg/decayћ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1000990*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/1000990*
_output_shapes
:2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/1000990*
_output_shapes
:2
AssignMovingAvg/mulЃ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1000990AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/1000990*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЦ
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/1000996*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg_1/decayџ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1000996*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1000996*
_output_shapes
:2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1000996*
_output_shapes
:2
AssignMovingAvg_1/mulЈ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1000996AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/1000996*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѓ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpЂ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         2
batchnorm/add_1х
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
юA
с	
?__inference_dnn_layer_call_and_return_conditional_losses_999824
dense1_input
dense1_999507
dense1_999509!
batch_normalization_60_999568!
batch_normalization_60_999570!
batch_normalization_60_999572!
batch_normalization_60_999574
dense2_999599
dense2_999601!
batch_normalization_61_999660!
batch_normalization_61_999662!
batch_normalization_61_999664!
batch_normalization_61_999666
dense3_999691
dense3_999693!
batch_normalization_62_999752!
batch_normalization_62_999754!
batch_normalization_62_999756!
batch_normalization_62_999758
output_999783
output_999785!
batch_normalization_63_999814!
batch_normalization_63_999816!
batch_normalization_63_999818!
batch_normalization_63_999820
identityѕб.batch_normalization_60/StatefulPartitionedCallб.batch_normalization_61/StatefulPartitionedCallб.batch_normalization_62/StatefulPartitionedCallб.batch_normalization_63/StatefulPartitionedCallбdense1/StatefulPartitionedCallбdense2/StatefulPartitionedCallбdense3/StatefulPartitionedCallб"dropout_45/StatefulPartitionedCallб"dropout_46/StatefulPartitionedCallб"dropout_47/StatefulPartitionedCallбoutput/StatefulPartitionedCallћ
dense1/StatefulPartitionedCallStatefulPartitionedCalldense1_inputdense1_999507dense1_999509*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_9994962 
dense1/StatefulPartitionedCallЌ
"dropout_45/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_45_layer_call_and_return_conditional_losses_9995242$
"dropout_45/StatefulPartitionedCall├
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCall+dropout_45/StatefulPartitionedCall:output:0batch_normalization_60_999568batch_normalization_60_999570batch_normalization_60_999572batch_normalization_60_999574*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_60_layer_call_and_return_conditional_losses_99901720
.batch_normalization_60/StatefulPartitionedCall┐
dense2/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0dense2_999599dense2_999601*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_9995882 
dense2/StatefulPartitionedCall╝
"dropout_46/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0#^dropout_45/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_46_layer_call_and_return_conditional_losses_9996162$
"dropout_46/StatefulPartitionedCall├
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall+dropout_46/StatefulPartitionedCall:output:0batch_normalization_61_999660batch_normalization_61_999662batch_normalization_61_999664batch_normalization_61_999666*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_61_layer_call_and_return_conditional_losses_99915720
.batch_normalization_61/StatefulPartitionedCallЙ
dense3/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0dense3_999691dense3_999693*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense3_layer_call_and_return_conditional_losses_9996802 
dense3/StatefulPartitionedCall╗
"dropout_47/StatefulPartitionedCallStatefulPartitionedCall'dense3/StatefulPartitionedCall:output:0#^dropout_46/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_47_layer_call_and_return_conditional_losses_9997082$
"dropout_47/StatefulPartitionedCall┬
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall+dropout_47/StatefulPartitionedCall:output:0batch_normalization_62_999752batch_normalization_62_999754batch_normalization_62_999756batch_normalization_62_999758*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_62_layer_call_and_return_conditional_losses_99929720
.batch_normalization_62/StatefulPartitionedCallЙ
output/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_62/StatefulPartitionedCall:output:0output_999783output_999785*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9997722 
output/StatefulPartitionedCallЙ
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall'output/StatefulPartitionedCall:output:0batch_normalization_63_999814batch_normalization_63_999816batch_normalization_63_999818batch_normalization_63_999820*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_63_layer_call_and_return_conditional_losses_99943720
.batch_normalization_63/StatefulPartitionedCall┬
IdentityIdentity7batch_normalization_63/StatefulPartitionedCall:output:0/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense3/StatefulPartitionedCall#^dropout_45/StatefulPartitionedCall#^dropout_46/StatefulPartitionedCall#^dropout_47/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*є
_input_shapesu
s:         ::::::::::::::::::::::::2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2H
"dropout_45/StatefulPartitionedCall"dropout_45/StatefulPartitionedCall2H
"dropout_46/StatefulPartitionedCall"dropout_46/StatefulPartitionedCall2H
"dropout_47/StatefulPartitionedCall"dropout_47/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:U Q
'
_output_shapes
:         
&
_user_specified_namedense1_input
»)
╬
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_1000913

inputs
assignmovingavg_1000888
assignmovingavg_1_1000894)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesЈ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:d2
moments/StopGradientц
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         d2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
moments/varianceђ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
moments/Squeezeѕ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
moments/Squeeze_1Ъ
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/1000888*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg/decayћ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1000888*
_output_shapes
:d*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/1000888*
_output_shapes
:d2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/1000888*
_output_shapes
:d2
AssignMovingAvg/mulЃ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1000888AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/1000888*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЦ
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/1000894*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
AssignMovingAvg_1/decayџ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1000894*
_output_shapes
:d*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1000894*
_output_shapes
:d2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1000894*
_output_shapes
:d2
AssignMovingAvg_1/mulЈ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1000894AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/1000894*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѓ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         d2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:d2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOpЂ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         d2
batchnorm/add_1х
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         d::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*├
serving_default»
E
dense1_input5
serving_default_dense1_input:0         J
batch_normalization_630
StatefulPartitionedCall:0         tensorflow/serving/predict:ищ
џV
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
к_default_save_signature
+К&call_and_return_all_conditional_losses
╚__call__"ВQ
_tf_keras_sequential═Q{"class_name": "Sequential", "name": "dnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "dnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense1_input"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_45", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_60", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_46", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_61", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_47", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_62", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_63", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "dnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense1_input"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_45", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_60", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_46", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_61", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_47", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_62", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_63", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["MeanSquaredError"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
█

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+╔&call_and_return_all_conditional_losses
╩__call__"┤
_tf_keras_layerџ{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
ж
trainable_variables
regularization_losses
	variables
	keras_api
+╦&call_and_return_all_conditional_losses
╠__call__"п
_tf_keras_layerЙ{"class_name": "Dropout", "name": "dropout_45", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_45", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
и	
axis
	gamma
beta
moving_mean
 moving_variance
!trainable_variables
"regularization_losses
#	variables
$	keras_api
+═&call_and_return_all_conditional_losses
╬__call__"р
_tf_keras_layerК{"class_name": "BatchNormalization", "name": "batch_normalization_60", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_60", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 300}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
▀

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
+¤&call_and_return_all_conditional_losses
л__call__"И
_tf_keras_layerъ{"class_name": "Dense", "name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
ж
+trainable_variables
,regularization_losses
-	variables
.	keras_api
+Л&call_and_return_all_conditional_losses
м__call__"п
_tf_keras_layerЙ{"class_name": "Dropout", "name": "dropout_46", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_46", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
и	
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4trainable_variables
5regularization_losses
6	variables
7	keras_api
+М&call_and_return_all_conditional_losses
н__call__"р
_tf_keras_layerК{"class_name": "BatchNormalization", "name": "batch_normalization_61", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_61", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
▀

8kernel
9bias
:trainable_variables
;regularization_losses
<	variables
=	keras_api
+Н&call_and_return_all_conditional_losses
о__call__"И
_tf_keras_layerъ{"class_name": "Dense", "name": "dense3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
ж
>trainable_variables
?regularization_losses
@	variables
A	keras_api
+О&call_and_return_all_conditional_losses
п__call__"п
_tf_keras_layerЙ{"class_name": "Dropout", "name": "dropout_47", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_47", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
и	
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
+┘&call_and_return_all_conditional_losses
┌__call__"р
_tf_keras_layerК{"class_name": "BatchNormalization", "name": "batch_normalization_62", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_62", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
В

Kkernel
Lbias
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
+█&call_and_return_all_conditional_losses
▄__call__"┼
_tf_keras_layerФ{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
│	
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
+П&call_and_return_all_conditional_losses
я__call__"П
_tf_keras_layer├{"class_name": "BatchNormalization", "name": "batch_normalization_63", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_63", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
Њ
Ziter

[beta_1

\beta_2
	]decay
^learning_ratemдmДmеmЕ%mф&mФ0mг1mГ8m«9m»Cm░Dm▒Km▓Lm│Rm┤SmхvХvиvИv╣%v║&v╗0v╝1vй8vЙ9v┐Cv└Dv┴Kv┬Lv├Rv─Sv┼"
	optimizer
ќ
0
1
2
3
%4
&5
06
17
88
99
C10
D11
K12
L13
R14
S15"
trackable_list_wrapper
 "
trackable_list_wrapper
о
0
1
2
3
4
 5
%6
&7
08
19
210
311
812
913
C14
D15
E16
F17
K18
L19
R20
S21
T22
U23"
trackable_list_wrapper
╬
_layer_metrics
`non_trainable_variables

alayers
bmetrics
trainable_variables
regularization_losses
clayer_regularization_losses
	variables
╚__call__
к_default_save_signature
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
-
▀serving_default"
signature_map
 :	г2dense1/kernel
:г2dense1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
dlayer_metrics
enon_trainable_variables

flayers
gmetrics
trainable_variables
regularization_losses
hlayer_regularization_losses
	variables
╩__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
ilayer_metrics
jnon_trainable_variables

klayers
lmetrics
trainable_variables
regularization_losses
mlayer_regularization_losses
	variables
╠__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)г2batch_normalization_60/gamma
*:(г2batch_normalization_60/beta
3:1г (2"batch_normalization_60/moving_mean
7:5г (2&batch_normalization_60/moving_variance
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
░
nlayer_metrics
onon_trainable_variables

players
qmetrics
!trainable_variables
"regularization_losses
rlayer_regularization_losses
#	variables
╬__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses"
_generic_user_object
!:
г╚2dense2/kernel
:╚2dense2/bias
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
░
slayer_metrics
tnon_trainable_variables

ulayers
vmetrics
'trainable_variables
(regularization_losses
wlayer_regularization_losses
)	variables
л__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
xlayer_metrics
ynon_trainable_variables

zlayers
{metrics
+trainable_variables
,regularization_losses
|layer_regularization_losses
-	variables
м__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)╚2batch_normalization_61/gamma
*:(╚2batch_normalization_61/beta
3:1╚ (2"batch_normalization_61/moving_mean
7:5╚ (2&batch_normalization_61/moving_variance
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
▓
}layer_metrics
~non_trainable_variables

layers
ђmetrics
4trainable_variables
5regularization_losses
 Ђlayer_regularization_losses
6	variables
н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 :	╚d2dense3/kernel
:d2dense3/bias
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
х
ѓlayer_metrics
Ѓnon_trainable_variables
ёlayers
Ёmetrics
:trainable_variables
;regularization_losses
 єlayer_regularization_losses
<	variables
о__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Єlayer_metrics
ѕnon_trainable_variables
Ѕlayers
іmetrics
>trainable_variables
?regularization_losses
 Іlayer_regularization_losses
@	variables
п__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(d2batch_normalization_62/gamma
):'d2batch_normalization_62/beta
2:0d (2"batch_normalization_62/moving_mean
6:4d (2&batch_normalization_62/moving_variance
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
C0
D1
E2
F3"
trackable_list_wrapper
х
їlayer_metrics
Їnon_trainable_variables
јlayers
Јmetrics
Gtrainable_variables
Hregularization_losses
 љlayer_regularization_losses
I	variables
┌__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
:d2output/kernel
:2output/bias
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
х
Љlayer_metrics
њnon_trainable_variables
Њlayers
ћmetrics
Mtrainable_variables
Nregularization_losses
 Ћlayer_regularization_losses
O	variables
▄__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_63/gamma
):'2batch_normalization_63/beta
2:0 (2"batch_normalization_63/moving_mean
6:4 (2&batch_normalization_63/moving_variance
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
R0
S1
T2
U3"
trackable_list_wrapper
х
ќlayer_metrics
Ќnon_trainable_variables
ўlayers
Ўmetrics
Vtrainable_variables
Wregularization_losses
 џlayer_regularization_losses
X	variables
я__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
X
0
 1
22
33
E4
F5
T6
U7"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
Џ0
ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
┐

Юtotal

ъcount
Ъ	variables
а	keras_api"ё
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
щ

Аtotal

бcount
Б
_fn_kwargs
ц	variables
Ц	keras_api"Г
_tf_keras_metricњ{"class_name": "MeanSquaredError", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32"}}
:  (2total
:  (2count
0
Ю0
ъ1"
trackable_list_wrapper
.
Ъ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
А0
б1"
trackable_list_wrapper
.
ц	variables"
_generic_user_object
%:#	г2Adam/dense1/kernel/m
:г2Adam/dense1/bias/m
0:.г2#Adam/batch_normalization_60/gamma/m
/:-г2"Adam/batch_normalization_60/beta/m
&:$
г╚2Adam/dense2/kernel/m
:╚2Adam/dense2/bias/m
0:.╚2#Adam/batch_normalization_61/gamma/m
/:-╚2"Adam/batch_normalization_61/beta/m
%:#	╚d2Adam/dense3/kernel/m
:d2Adam/dense3/bias/m
/:-d2#Adam/batch_normalization_62/gamma/m
.:,d2"Adam/batch_normalization_62/beta/m
$:"d2Adam/output/kernel/m
:2Adam/output/bias/m
/:-2#Adam/batch_normalization_63/gamma/m
.:,2"Adam/batch_normalization_63/beta/m
%:#	г2Adam/dense1/kernel/v
:г2Adam/dense1/bias/v
0:.г2#Adam/batch_normalization_60/gamma/v
/:-г2"Adam/batch_normalization_60/beta/v
&:$
г╚2Adam/dense2/kernel/v
:╚2Adam/dense2/bias/v
0:.╚2#Adam/batch_normalization_61/gamma/v
/:-╚2"Adam/batch_normalization_61/beta/v
%:#	╚d2Adam/dense3/kernel/v
:d2Adam/dense3/bias/v
/:-d2#Adam/batch_normalization_62/gamma/v
.:,d2"Adam/batch_normalization_62/beta/v
$:"d2Adam/output/kernel/v
:2Adam/output/bias/v
/:-2#Adam/batch_normalization_63/gamma/v
.:,2"Adam/batch_normalization_63/beta/v
С2р
!__inference__wrapped_model_998921╗
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *+б(
&і#
dense1_input         
╠2╔
@__inference_dnn_layer_call_and_return_conditional_losses_1000466
@__inference_dnn_layer_call_and_return_conditional_losses_1000367
?__inference_dnn_layer_call_and_return_conditional_losses_999824
?__inference_dnn_layer_call_and_return_conditional_losses_999887└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Р2▀
%__inference_dnn_layer_call_fn_1000572
%__inference_dnn_layer_call_fn_1000120
%__inference_dnn_layer_call_fn_1000519
%__inference_dnn_layer_call_fn_1000004└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ь2Ж
C__inference_dense1_layer_call_and_return_conditional_losses_1000583б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense1_layer_call_fn_1000592б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╠2╔
G__inference_dropout_45_layer_call_and_return_conditional_losses_1000604
G__inference_dropout_45_layer_call_and_return_conditional_losses_1000609┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ќ2Њ
,__inference_dropout_45_layer_call_fn_1000614
,__inference_dropout_45_layer_call_fn_1000619┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
С2р
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_1000655
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_1000675┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
«2Ф
8__inference_batch_normalization_60_layer_call_fn_1000688
8__inference_batch_normalization_60_layer_call_fn_1000701┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ь2Ж
C__inference_dense2_layer_call_and_return_conditional_losses_1000712б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense2_layer_call_fn_1000721б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╠2╔
G__inference_dropout_46_layer_call_and_return_conditional_losses_1000738
G__inference_dropout_46_layer_call_and_return_conditional_losses_1000733┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ќ2Њ
,__inference_dropout_46_layer_call_fn_1000743
,__inference_dropout_46_layer_call_fn_1000748┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
С2р
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1000804
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1000784┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
«2Ф
8__inference_batch_normalization_61_layer_call_fn_1000830
8__inference_batch_normalization_61_layer_call_fn_1000817┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ь2Ж
C__inference_dense3_layer_call_and_return_conditional_losses_1000841б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense3_layer_call_fn_1000850б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╠2╔
G__inference_dropout_47_layer_call_and_return_conditional_losses_1000862
G__inference_dropout_47_layer_call_and_return_conditional_losses_1000867┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ќ2Њ
,__inference_dropout_47_layer_call_fn_1000877
,__inference_dropout_47_layer_call_fn_1000872┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
С2р
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_1000933
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_1000913┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
«2Ф
8__inference_batch_normalization_62_layer_call_fn_1000959
8__inference_batch_normalization_62_layer_call_fn_1000946┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ь2Ж
C__inference_output_layer_call_and_return_conditional_losses_1000970б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_output_layer_call_fn_1000979б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
С2р
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_1001015
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_1001035┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
«2Ф
8__inference_batch_normalization_63_layer_call_fn_1001061
8__inference_batch_normalization_63_layer_call_fn_1001048┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
9B7
%__inference_signature_wrapper_1000183dense1_input╚
!__inference__wrapped_model_998921б %&302189FCEDKLURTS5б2
+б(
&і#
dense1_input         
ф "OфL
J
batch_normalization_630і-
batch_normalization_63         ╗
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_1000655d 4б1
*б'
!і
inputs         г
p
ф "&б#
і
0         г
џ ╗
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_1000675d 4б1
*б'
!і
inputs         г
p 
ф "&б#
і
0         г
џ Њ
8__inference_batch_normalization_60_layer_call_fn_1000688W 4б1
*б'
!і
inputs         г
p
ф "і         гЊ
8__inference_batch_normalization_60_layer_call_fn_1000701W 4б1
*б'
!і
inputs         г
p 
ф "і         г╗
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1000784d23014б1
*б'
!і
inputs         ╚
p
ф "&б#
і
0         ╚
џ ╗
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1000804d30214б1
*б'
!і
inputs         ╚
p 
ф "&б#
і
0         ╚
џ Њ
8__inference_batch_normalization_61_layer_call_fn_1000817W23014б1
*б'
!і
inputs         ╚
p
ф "і         ╚Њ
8__inference_batch_normalization_61_layer_call_fn_1000830W30214б1
*б'
!і
inputs         ╚
p 
ф "і         ╚╣
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_1000913bEFCD3б0
)б&
 і
inputs         d
p
ф "%б"
і
0         d
џ ╣
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_1000933bFCED3б0
)б&
 і
inputs         d
p 
ф "%б"
і
0         d
џ Љ
8__inference_batch_normalization_62_layer_call_fn_1000946UEFCD3б0
)б&
 і
inputs         d
p
ф "і         dЉ
8__inference_batch_normalization_62_layer_call_fn_1000959UFCED3б0
)б&
 і
inputs         d
p 
ф "і         d╣
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_1001015bTURS3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ ╣
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_1001035bURTS3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ Љ
8__inference_batch_normalization_63_layer_call_fn_1001048UTURS3б0
)б&
 і
inputs         
p
ф "і         Љ
8__inference_batch_normalization_63_layer_call_fn_1001061UURTS3б0
)б&
 і
inputs         
p 
ф "і         ц
C__inference_dense1_layer_call_and_return_conditional_losses_1000583]/б,
%б"
 і
inputs         
ф "&б#
і
0         г
џ |
(__inference_dense1_layer_call_fn_1000592P/б,
%б"
 і
inputs         
ф "і         гЦ
C__inference_dense2_layer_call_and_return_conditional_losses_1000712^%&0б-
&б#
!і
inputs         г
ф "&б#
і
0         ╚
џ }
(__inference_dense2_layer_call_fn_1000721Q%&0б-
&б#
!і
inputs         г
ф "і         ╚ц
C__inference_dense3_layer_call_and_return_conditional_losses_1000841]890б-
&б#
!і
inputs         ╚
ф "%б"
і
0         d
џ |
(__inference_dense3_layer_call_fn_1000850P890б-
&б#
!і
inputs         ╚
ф "і         dЙ
@__inference_dnn_layer_call_and_return_conditional_losses_1000367z %&230189EFCDKLTURS7б4
-б*
 і
inputs         
p

 
ф "%б"
і
0         
џ Й
@__inference_dnn_layer_call_and_return_conditional_losses_1000466z %&302189FCEDKLURTS7б4
-б*
 і
inputs         
p 

 
ф "%б"
і
0         
џ ─
?__inference_dnn_layer_call_and_return_conditional_losses_999824ђ %&230189EFCDKLTURS=б:
3б0
&і#
dense1_input         
p

 
ф "%б"
і
0         
џ ─
?__inference_dnn_layer_call_and_return_conditional_losses_999887ђ %&302189FCEDKLURTS=б:
3б0
&і#
dense1_input         
p 

 
ф "%б"
і
0         
џ ю
%__inference_dnn_layer_call_fn_1000004s %&230189EFCDKLTURS=б:
3б0
&і#
dense1_input         
p

 
ф "і         ю
%__inference_dnn_layer_call_fn_1000120s %&302189FCEDKLURTS=б:
3б0
&і#
dense1_input         
p 

 
ф "і         ќ
%__inference_dnn_layer_call_fn_1000519m %&230189EFCDKLTURS7б4
-б*
 і
inputs         
p

 
ф "і         ќ
%__inference_dnn_layer_call_fn_1000572m %&302189FCEDKLURTS7б4
-б*
 і
inputs         
p 

 
ф "і         Е
G__inference_dropout_45_layer_call_and_return_conditional_losses_1000604^4б1
*б'
!і
inputs         г
p
ф "&б#
і
0         г
џ Е
G__inference_dropout_45_layer_call_and_return_conditional_losses_1000609^4б1
*б'
!і
inputs         г
p 
ф "&б#
і
0         г
џ Ђ
,__inference_dropout_45_layer_call_fn_1000614Q4б1
*б'
!і
inputs         г
p
ф "і         гЂ
,__inference_dropout_45_layer_call_fn_1000619Q4б1
*б'
!і
inputs         г
p 
ф "і         гЕ
G__inference_dropout_46_layer_call_and_return_conditional_losses_1000733^4б1
*б'
!і
inputs         ╚
p
ф "&б#
і
0         ╚
џ Е
G__inference_dropout_46_layer_call_and_return_conditional_losses_1000738^4б1
*б'
!і
inputs         ╚
p 
ф "&б#
і
0         ╚
џ Ђ
,__inference_dropout_46_layer_call_fn_1000743Q4б1
*б'
!і
inputs         ╚
p
ф "і         ╚Ђ
,__inference_dropout_46_layer_call_fn_1000748Q4б1
*б'
!і
inputs         ╚
p 
ф "і         ╚Д
G__inference_dropout_47_layer_call_and_return_conditional_losses_1000862\3б0
)б&
 і
inputs         d
p
ф "%б"
і
0         d
џ Д
G__inference_dropout_47_layer_call_and_return_conditional_losses_1000867\3б0
)б&
 і
inputs         d
p 
ф "%б"
і
0         d
џ 
,__inference_dropout_47_layer_call_fn_1000872O3б0
)б&
 і
inputs         d
p
ф "і         d
,__inference_dropout_47_layer_call_fn_1000877O3б0
)б&
 і
inputs         d
p 
ф "і         dБ
C__inference_output_layer_call_and_return_conditional_losses_1000970\KL/б,
%б"
 і
inputs         d
ф "%б"
і
0         
џ {
(__inference_output_layer_call_fn_1000979OKL/б,
%б"
 і
inputs         d
ф "і         ▄
%__inference_signature_wrapper_1000183▓ %&302189FCEDKLURTSEбB
б 
;ф8
6
dense1_input&і#
dense1_input         "OфL
J
batch_normalization_630і-
batch_normalization_63         