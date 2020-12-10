	.data


train_image:		.asciiz			"train-images.idx3-ubyte"			# Images in MNIST trainset
train_label:		.asciiz			"train-labels.idx1-ubyte"			# Labels in MNIST trainset
test_image:			.asciiz			"t10k-images.idx3-ubyte"			# Images in MNIST testset
test_label:			.asciiz			"t10k-labels.idx1-ubyte"			# Labels in MNIST testset	
N_train:			.word			60000								# Number of train samples
N_test:				.word			10000								# Number of train samples
N_classes:			.word			10									# Number of classes
N_dim:				.word			784									# Input image dimensions

epsilon:			.float			0.000001							# 1e-6, reserved for calculations
learning_rate:		.float			0.0002								# 2e-4, reserved for calculations

buffer:				.space			0x10								# Buffer memory for file reading	
print_iter:			.asciiz			"Iteration: "
print_loss:			.asciiz			". Loss: "
									
test_exp:			.float			100

X:					.word			0									# Address of array of input images
Y:					.word  			0									# Address of array of labels
w:					.word  			0									# Address of weights
x:					.word			0									# Address of temporary x

#####################################################################
# FLOATS RESERVED FOR FASTER COMPUTING ##############################
#####################################################################

float_const:		.float			0,	1,	2,	3,	4,	5,	6,	7, 
									8,	9,	10,	11,	12,	13,	14,	15, 
									16,	17,	18,	19,	20,	21,	22,	23,
									24,	25,	26,	27,	28,	29,	30,	31, 
									-1,	-2,	-3,	-4,	-5,	-6,	-7,	-8,
									255

######################################################################
									
	.text
main:

	or			$a0, $0, $0				
	ori			$a1, $0, 19520624			# Set random seed to my ID
	ori			$v0, $0, 0x28				# 0x28 = set random seed
	syscall
	
	la			$a0, train_image
	jal			open_file
	
	jal			read_16byte					# Skip 16 bytes of MNIST which contains metadata
	
	jal			allocate_images

	lw			$t4, N_train
	jal			load_dataset

	lw			$a1, N_dim
	sll			$a1, $a1, 0x02
	lw			$a2, X
	lw			$a2, 0($a2)
	jal			print_data

	jal			exit



# Function: Load dataset into memory
# Parameters:
# $s0: fd of dataset file
# $t4: number of images
# Used registers: $a0-3, $s1-3, $t0-3, $f0-1
# Author: moriaty
load_dataset:
	or			$s3, $0, $ra				# Store return address into $s3
	
	la			$t0, float_const			# #######################
	lwc1		$f1, 160($t0)				# Load $f1 = 255.0
	lw			$t0, N_dim					# Load $t0 = N_dim = 784

	or			$t3, $0, $0					# Initialize iterating variable

	loop_load_dataset:

		lw			$a3, X
		lw			$a3, 0($a3)
		jal			load_img

		and			$t4, $t3, 1023
		bnez		$t4, no_print_load_dataset

		li			$v0, 1
		move		$a0, $t3
		syscall
		li			$v0, 11
		li			$a0, 10
		syscall

		no_print_load_dataset:

		addi		$t3, $t3, 1
		bne			$t3, $t4, loop_load_dataset

	jr			$s3


# Function: Read an 28x28 image from dataset and store it to heap memory
# Parameters:
# $a3: Address of image
# $t0: N_dim
# $s0: fd of dataset file
# $f1: 255.0 (constant for normalize)
# Used registers: $a0-3, $s1-2, $t1-2, $f0
# Author: moriaty
load_img:
	or			$s2, $0, $ra					# Store return address into $s2

	or			$t1, $0, $0						# Initialize iterating variable
	loop_load_img:
		jal			read_1byte					# Read 1 byte from file
		lw			$t2, buffer					# Read from buffer
		mtc1		$t2, $f0					# Move buffer value to coproc1
		cvt.s.w		$f0, $f0					# Convert 32-bit int to single floating point representation
		div.s		$f0, $f0, $f1				# Divide by 255 (normalize)

		sll			$a0, $t1, 0x02				# Calculate byte offset in array
		add			$a0, $a0, $a3				# $a0 = address of pixel in image
		swc1		$f0, 0($a0)					# Store the pixel in image

		addi		$t1, $t1, 1
		bne			$t1, $t0, loop_load_img		# Next iteration

	jr			$s2

# Function: Compute e^x using Taylor Series, if x is negative, compute 1/e^(-x) instead
# Parameters:
# $f1 = x
# Return:
# $f2 = e^x
# Used registers: $f2-4, $a0-3
# Author: moriaty
exp:
	or			$a3, $0, $ra				# Store return address into $s1
	la			$a0, float_const			# $a2 is the address of float constant array (used for factorial computing)
	
	cvt.w.s		$f3, $f1					# ##############################################
	mfc1		$a1, $f3					# If x is negative then compute 1/e^(-x) instead
	blt			$a1, $0, exp_negative		# ##############################################

	jal			exp_positive	
	jr			$a3
	
	exp_negative:
		lwc1	$f3, 128($a0)				# $f3 = -1
		mul.s	$f1, $f1, $f3				# $f1 = -x
		jal		exp_positive
		lwc1	$f3, 4($a0)					# $f3 = 1
		div.s	$f2, $f3, $f2				# $f2 = 1/e^(-x)
		jr		$a3
	
	

# Function: Compute e^x using Taylor Series, x must be positive
# Parameters:
# $a0 = address of constant floats
# $f1 = x
# Return:
# $f2 = e^x, $f3 = x^5
# Used registers: $f2-4, $a1-2
# Author: moriaty
exp_positive:
	or			$a2, $0, $a0
	lwc1		$f3, 4($a2)					# Initialize $f3 = 1
	
	ori			$a1, $0, 0x04				# ###################
	mul			$a1, $a1, 30				# Max iterations = 30
	add			$a1, $a1, $a2				# ###################
	
	mov.s		$f2, $f3					# $f2 = $f3 = 1
	add.s		$f2, $f2, $f1				# $f2 = 1 + x
	
	mov.s		$f3, $f1					# $f3 = x
	
	loop_exp:
	
		mul.s		$f3, $f3, $f1			# $f3 = x^i/(i-1)!
		lwc1		$f4, 8($a2)				# $f4 = i
		div.s		$f3, $f3, $f4			# $f3 = x^i/i!
		add.s		$f2, $f2, $f3			# Approximating e^x by adding $f3 into $f2
	
		addiu		$a2, $a2, 4				# Compute the next approximation degree
		bne			$a2, $a1, loop_exp		# Next iteration

	jr			$ra


# Function: Allocate an array for MNIST dataset
# Used registers: $a0-3, $s1-3, $t0-2, $f1
# Author: moriaty
allocate_images:
	lw			$t0, N_train
	lw			$t1, N_dim
	la			$a1, float_const

	ori			$v0, $0, 0x09					# 0x09 = call sbrk
	sll			$a0, $t0, 0x02					# Allocate N * 4 bytes
	syscall										# Allocate images array
	la			$a0, X
	sw			$v0, 0($a0)						# Save pointer to X
	or			$s1, $0, $v0					# $s1 = pointer to images array

	or			$t2, $0, $0
	loop_allocate_images:
		ori			$v0, $0, 0x09				# 0x09 = call sbrk
		addi		$a0, $t1, 1					# Allocate image with bias added
		sll			$a0, $a0, 0x02				# (N + 1) * 4 bytes
		syscall
		sll			$a0, $t2, 0x02				# byte offset = 4 * i
		add			$a0, $a0, $s1				# $a0 = address of X->[i]
		sw			$v0, 0($a0)					# Save pointer to X->[i]

		sll			$a0, $t0, 0x02				# byte offset of last element in array (bias)
		add			$a0, $v0, $0				# $a0 = address of X->[i]->[-1]
		lwc1		$f1, 4($a1)					# $f1 = bias = 1.0
		swc1		$f1, 0($a0)					# Bias is always == 1.0

		addi		$t2, $t2, 1
		bne			$t2, $t0, loop_allocate_images
	
	jr			$ra


# Function: print an array of N elements
# Parameters:
# $a1 = N
# $a2 = address of the array
# Used registers: $t0
# Author: moriaty
print_data:
	
	li		$t0, 0
	
	loop_print_data:
		
		li		$v0, 2
		add		$at, $a2, $t0
		lwc1	$f12, ($at)
		syscall		
		li		$v0, 0x0B
		li		$a0, 0x20
		syscall
		
		addi	$t0, $t0, 4
		bne		$t0, $a1, loop_print_data
		
	li		$v0, 0x0B
	li		$a0, 0x0A
	syscall
	
	jr		$ra
	
	
# Function: Print byte stored in buffer
# Used registers: $a0-1
# Author: moriaty
print_buffer:
	li		$v0, 0x22		# 0x22 = print int hexa
	lw		$a0, buffer		# buffer address
	syscall					# print int
	
	li		$v0, 0x0B		# 0x0B = print char
	li		$a0, 0x0A		# 0x0A = LF char
	syscall
	
	jr		$ra


# Function: read some bytes and store it in buffer
# Parameters:
# $s0 = File Descriptor
# $a2 = number of bytes to read
# Return the read bytes inside buffer data.
# Used registers: $a0-1
# Author: moriaty
read_bytes:
	ori		$v0, $0, 0x0E			# 0x0E = read from  file
	or		$a0, $0, $s0			# $s0 contains fd
	la		$a1, buffer				# buffer to hold int
	syscall
	
	jr		$ra
	
	
# Function: read 1 byte and store it in buffer
# Parameters:
# $s0 = File Descriptor
# Return the read byte inside buffer data.
# Used registers: $s1, $a0-2
# Author: moriaty
read_1byte:
	or		$s1, $0, $ra			# Store return address into $s1
	
	sw		$0, buffer
	ori		$a2, $0, 0x01			# Set $a2 = 1 = number of bytes read
	jal		read_bytes				# Call read_bytes 
									# => read byte is stored in buffer
	jr		$s1
	
	
# Function: read 4 bytes and store it in buffer
# Parameters:
# $s0 = File Descriptor
# Return the read bytes inside buffer data.
# Used registers: $s1, $a0-2
# Author: moriaty
read_16byte:
	or		$s1, $0, $ra			# Store return address into $s1
	
	ori		$a2, $0, 0x10			# Set $a2 = 16 = number of bytes read
	jal		read_bytes				# Call read_bytes 
									# => read byte is stored in buffer
	jr		$s1
	
	
# Function: open a file and returns its File Description
# Parameters:
# $a0 = file name
# Return $s0 = file description
# Used registers: $a1-2
# Author: moriaty
open_file:
	ori		$v0, $0, 0x0D				# 0x0D = open file
	or		$a1, $0, $0					# $a1=flags=O_RDONLY=0
	or		$a2, $0, $0					# $a2=mode=0
	syscall								# Open File, $v0<-fd
	or		$s0, $0, $v0				# Store fd in $s0

	jr		$ra


# Function: open a file and returns its File Description
# Parameters:
# $s0 = file description
# Used registers: $a0
# Author: moriaty
close_file:
	ori		$v0, $0, 0x10				# 0x10 = close file
	or		$a0, $0, $s0				# $s0 contains fd
	syscall								# close file
	
	jr		$ra


# Exit the program
exit:
	ori		$v0, $0, 0x0A
	syscall
	
