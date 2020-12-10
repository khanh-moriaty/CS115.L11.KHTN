	.data


train_image:		.asciiz			"train-images.idx3-ubyte"			# Images in MNIST trainset
train_label:		.asciiz			"train-labels.idx1-ubyte"			# Labels in MNIST trainset
test_image:			.asciiz			"t10k-images.idx3-ubyte"			# Images in MNIST testset
test_label:			.asciiz			"t10k-labels.idx1-ubyte"			# Labels in MNIST testset	
N_train:			.word			60000								# Number of train samples
N_test:				.word			10000								# Number of train samples
N_classes:			.word			10									# Number of classes
N_dim:				.word			784									# Input image dimensions

out_name:			.space			10									# Output file name

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
	
	la			$a0, train_label
	jal			open_file
	
	jal			read_16byte					# Skip 16 bytes of MNIST which contains metadata
	
	jal			allocate_images

	li			$v0, 5
	syscall
	
	jal			exit



# Function: Allocate an array for MNIST dataset
# Used registers: $a0-3, $s1-3, $t0-2
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

		move		$t3, $v0
		move		$a0, $v0
		li			$v0, 34
		syscall
		li			$v0, 11
		li			$a0, 10
		syscall
		move		$v0, $t3

		sll			$a0, $t0, 0x02				# byte offset of last element in array (bias)
		add			$a0, $v0, $0				# $a0 = address of X->[i]->[-1]
		lwc1		$f1, 4($a1)					# $f1 = bias = 1.0
		swc1		$f1, 0($a0)					# Bias is always == 1.0

		addi		$t2, $t2, 1
		bne			$t2, $t0, loop_allocate_images
	
	jr			$ra


# Function: Load dataset into memory
# Parameters:
# $a3: Address of image in array
# $t0: N_dim
# $s0: fd of dataset file
# Used registers: $a0-3, $s1-3, $t0-2
# Author: moriaty
load_dataset:
	or			$s3, $0, $ra				# Store return address into $s3
	lw			$t0, N_dim

	jal			load_img


# Function: Read an 28x28 image from dataset and allocate heap memory for it
# Parameters:
# $a3: Address of image
# $t0: N_dim
# $s0: fd of dataset file
# Used registers: $a0-3, $s1-2, $t1-2
# Author: moriaty
load_img:
	or			$s2, $0, $ra				# Store return address into $s2

	or			$t1, $0, $0					# Initialize iterating variable
	loop_load_img:
		jal		read_1byte					# Read 1 byte from file
		lw		$t2, buffer					# Read from buffer
		sll		$a0, $t1, 0x02				# Calculate byte offset in array
		add		$a0, $a0, $a3				# $a0 = address of pixel in image
		sw		$t2, 0($a0)					# Store the pixel in image

		addi	$t1, $t1, 1
		bne		$t1, $t0, loop_load_img		# Next iteration

	jr			$s2

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
	
	jr		$ra
	
	
# Function: read 1 byte and store it in buffer
# Parameters:
# $s0 = File Descriptor
# Return the read byte inside buffer data.
# Used registers: $s1, $a0-2
# Author: moriaty
read_1byte:
	or		$s1, $0, $ra			# Store return address into $s1
	
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
	
