# Read and echo file on a character by character basis
# David A. Reimann
# April 2008



	.data

file:
	.asciiz	"diem_nmlt.csv"	# File name
	.word	0
	
buffer:			.space	1			# Place to store character
N:				.word	93			# N
n:				.word	372			# dataset consists of 93 records => n = 93 * 4 = 372 bytes
x:				.space	372			# 
y:				.space  372			# 
	
print_iter:		.asciiz	"Iteration: "
print_loss:		.asciiz	". Loss: "
print_answ:		.asciiz	". Weights: ("

const_half:		.float	0.5			# 1/2, reserved for calculations
epsilon:		.float	0.000001	# 1e-6, reserved for calculations
learning_rate:	.float	0.0002		# 2e-4, reserved for calculations

	.text
main:

	li		$a0, 0				
	li		$a1, 19520624				# Set random seed to my ID
	li		$v0, 40						# :)
	syscall

	lw		$s4, n						# $s4 = number of lines
	la		$s5, x						# $s5 = address of x
	la		$s6, y						# $s6 = address of y
	jal		read_csv

	lw		$s0, N						# $s4 = number of lines
	la		$s1, x						# $s5 = address of x
	la		$s2, y						# $s6 = address of y
	lwc1	$f2, learning_rate
	lwc1	$f3, epsilon
	jal		gradient_descent
	
	jal		exit
	
# Function: Stochastic Gradient Descent algorithm
# Parameters:
# $s0: N
# $s1: address of x
# $s2: address of y
# $f2: learning rate
# $f3: epsilon (for derivative calculation)
# Return: $f0 and $f1, which are the solution for the Linear Regression model. (w_0 and w_1, respectively)
# Used registers: $t0
# Author: moriaty
gradient_descent:

	move	$s3, $ra				# Store return address into $s3
	
	
	###################################################################################
	#########################    MODEL INITIALIZATION    ##############################
	###################################################################################
	
	add		$a0, $0, $0				###########################
	li		$v0, 43					# Generate a random value # <-- Stored in $f6
	syscall							#        for w_0          #
	mov.s	$f6, $f0				###########################
	
	add		$a0, $0, $0				###########################
	li		$v0, 43					# Generate a random value # <-- Stored in $f7
	syscall							#        for w_1          #
	mov.s	$f7, $f0				###########################
	
	mov.s	$f0, $f6				# Move $f6 to $f0
	mov.s	$f1, $f7				# Move $f7 to $f1
	
	###################################################################################
	#########################  DONE MODEL INITIALIZATION ##############################
	###################################################################################

	lwc1	$f9, const_half			# $f10 = 1/2
	li		$t1, 0
	
	loop_gradient_descent:
		
		addi	$t1, $t1, 1						# Increase the iters count
	
		add		$a0, $0, $0
		move	$a1, $s0						###########################
		li		$v0, 42							# Generate a random index # <-- Stored in $t0
		syscall									#        for SGD          #
		sll		$t0, $a0, 2	# Multiplies by 4	###########################
		
		add		$at, $s1, $t0					# Address of x[i]
		lwc1	$f4, ($at)
		
		add		$at, $s2, $t0					# Address of y[i]
		lwc1	$f5, ($at)
		
		mov.s	$f6, $f0						# Copy w_0 to $f6 for derivative calculation
		mov.s	$f7, $f1						# Copy w_1 to $f7 for derivative calculation
		
		jal		derivative						# Call derivative calculation function
		
		mul.s	$f10, $f10, $f2					####################################
		sub.s	$f0, $f0, $f10					# Update w_0 <- w_0 - lr * L'(w_0) #
												####################################
										
		mul.s	$f11, $f11, $f2					####################################
		sub.s	$f1, $f1, $f11					# Update w_1 <- w_1 - lr * L'(w_1) #
												####################################
		
		li		$at, 0xfff						# If iters count is  
		and		$at, $t1, $at					# divisible by 4096 
		bnez	$at, loop_gradient_descent		# then start printing log...
		
		############################################################################################
		############################           PRINT LOG          ##################################
		############################################################################################
		
		
		mov.s	$f6, $f0						# Copy w_0 to $f6 for loss calculation
		mov.s	$f7, $f1						# Copy w_1 to $f7 for loss calculation
		jal		loss							# Re-evaluate loss value after updating weights
		
		la		$a0, print_iter
		li		$v0, 4
		syscall
		li		$v0, 1
		move	$a0, $t1
		syscall
		
		la		$a0, print_loss
		li		$v0, 4
		syscall
		li		$v0, 2
		mov.s	$f12, $f8
		syscall
		
		la		$a0, print_answ
		li		$v0, 4
		syscall
		li		$v0, 2
		mov.s	$f12, $f0
		syscall
		
		li		$v0, 11
		li		$a0, 0x2C
		syscall
		li		$v0, 11
		li		$a0, 0x20
		syscall
		
		li		$v0, 2
		mov.s	$f12, $f1
		syscall
		
		li		$v0, 11
		li		$a0, 0x29
		syscall
		li		$v0, 11
		li		$a0, 0x0A
		syscall
		
		j		loop_gradient_descent
		
		############################################################################################
		############################         END PRINT LOG        ##################################
		############################################################################################
		
		
	jr		$s3

# Function: Calculate the derivative of loss function by differentiation L'(w)
# Parameters:
# $f3 = epsilon
# $f4 = x
# $f5 = y
# $f6, $f7 = w_0, w_1
# $f9 = 1/2
# Return: $f10, $f11 = L'(w)
# Used registers: $f6-7, $f14-15, $s4
derivative:
	move		$s4, $ra						# Store the return address into $s4

	mov.s		$f14, $f6						# Temporary holder for w_0
	mov.s		$f15, $f7						# Temporary holder for w_1
	
	add.s		$f6, $f14, $f3					#######################
	mov.s		$f7, $f15						#      CALCULATE      # <-- Stored in $f11
	jal			loss							# L(w_0+epsilon, w_1) #
	mov.s 		$f11, $f8						#######################
	
	sub.s		$f6, $f14, $f3					#######################
	# mov.s		$f7, $f15						#       CALCULATE     # <-- Stored in $f10
	jal			loss							# L(w_0+epsilon, w_1) #
	mov.s 		$f10, $f8						#######################
	
	sub.s		$f10, $f11, $f10				# $f10 = L(w_0+epsilon, w_1) - L(w_0-epsilon, w_1)
	mul.s		$f10, $f9, $f10					# $f10 = $f10 / 2
	div.s		$f10, $f10, $f3					# $f10 = $f10 / epsilon
	
	mov.s		$f6, $f14						#######################
	add.s		$f7, $f15, $f3					#      CALCULATE      # <-- Stored in $f12
	jal			loss							# L(w_0, w_1+epsilon) #
	mov.s 		$f12, $f8						#######################
	
	# mov.s		$f6, $f14						#######################
	sub.s		$f7, $f15, $f3					#       CALCULATE     # <-- Stored in $f11
	jal			loss							# L(w_0, w_1-epsilon) #
	mov.s 		$f11, $f8						#######################
	
	sub.s		$f11, $f12, $f11				# $f11 = L(w_0, w_1+epsilon) - L(w_0, w_1-epsilon)
	mul.s		$f11, $f9, $f11					# $f11 = $f11 / 2
	div.s		$f11, $f11, $f3					# $f11 = $f11 / epsilon
	
	jr			$s4


# Function: Calculate loss function L(w) = 1/2 * ((w_0 + w_1*x) - y)^2
# Parameters:
# $f4 = x
# $f5 = y
# $f6, $f7 = w_0, w_1
# $f9 = 1/2
# Return: $f8 = L(w)
# Author: moriaty
loss:
	mul.s	 	$f8, $f7, $f4					# $f8 = w_1*x
	add.s		$f8, $f6, $f8					# $f8 = w_0 + w_1*x
	sub.s		$f8, $f8, $f5					# $f8 = (w_0 + w_1*x) - y
	mul.s		$f8, $f8, $f8					# $f8 = ((w_0 + w_1*x) - y)^2
	mul.s		$f8, $f9, $f8					# $f8 = 1/2 * ((w_0 + w_1*x) - y)^2
	
	jr			$ra
	
# Exit the program
exit:
	li		$v0, 0x0A
	syscall

# Function: print an array of N elements
# Parameters:
# $s0 = N
# $s1 = address of the array
# Used registers: $t0, $s0-1
# Author: moriaty
print_data:
	
	li		$t0, 0
	
	loop_print_data:
		
		li		$v0, 2
		add		$at, $s1, $t0
		lwc1	$f12, ($at)
		syscall		
		li		$v0, 0x0B
		li		$a0, 0x20
		syscall
		
		addi	$t0, $t0, 4
		bne		$t0, $s0, loop_print_data
		
	li		$v0, 0x0B
	li		$a0, 0x0A
	syscall
	
	jr		$ra


# Function: read 1 byte and store it in buffer
# Parameters:
# $s0 = File Descriptor
# Return the read byte inside buffer data.
# Used registers: $s0
# Author: moriaty
read_byte:
	li		$v0, 14			# 14=read from  file
	add		$a0, $s0, $0	# $s0 contains fd
	la		$a1, buffer		# buffer to hold int
	li		$a2, 1			# Read 4 bytes
	syscall
	
	jr		$ra
	
# Function: print byte stored in buffer
	
print_buffer:
	li		$v0, 11			# 1 = print int
	lw		$a0, buffer		# buffer contains the int
	syscall				# print int
	
	li		$v0, 11			# 11 = print char
	li		$a0, 0xA		# 0xA = LF char
	syscall
	
	jr		$ra
	
	
# Function: read a csv line and returns $f0, $f1 as two values from the line.
# If $s3 is 0, this function will not return anything.
# This function uses $t1-9, $s2-3, $f0-3
# Author: moriaty.
read_csv_line:
	
	add		$s2, $ra, $0	# Save return address to $s2
	
	li		$t2, 0x0A		# 0x0A is LF character
	li		$t3, 0x2C		# 0x2E is comma character
	li		$t4, 0x2E		# 0x2C is dot character
	
	mtc1	$0, $f0			# Initialize $f0
	mtc1	$0, $f1			# Initialize $f1
	li		$at, 10
	mtc1 	$at, $f3		# $f3 stores 10.0 (used for finalization divisions)
	cvt.s.w	$f3, $f3
	
	li		$t5, 0			# $t5 == 0 if parsing $f0, otherwise parsing $f1
	li		$t6, 0			# $t6 = number of digits after comma
	li		$t7, 0			# $t7 = parsed number (before division)
	
	loop_read_csv_line:
		jal		read_byte						# Read bytes sequentially
		# jal		print_buffer					# Debugging purpose
		lw		$t1, buffer						# Load buffer char into $t1
		
		li		$at, 0x0D						# 0x0D = CR character
		beq		$t1, $at, loop_condition_read_csv_line
		
		bnez	$s3, parse_read_csv_line		# If $s3 is not zero: parse number
		j 		loop_condition_read_csv_line	# Else, proceed to loop condition checking
		parse_read_csv_line:
			beq		$t1, $t2, parse_comma_read_csv_line		# If encounters EOL
			beq		$t1, $t3, parse_comma_read_csv_line		# If encounters a comma
			beq		$t1, $t4, parse_dot_read_csv_line		# If encounters a dot
			subi	$t9, $t1, 0x30							# Get decimal value of the character you just read
			
			li		$at, 10									################################
			mul		$t7, $t7, $at							# Append $t1 at the end of $t7 #
			add		$t7, $t7, $t9							################################
			
			bgt		$t6, $0, increase_num_digit_read_csv_line
			
			j		loop_condition_read_csv_line			# If current character is a digit then continue to next loop
			
			increase_num_digit_read_csv_line:				# Increases digits count after dot
				addi	$t6, $t6, 1
				j		loop_condition_read_csv_line
			
			parse_comma_read_csv_line:
				li		$t8, 1								# Loop variable for parse_finalize_loop_read_csv_line
				
				
				mtc1 	$t7, $f2							# Load $t7 (parsed number) onto $f2 for finalization step
				cvt.s.w	$f2, $f2
				
				
				j		parse_finalize_loop_read_csv_line	# Proceed to finalize parsed number (dividing by multiple of 10's)
				
			parse_dot_read_csv_line:
				li		$t6, 1								# Start counting digits count after dot
				j		loop_condition_read_csv_line
				
			parse_finalize_loop_read_csv_line:							# Finalize $f2 by repeatedly dividing it by 10 for $t6 times
				
				div.s 	$f2, $f2, $f3									# Divide $f2 by 10 (note that $f3 == 10)
				
				addi	$t8, $t8, 1
				bne		$t8, $t6, parse_finalize_loop_read_csv_line		# Loop condition
				
				li		$t6, 0											# Reset digits count after dot
				bnez	$t5, parse1_finalize_loop_read_csv_line			# If parsing to $f1 then skip to $f1 parsing branch
				
				parse0_finalize_loop_read_csv_line:						# Here we are parsing into $f0
					mov.s	$f0, $f2
					li		$t5, 1										# Switch to parsing $f1
					li		$t6, 0										# $t6 = number of digits after comma
					li		$t7, 0										# $t7 = parsed number (before division)
					j		loop_condition_read_csv_line
					
				parse1_finalize_loop_read_csv_line:						# Here we are parsing into $f1
					mov.s	$f1, $f2
					j		loop_condition_read_csv_line
					
	loop_condition_read_csv_line:
		beq		$t1, $t2, end_read_csv_line		# Compare if EOL ? => return
		j		loop_read_csv_line				# Else, just continue the loop
							
	end_read_csv_line:
		jr $s2
	
	
# Function: read a csv file
# $s4 = n
# $s5 = address of x
# %s6 = address of y
# Used registers: $t0-9, $s0-5, $f0-3
# Author: moriaty
read_csv:

	##############################################
	###########        OPEN FILE       ###########
	##############################################
	li		$v0, 13			# 13=open file
	la		$a0, file		# $a2 = name of file to read
	add		$a1, $0, $0		# $a1=flags=O_RDONLY=0
	add		$a2, $0, $0		# $a2=mode=0
	syscall					# Open File, $v0<-fd
	add		$s0, $v0, $0	# Store fd in $s0
	##############################################
	###########   FINISH OPENING FILE  ########### <----- File Descriptor is stored in $s0
	##############################################
	
	add		$s1, $ra, $0	# Save return address to $s1
	
	li		$s3, 0			##################################
	jal		read_csv_line	# Read first line (columns name) #
							##################################
	
	li		$t0, 0
	
	loop_read_csv:							# Read n lines of csv file
	
		li		$s3, 1						##################################
		jal		read_csv_line				# Read first line (columns name) #
											##################################
		
		add		$at, $s5, $t0				##########################
		swc1	$f0, ($at)					# Store $f0 into array x #
											##########################
											
		add		$at, $s6, $t0				##########################
		swc1	$f1, ($at)					# Store $f1 into array y #
											##########################
		
		addi	$t0, $t0, 4					# Loop increment
		bne		$t0, $s4, loop_read_csv		#################


	########################################
	###########    CLOSE FILE    ###########
	########################################
	li		$v0, 16			# 16=close file
	add		$a0, $s0, $0	# $s0 contains fd
	syscall					# close file

	jr	$s1									# Return to main
