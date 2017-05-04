#!/usr/bin/env ruby
#
# Author: Riccardo Orizio
# Date: Wed 03 May 2017
# Description: Neural Network try
#

# Starting with some circuits examples
class Gate

	# Attributes of the class
	#  - type: type of gate
	#  - inputs: number and corresponding values of inputs
	#  - outputs: output value
	attr_accessor :inputs, :output, :type, :backpropagation

	# Initializing the gate
	def initialize( inputs, type )
		@type = type
		@inputs = inputs
		@backpropagation = 1
		puts "Created a new #{@type} Gate with #{@inputs.length} inputs"
	end

	# Computing the output
	def compute()
		case @type
		when "+"
			@output = @inputs.inject( 0, :+ )
		when "*"
			@output = @inputs.inject( 1, :* )
		else
			@output = nil
		end
	end

	def derivative( index )
		nil
	end
	
	def gradient()
		res = Array.new( @inputs.length )
		res.each_index{ |i| res[ i ] = derivative( i ) * @backpropagation }
	end

end

class SumGate < Gate

	def initialize( inputs )
		super( inputs, "+" )
	end

	def compute()
		@output = @inputs.inject( 0, :+ )
	end

	def derivative( index )
		1
	end

end

class MulGate < Gate
	
	def initialize( inputs )
		super( inputs, "*" )
	end

	def compute()
		@output = @inputs.inject( 1, :* )
	end

	def derivative( index )
		#	res = 1
		#	@inputs.each.with_index{ |i, j|
		#		unless( j == index )
		#			res *= i
		#		end
		#	}
		#	return res
		@inputs.map.with_index{ |v,i| i == index ? 1 : v }.inject( 1, :* )
	end

end

# Sigmoid neuron
class Sigmoid 
	
	attr_accessor :input, :output, :backpropagation

	def initialize( input )
		@input = input
		@backpropagation = 1
		@output = compute()
		puts "Sigmoid neuron created"
	end

	# Output computation:
	#                  1
	# sigma(x) = ---------------
	#              1 + exp(-x)
	def compute()
		@output = 1 / ( 1 + Math.exp( -@input ) )
	end

	# Derivation
	# dx = sigma( x )( 1 - sigma( x ) )
	def derivative()
		( @output * ( 1 - @output ) ) * @backpropagation
	end

end

# Testing part
x = -2
y = 3
a_gate = Gate.new( [ x, y ], "*" )

puts "Gate result: #{a_gate.compute}"

# Improvement 1: Derivative and Gradient
step = 1e-4
dx = ( Gate.new( [ x+step, y ], "*" ).compute - a_gate.compute ) / step
dy = ( Gate.new( [ x, y+step ], "*" ).compute - a_gate.compute ) / step

puts "dx: #{dx}"
puts "dy: #{dy}"

# Testing the derivative results
step_size = 0.01
i1_gate = Gate.new( [ x+dx*step_size, y+dy*step_size ], "*" )
puts "Gate i2: #{i1_gate.compute}"

# Improvement 2: Analytic Gradient
# Partial derivative of our function xy:
# dx(f) = dx( xy ) = y
# dy(f) = dy( xy ) = x
pdx = y
pdy = x
i2_gate = Gate.new( [ x+pdx*step_size, y+pdy*step_size ], "*" )
puts "Gate i2: #{i2_gate.compute}"

puts "**********************************"

# Multi gates: f(x,y,z) = ( x + y ) * z
x = -2
y = 5
z = -4
#	sum_gate = Gate.new( [ x, y ], "+" )
#	mul_gate = Gate.new( [ sum_gate.compute, z ], "*" )
sum_gate = SumGate.new( [ x, y ] )
mul_gate = MulGate.new( [ sum_gate.compute, z ] )

puts "Multi gate result: #{mul_gate.compute}"

# Backpropagation: starting from the last gate I compute the gradients and
# propagate it backwards
puts "MulGate gradient: #{mul_gate.gradient()}"
# Setting the backpropagation factor to the SumGate
# I use the first value of the gradient because that is the one connected to the
# SumGate
sum_gate.backpropagation = mul_gate.gradient()[ 0 ]
puts "SumGate gradient: #{sum_gate.gradient()}"

# Composing the final gradient in respect to the inputs
#	circuit_gradient = [ sum_gate.gradient().map{ |v| v * mul_gate.gradient()[0] }, mul_gate.gradient()[1] ].flatten
circuit_gradient = sum_gate.gradient() << mul_gate.gradient()[ 1 ]
puts "Circuit gradient: #{circuit_gradient}"

# Improving the result using the gradient
sum_gate.inputs = [ x+circuit_gradient[0]*step_size, y+circuit_gradient[1]*step_size ]
mul_gate.inputs = [ sum_gate.compute(), z+circuit_gradient[2]*step_size ]
puts "Improved: #{mul_gate.compute()}"

puts "**********************************"

sig = Sigmoid.new( 3 )
puts "Output: #{sig.output}"
puts "Dsx: #{sig.derivative}"
