#!/usr/bin/env ruby
#
# Author: Riccardo Orizio
# Date: Wed 03 May 2017
# Description: Neural Network try
#

# Class defining a connection, a single wire connecting two elements
class Connection

	# Keeping track of all the connection created
	@@array = Array.new

	attr_accessor :forward, :backward, :name

	def self.all_instances
		@@array
	end

	def initialize( f, b = 1, name = "" )
		@forward = f
		@backward = b
		@name = name
		puts "Created Connection #{to_string}"

		@@array << self
	end

	def update_after_backpropagation( step_size )
		@forward += step_size * @backward
	end

	def to_string
		sprintf( "%s:\tF = %+.5f\tB = %+.5f\n", @name, @forward, @backward )
	end
end

# Starting with some circuits examples
class Gate

	# Attributes of the class
	#  - type: type of gate
	#  - inputs: number and corresponding values of inputs
	#  - outputs: output value
	attr_accessor :inputs, :output, :type

	# Initializing the gate
	def initialize( inputs, type )
		@type = type
		@inputs = inputs
		@output = Connection.new( 0, 1, @inputs.map( &:name ).join( @type ) )
		forward()
		puts "Created Gate #{@type} with #{@inputs.length} inputs"
	end

	# Computing the output, forward evolution
	def forward()
		case @type
		when "+"
			@output.forward = @inputs.inject( 0, :+ )
		when "*"
			@output.forward = @inputs.inject( 1, :* )
		else
			@output.forward = nil
		end
	end

	def derivative( index )
		nil
	end
	
	# Gradient computation or backward evolution
	def backward()
		@inputs.each_index{ |i|
			@inputs[ i ].backward = ( derivative( i ) * @output.backward )
		}
	end

end

class SumGate < Gate

	def initialize( inputs )
		super( inputs, "+" )
	end

	def forward()
		@output.forward = @inputs.map( &:forward ).inject( 0, :+ )
	end

	def derivative( index )
		1
	end

end

class MulGate < Gate
	
	def initialize( inputs )
		super( inputs, "*" )
	end

	def forward()
		@output.forward = @inputs.map( &:forward ).inject( 1, :* )
	end

	def derivative( index )
		# Calculating the partial derivative excluding the variable of which the
		# derivative is been computed
		@inputs.map( &:forward ).map.with_index{ |v,i| i == index ? 1 : v }.inject( 1, :* )
	end

end

# Connection class to handle the connection between the Gates
class Sigmoid 
	
	attr_accessor :input, :output, :backpropagation

	def initialize( input )
		@input = input
		@output = Connection.new( 0, 1, "s(".concat( @input.name ).concat( ")" ) )
		forward()
		puts "Created Sigmoid neuron"
	end

	# Output computation / forward evolution:
	#                  1
	# sigma(x) = ---------------
	#              1 + exp(-x)
	def forward()
		@output.forward = 1 / ( 1 + Math.exp( -@input.forward ) )
	end

	# Derivation / backward evolution:
	# dx = sigma( x )( 1 - sigma( x ) )
	def backward()
		@input.backward = ( @output.forward * ( 1 - @output.forward ) ) * @output.backward
	end

end

# Testing example
#	# Function: f(.) = (x+y)z
#	x = Connection.new( -2, 0, "x" )
#	y = Connection.new( 5, 0, "y" )
#	z = Connection.new( -4, 0, "z" )
#	
#	sum = SumGate.new( [ x, y ] )
#	mul = MulGate.new( [ sum.output, z ] )
#	
#	puts "Forward: #{mul.output.to_string}"
#	
#	# Backpropagating
#	mul.backward()
#	sum.backward()
#	
#	puts "\n"
#	Connection.all_instances.each{ |i| puts "\t#{i.to_string}" }
#	
#	step_size = 0.01
#	x.update_after_backpropagation( step_size )
#	y.update_after_backpropagation( step_size )
#	z.update_after_backpropagation( step_size )
#	
#	puts "Backpropagated"
#	Connection.all_instances.each{ |i| puts "\t#{i.to_string}" }
#	
#	sum.forward()
#	mul.forward()
#	
#	puts "Forward: #{mul.output.to_string}"

# Simulating f() = sigma( ax + by + c )

a = Connection.new( 1, 0, "a" )
b = Connection.new( 2, 0, "b" )
c = Connection.new( -3, 0, "c" )
x = Connection.new( -1, 0, "x" )
y = Connection.new( 3, 0, "y" )

# Elements of the network
m_ax = MulGate.new( [ a, x ] )
m_by = MulGate.new( [ b, y ] )
sum = SumGate.new( [ m_ax.output, m_by.output, c ] )
res = Sigmoid.new( sum.output )

puts "\nForwarded: #{res.output.to_string}"

# Backwarding
# Starting from the output I'll go backwards running the backward function of
# each element on the way
res.backward()
sum.backward()
m_by.backward()
m_ax.backward()

# Updating all the connection values
step_size = 0.01
a.update_after_backpropagation( step_size )
b.update_after_backpropagation( step_size )
c.update_after_backpropagation( step_size )
x.update_after_backpropagation( step_size )
y.update_after_backpropagation( step_size )

puts "Backpropagated"
Connection.all_instances.each{ |i| puts "\t#{i.to_string}" }

# Rerunning the network forward after the updates
m_ax.forward()
m_by.forward()
sum.forward()
res.forward()

puts "Forwarded: #{res.output.to_string}"

