#!/usr/bin/env ruby
#
# Author: Riccardo Orizio
# Date: Wed 03 May 2017
# Description: Neural Network Components
#

# Class defining a connection, a single wire connecting two elements
class Connection

	# Keeping track of all the connection created
	@@array = Array.new

	attr_accessor :forward, :backward, :name

	def self.all_instances
		@@array
	end

	def initialize( f, name = "", b = 1 )
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
		@output = Connection.new( 0, @inputs.map( &:name ).join( @type ), 1 )
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
		@output = Connection.new( 0, "s(".concat( @input.name ).concat( ")" ), 1 )
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

