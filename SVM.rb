#!/usr/bin/env ruby
#
# Author: Riccardo Orizio
# Date: Thu 04 May 2017
# Description: Support Vector Machine classifier
#

require( "./NeuralNetworkComponents.rb" )

class ClassifierFunction

	# TODO: It sucks.
	attr_accessor :a, :b

	#	def function( n_par, n_input )
	def initialize 
		# Function used for the classifier
		# f(.) = a*x + b*y + c
		# We will read x and y values from our dataset, so the backpropagation will not
		# affect them, we will only focus on the parameters a, b and c.
		@a = Connection.new( 0, "a" )
		@b = Connection.new( 0, "b" )
		@c = Connection.new( 0, "c" )
		@x = Connection.new( 0, "x" )
		@y = Connection.new( 0, "y" )
		
		@m_ax = MulGate.new( [ @a, @x ] )
		@m_by = MulGate.new( [ @b, @y ] )
		@res = SumGate.new( [ @m_ax.output, @m_by.output, @c ] )

		#	TODO
		#	@par = Array.new( n_par, Connection.new( 0 ) )
		#	# Naming the parameters, probably wrong
		#	@par.each_with_index{ |x| x.name = i.to_s }

		#	@m_x = MulGate.new( [ @par[ 0 ], @x ] )
		#	@m_y = MulGate.new( [ @par[ 1 ], @y ] )
		#	@res = SumGate.new( [ @m_ax.output, @m_by.output, @par[ 2 ] ] )
		
	end

	def set_parameters( a, b, c )
		@a.forward = a
		@b.forward = b
		@c.forward = c
	end

	def set_inputs( x, y )
		@x.forward = x
		@y.forward = y
	end

	def forward( x, y )
		set_inputs( x, y )

		@m_ax.forward()
		@m_by.forward()
		@res.forward()

		return @res.output
	end

	def backward( direction )
		# Setting the direction in which I want to move
		@res.output.backward = direction

		@res.backward()
		@m_by.backward()
		@m_ax.backward()
	end

	def update_parameters( step_size )
		@a.update_after_backpropagation( step_size )
		@b.update_after_backpropagation( step_size )
		@c.update_after_backpropagation( step_size )
	end

	def output
		@res.output
	end
end

class SVM

	def initialize
		@function = ClassifierFunction.new
	end

	def forward( x, y )
		@function.forward( x, y )
	end

	def backward( label )
		# Depending on the label received I set the direction for the Classifier
		dir = 0
		if( label == 1 and @function.output < 1 )
			dir = -1
		end
		if( label == -1 and @function.output > -1 )
			dir = 1
		end

		@function.backward( dir )

		# TODO: Find another way to do this
		# Smoothing a and b parameters toward zero adjusting their backward
		# value
		@function.a.backward += -@function.a.forward
		@function.b.backward += -@function.b.forward

	end

	def update_parameters
		@function.update_parameters( @step_size )
	end

	def learn( x, y, label )
		forward( x, y )
		backward( label )
		parameter_update()
	end
end
