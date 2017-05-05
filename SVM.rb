#!/usr/bin/env ruby
#
# Author: Riccardo Orizio
# Date: Thu 04 May 2017
# Description: Support Vector Machine classifier
#

require( "./NeuralNetworkComponents.rb" )

class ClassifierFunction

	# TODO: It sucks.
	attr_accessor :a, :b, :c

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

		return output
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
		@res.output.forward
	end

	def par_to_string
		sprintf( "%+.5f(%+.5f) %+.5f(%+.5f) %+.5f(%+.5f)",
						@a.forward, @a.backward,
						@b.forward, @b.backward,
						@c.forward, @c.forward )
	end
end

class SVM

	def initialize( function )
		@function = function
		@step_size = 0.01
	end

	def forward( x, y )
		@function.forward( x, y )
	end

	def backward( label )

		# Resetting backwards values of our parameters
		# Probably not really needed
		#	@function.a.backward = 0
		#	@function.b.backward = 0
		#	@function.c.backward = 0

		# Depending on the label received I set the direction for the Classifier
		dir = 0
		if( label == 1 and @function.output < 1 )
			dir = 1
		end
		if( label == -1 and @function.output > -1 )
			dir = -1
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
		update_parameters()
	end

	def par_to_string
		@function.par_to_string
	end
end

def training_accuracy( data, classifier )
	correct = 0
	data.each{ |x|
		# Calculataing the prediction of the classifier
		predicted = classifier.forward( x.value[ 0 ], x.value[ 1 ] )
		print "\tPredicted: #{predicted}"
		# Evaluating its label
		predicted = ( predicted > 0 ? 1 : -1 )
		puts " => #{predicted}/#{x.label}"
		# Correct prediction if the labels match
		if( predicted == x.label )
			correct += 1
		end
	}

	puts "\t#{correct} #{data.length} #{100.0*correct/data.length}"

	# Returning the rate of correctness of the classifier
	return( 100.0 * correct / data.length )
end

def evaluate_score( x, y, a, b, c )
	a*x + b*y + c
end

def temp_accuracy( data, a, b, c )
	correct = 0
	data.each{ |x|
		# Calculataing the prediction of the classifier
		predicted = evaluate_score( x.value[ 0 ], x.value[ 1 ], a, b, c )
		print "\tPredicted: #{predicted}"
		# Evaluating its label
		predicted = ( predicted > 0 ? 1 : -1 )
		puts " => #{predicted}/#{x.label}"
		# Correct prediction if the labels match
		if( predicted == x.label )
			correct += 1
		end
	}

	puts "\t#{correct} #{data.length} #{100.0*correct/data.length}"

	# Returning the rate of correctness of the classifier
	return( 100.0 * correct / data.length )
end

# **************************************************************************** #

# Data to train the Neural Network
Struct.new( "Data", :value, :label )
dataset = Array.new

dataset.push( Struct::Data.new( [  1.2,  0.7 ],  1 ) )
dataset.push( Struct::Data.new( [ -0.3, -0.5 ], -1 ) )
dataset.push( Struct::Data.new( [  3.0,  0.1 ],  1 ) )
dataset.push( Struct::Data.new( [ -0.1, -1.0 ], -1 ) )
dataset.push( Struct::Data.new( [ -1.0,  1.1 ], -1 ) )
dataset.push( Struct::Data.new( [  2.1, -3.0 ],  1 ) )

function = ClassifierFunction.new
function.set_parameters( 1, -2, -1 )
classifier = SVM.new( function )

# Learning through a finite loop
max_steps = 400
(1..max_steps).each{ |i|
	# Random data from the dataset used for this loop
	data = dataset.sample
	classifier.learn( data.value[ 0 ], data.value[ 1 ], data.label )

	# Updating the user over the training session
	if( i % 25 == 0 )
		printf( "%d) %.2f%% [%s]\n",
					i,
					training_accuracy( dataset, classifier ),
					classifier.par_to_string )
	end
}


# EXPERIMENT
puts "***********************************************"
a = 1
b = -2
c = -1
(1..max_steps).each{ |i|
	data = dataset.sample
	score = evaluate_score( data.value[ 0 ],
							data.value[ 1 ],
							a, b, c )
	pull = 0
	if( data.label == 1 and score < 1 )
		pull = 1
	end
	if( data.label == -1 and score > -1 )
		pull = -1
	end

	step_size = 0.01
	a += step_size * ( data.value[ 0 ] * pull - a )
	b += step_size * ( data.value[ 1 ] * pull - b )
	c += step_size * ( 1 * pull )
	
	# Updating the user over the training session
	if( i % 25 == 0 )
		printf( "%d) %.2f [%+.5f/%+.5f/%+.5f]\n",
					i,
					temp_accuracy( dataset, a, b, c ),
					a, b, c )
	end
}

printf( "Comparison:\n\ta: %+.5f/%+.5f\n\tb: %+.5f/%+.5f\n\tc: %+.5f/%+.5f\n",
			function.a.forward, a,
			function.b.forward, b,
			function.c.forward, c )
printf( "\t%%: %.2f%%/%.2f%%\n",
			training_accuracy( dataset, classifier ),
			temp_accuracy( dataset, a, b, c ) )
