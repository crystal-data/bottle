# Copyright (c) 2020 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# A variable is an abstraction of a Tensor that tracks
# the operations done to the Tensor.  It also keeps
# track of the gradient of the operation if a Variable
# needs to backpropogate.
#
# This is the fundamental object used in automatic
# differentiation, as well as the neural network aspects
# of Num.cr
class Num::Grad::Variable(T)
  private macro operator_op(operator, gate_cls, *args)
    def {{operator.id}}(other : Num::Grad::Variable(T)) : Num::Grad::Variable(T)
      result = @context.variable(@value {{operator.id}} other.value)

      if self.is_grad_needed || other.is_grad_needed
        gate = {{gate_cls}}.new {{*args}}
        gate.cache(result, self, other)
      end

      result
    end
  end

  # Adds a variable to another variable and stores
  # the derivative of the operation in the computational
  # graph.
  #
  # ## Arguments
  #
  # * other : `Num::Grad::Variable` - right hand side of the operation
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([2.0])
  # b = ctx.variable([3.0])
  #
  # f = a + b # => [5.0]
  # f.backprop
  # ```
  operator_op :+, Num::Grad::AddGate(T), self, other

  # Subtracts a variable from another variable and stores
  # the derivative of the operation in the computational
  # graph.
  #
  # ## Arguments
  #
  # * other : `Num::Grad::Variable` - right hand side of the operation
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([2.0])
  # b = ctx.variable([3.0])
  #
  # f = a - b # => [-1.0]
  # f.backprop
  # ```
  operator_op :-, Num::Grad::SubtractGate(T), self, other

  # Multiples a variable to another variable and stores
  # the derivative of the operation in the computational
  # graph.
  #
  # ## Arguments
  #
  # * other : `Num::Grad::Variable` - right hand side of the operation
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([2.0])
  # b = ctx.variable([3.0])
  #
  # f = a * b # => [6.0]
  # f.backprop
  # ```
  operator_op :*, Num::Grad::MultiplyGate(T), self, other

  # Raises a variable to another variable and stores
  # the derivative of the operation in the computational
  # graph.
  #
  # ## Arguments
  #
  # * other : `Num::Grad::Variable` - right hand side of the operation
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([2.0])
  # b = ctx.variable([3.0])
  #
  # f = a ** b # => [8.0]
  # f.backprop
  # ```
  operator_op :**, Num::Grad::PowerGate(T), self, other

  # Divides a variable by another variable and stores
  # the derivative of the operation in the computational
  # graph.
  #
  # ## Arguments
  #
  # * other : `Num::Grad::Variable` - right hand side of the operation
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([2.0])
  # b = ctx.variable([3.0])
  #
  # f = a / b # => [0.66667]
  # f.backprop
  # ```
  operator_op :/, Num::Grad::DivideGate(T), self, other

  # Matrix multiply operator for two variables.  Computes the
  # dot product of two matrices and stores the result in the
  # computational graph
  #
  # ## Arguments
  #
  # * other : `Num::Grad::Variable` - right hand side of the operation
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([[2.0], [2.0]])
  # b = ctx.variable([[3.0, 3.0]])
  #
  # f = a.matmul(b)
  #
  # # [[6, 6],
  # #  [6, 6]]
  #
  # f.backprop
  # ```
  def matmul(b : Num::Grad::Variable(T)) : Num::Grad::Variable(T)
    result = Num::Grad::Variable.new(@context, self.value.matmul(b.value))

    if self.is_grad_needed || b.is_grad_needed
      gate = Num::Grad::MatMulGate(T).new(self, b)
      gate.cache(result, self, b)
    end
    result
  end

  # Slices a variable.  Slices the gradient of the variable
  # using the same arguments
  #
  # ## Arguments
  #
  # * args - Slicing arguments, slicing behavior is the same as
  #   it is for a standard `Tensor`
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([[2.0], [3.0]])
  # b = a[1]
  # b # => [3]
  # ```
  def [](*args)
    output = @value[*args]
    result = @context.variable(output, requires_grad: @requires_grad)
    result.grad = @grad[*args]
    result
  end

  # Reduces a `Tensor` along an axis, summing each view into
  # the variable
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of summation
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new
  # x = ctx.variable([[1.0, 2.0], [3.0, 4.0]])
  # x.sum(0) # => [[4.0, 6.0]]
  # x.sum(1) # => [[3.0], [7.0]]
  # ```
  def sum(axis : Int) : Num::Grad::Variable(T)
    s = Num.sum(@value, axis, dims: true)
    result = @context.variable(s, requires_grad: @requires_grad)
    if self.is_grad_needed
      gate = Num::Grad::SumGate(T).new self
      gate.cache(result, self)
    end
    result
  end

  # Reduces a `Tensor` along an axis, finding the average of each
  # view into the `Tensor`
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of reduction
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new
  # x = ctx.variable([[1.0, 2.0], [3.0, 4.0]])
  # x.mean(0) # => [[2.0, 3.0]]
  # x.mean(1) # => [[1.5], [3.5]]
  # ```
  def mean(axis : Int) : Num::Grad::Variable(T)
    s = sum(axis)
    sz = Num.as_tensor(@value.shape[axis], like: s.value)
    b = @context.variable(sz, requires_grad: @requires_grad)
    s / b
  end

  # Negates the variable
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new
  # x = ctx.variable([1.0, 2.0])
  # -x # => [-1.0, -2.0]
  # ```
  def -
    zero = @context.variable(0, requires_grad: @requires_grad)
    zero - self
  end

  private macro num_op(fn, gate_cls)
    def {{fn.id}} : Num::Grad::Variable(T)
      result = @context.variable(Num.{{ fn.id }}(@value))
      if self.is_grad_needed
        gate = {{gate_cls}}.new self
        gate.cache(result, self)
      end
      result
    end
  end

  # Computes the sine of a variable
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new
  # x = ctx.variable([1.0])
  # x.sin # => [0.841471]
  # ```
  num_op sin, Num::Grad::SinGate(T)

  # Computes the cosine of a variable
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new
  # x = ctx.variable([1.0])
  # x.cos # => [0.540302]
  # ```
  num_op cos, Num::Grad::CosGate(T)

  # Computes the tangent of a variable
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new
  # x = ctx.variable([1.0])
  # x.tan # => [1.55741]
  # ```
  num_op tan, Num::Grad::TanGate(T)

  # Computes the tanh of a variable
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new
  # x = ctx.variable([1.0])
  # x.tanh # => [0.761594156]
  # ```
  num_op tanh, Num::Grad::TanhGate(T)

  # Computes the arcsine of a variable
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new
  # x = ctx.variable([1.0])
  # x.asin # => [1.5708]
  # ```
  num_op asin, Num::Grad::ASinGate(T)

  # Computes the arccosine of a variable
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new
  # x = ctx.variable([1.0])
  # x.acos # => [0]
  # ```
  num_op acos, Num::Grad::ACosGate(T)

  # Computes the arctangent of a variable
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new
  # x = ctx.variable([1.0])
  # x.atan # => [0.785398]
  # ```
  num_op atan, Num::Grad::ATanGate(T)

  # Computes the exp of a variable
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new
  # x = ctx.variable([1.0])
  # x.exp # => [2.71828]
  # ```
  num_op exp, Num::Grad::ExpGate(T)

  # Computes the log of a variable
  #
  # ## Examples
  #
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new
  # x = ctx.variable([2.7182818285])
  # x.log # => [1.0]
  # ```
  num_op log, Num::Grad::LogGate(T)
end
