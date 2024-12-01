# Copyright (c) 2021 Crystal Data Contributors
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

module Num::Grad
  extend self

  #
  # This returns the appropriate backward gradient processing for
  # addition and subtraction based on the
  # size and rank of the two variables
  #
  private def sum_grad_backward(gradient : U, a : U, b : U) : Array(U) forall U
    if a.rank != b.rank
      # broadcast along an axis, so sum dwn by axis
      swap = a.rank > b.rank
      gless = gradient
      (b.rank - a.rank).abs.times do
        gless = gless.sum(0)
      end
      if a.size == 1 || b.size == 1
        gless = gless.sum(0)
      end
      swap ? [gradient, gless] : [gless, gradient]
    else
      [gradient, gradient]
    end
  end

  # :nodoc:
  def add_backward(gradient : U, a : Variable(U), b : Variable(U)) : Array(U) forall U
    sum_grad_backward(gradient, a.value, b.value)
  end

  # :nodoc:
  def subtract_backward(gradient : U, a : Variable(U), b : Variable(U)) : Array(U) forall U
    r = sum_grad_backward(gradient, a.value, b.value)
    r[1] = -r[1]
    r
  end

  # :nodoc:
  def multiply_backward(
    gradient : U,
    a : Variable(U),
    b : Variable(U)
  ) : Array(U) forall U
    [gradient * b.value, a.value * gradient]
  end

  # :nodoc:
  def matmul_backward(
    gradient : U,
    a : Variable(U),
    b : Variable(U)
  ) : Array(U) forall U
    [gradient.matmul(b.value.transpose), a.value.transpose.matmul(gradient)]
  end
end
