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
  def divide_backward(
    gradient : Tensor(U, CPU(U)),
    a : Variable(Tensor(U, CPU(U))),
    b : Variable(Tensor(U, CPU(U)))
  ) : Array(Tensor(U, CPU(U))) forall U
    r0 = gradient.map(b.value) { |i, j| i / j }
    r1 = gradient.map(a.value, b.value) { |i, j, k| -i * j / (k ** 2) }
    [r0, r1]
  end

  def power_backward(
    gradient : Tensor(U, CPU(U)),
    a : Variable(Tensor(U, CPU(U))),
    b : Variable(Tensor(U, CPU(U)))
  ) : Array(Tensor(U, CPU(U))) forall U
    r0 = gradient.map(a.value, b.value) do |grad, x, y|
      grad * y * (x ** (y == 0 ? 1 : y - 1))
    end
    r1 = gradient.map(a.value, b.value) do |grad, x, y|
      grad * (x ** y) * Math.log(x == 0 ? 1 : x)
    end
    [r0, r1]
  end
end
