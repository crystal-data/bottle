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

# :nodoc:
abstract class Num::Grad::TwoOpGate(T) < Num::Grad::Gate(T)
  getter a : Num::Grad::Variable(T)
  getter b : Num::Grad::Variable(T)
  @@name = "TwoOp"

  # :nodoc:
  def initialize(@a, @b)
  end

  abstract def backward(payload : Num::Grad::Payload(T)) : Array(T)

  # :nodoc:
  def cache(result : Num::Grad::Variable(T), *args)
    a, b = args
    result.grad = T.zeros_like(result.value)
    result.requires_grad = true

    Num::Grad.register(@@name, self, result, a, b)
  end
end

# :nodoc:
class Num::Grad::AddGate(T) < Num::Grad::TwoOpGate(T)
  @@name = "Add"

  # :nodoc:
  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    Num::Grad.add_backward(payload.variable.grad, a, b)
  end
end

# :nodoc:
class Num::Grad::SubtractGate(T) < Num::Grad::TwoOpGate(T)
  @@name = "Sub"

  # :nodoc:
  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    Num::Grad.subtract_backward(payload.variable.grad, a, b)
  end
end

# :nodoc:
class Num::Grad::MultiplyGate(T) < Num::Grad::TwoOpGate(T)
  @@name = "Multiply"

  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    Num::Grad.multiply_backward(payload.variable.grad, a, b)
  end
end

# :nodoc:
class Num::Grad::DivideGate(T) < Num::Grad::TwoOpGate(T)
  @@name = "Divide"

  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    Num::Grad.divide_backward(payload.variable.grad, a, b)
  end
end

# :nodoc:
class Num::Grad::PowerGate(T) < Num::Grad::TwoOpGate(T)
  @@name = "Power"

  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    Num::Grad.power_backward(payload.variable.grad, a, b)
  end
end
