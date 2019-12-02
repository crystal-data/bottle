require "./macros"
require "./exceptions"
require "./common"
require "../base/base"

module Bottle::Assemble
  include Internal
  # Concatenates an array of `Tensor's` along a provided axis.
  #
  # Parameters
  # ----------
  # alist : Array(Tensor)
  #   - Array containing Tensors to be concatenated
  # axis : Int32
  #   - Axis for concatentation, must be an existing
  #     axis present in all Tensors, and all Tensors
  #     must have the same shape off-axis
  #
  # Returns
  # -------
  # ret : Tensor
  #   - Result of the concatenation
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([2, 2, 3]) { |i| i }
  #
  # concatenate([t, t, t], axis=-1)
  #
  # Tensor([[[ 0,  1,  2,  0,  1,  2,  0,  1,  2],
  #          [ 3,  4,  5,  3,  4,  5,  3,  4,  5]],
  #
  #         [[ 6,  7,  8,  6,  7,  8,  6,  7,  8],
  #          [ 9, 10, 11,  9, 10, 11,  9, 10, 11]]])
  # ```
  def concatenate(alist : Array(BaseArray(U)), axis : Int32) forall U
    newshape = alist[0].shape.dup
    clipaxis axis, newshape.size
    newshape[axis] = 0
    shape = assert_shape_off_axis(alist, axis, newshape)
    ret = alist[0].class.new(newshape)
    lo = [0] * newshape.size
    hi = shape.dup
    hi[axis] = 0
    alist.each do |a|
      if a.shape[axis] != 0
        hi[axis] += a.shape[axis]
        ranges = lo.zip(hi).map { |i, j| i...j }
        ret[ranges] = a
        lo[axis] = hi[axis]
      end
    end
    ret
  end

  # Concatenates a list of `Tensor`s along axis 0
  #
  # Parameters
  # ----------
  # alist : Array(Tensor)
  #   - Array containing Tensors to be stacked
  #
  # Returns
  # -------
  # ret : Tensor
  #   - Result of the concatenation
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([2, 2, 3])
  # vstack([t, t, t])
  #
  # Tensor([[[ 0,  1,  2],
  #          [ 3,  4,  5]],
  #
  #         [[ 6,  7,  8],
  #          [ 9, 10, 11]],
  #
  #         [[ 0,  1,  2],
  #          [ 3,  4,  5]],
  #
  #         [[ 6,  7,  8],
  #          [ 9, 10, 11]],
  #
  #         [[ 0,  1,  2],
  #          [ 3,  4,  5]],
  #
  #         [[ 6,  7,  8],
  #          [ 9, 10, 11]]])
  # ```
  def vstack(alist : Array(BaseArray(U))) forall U
    concatenate(alist, 0)
  end

  # Concatenates a list of `Tensor`s along axis 1
  #
  # Parameters
  # ----------
  # alist : Array(Tensor)
  #   - Array containing Tensors to be stacked
  #
  # Returns
  # -------
  # ret : Tensor
  #   - Result of the concatenation
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([2, 2, 3])
  # hstack([t, t, t])
  #
  # Tensor([[[ 0,  1,  2],
  #          [ 3,  4,  5],
  #          [ 0,  1,  2],
  #          [ 3,  4,  5],
  #          [ 0,  1,  2],
  #          [ 3,  4,  5]],
  #
  #         [[ 6,  7,  8],
  #          [ 9, 10, 11],
  #          [ 6,  7,  8],
  #          [ 9, 10, 11],
  #          [ 6,  7,  8],
  #          [ 9, 10, 11]]])
  # ```
  def hstack(alist : Array(BaseArray(U))) forall U
    concatenate(alist, 1)
  end

  def dstack(alist : Array(BaseArray(U))) forall U
    first = alist[0]
    shape = first.shape
    assert_shape(shape, alist)

    case first.ndims
    when 1
      alist = alist.map do |a|
        a.reshape([1, a.size, 1])
      end
      concatenate(alist, 2)
    when 2
      alist = alist.map do |a|
        a.reshape(a.shape + [1])
      end
      concatenate(alist, 2)
    else
      raise ShapeError.new("dstack was given arrays with more than two dimensions")
    end
  end

  def column_stack(alist : Array(BaseArray(U))) forall U
    first = alist[0]
    shape = first.shape
    assert_shape(first, alist)

    case first.ndims
    when 1
      alist = alist.map do |a|
        a.reshape([a.size, 1])
      end
      concatenate(alist, 1)
    when 2
      concatenate(alist, 1)
    else
      raise ShapeError.new("dstack was given arrays with more than two dimensions")
    end
  end
end
