require "./api"

module Bottle
  extend self
  VERSION = "0.2.2"
end

include Bottle

c = CharArray.new([3, 2, 2]) { |i| 'a' + 3 }
