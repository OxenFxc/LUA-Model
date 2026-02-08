local Tensor = require("luanet.tensor")

local function assert_eq(a, b, msg)
    if math.abs(a - b) > 1e-6 then
        error("Assertion failed: " .. a .. " ~= " .. b .. " " .. (msg or ""))
    end
end

print("Testing Tensor Enhancements...")

-- Test exp
local t1 = Tensor.new({0, 1}, {2})
local t2 = t1:exp()
assert_eq(t2:get({1}), 1)
assert_eq(t2:get({2}), math.exp(1))
print("Exp passed")

-- Test log
local t3 = t2:log()
assert_eq(t3:get({1}), 0)
assert_eq(t3:get({2}), 1)
print("Log passed")

-- Test sqrt
local t4 = Tensor.new({4, 9}, {2})
local t5 = t4:sqrt()
assert_eq(t5:get({1}), 2)
assert_eq(t5:get({2}), 3)
print("Sqrt passed")

-- Test cat 1D
local c1 = Tensor.new({1, 2}, {2})
local c2 = Tensor.new({3, 4}, {2})
local c3 = Tensor.cat({c1, c2}, 1) -- {1, 2, 3, 4}
assert_eq(c3.shape[1], 4)
assert_eq(c3:get({1}), 1)
assert_eq(c3:get({3}), 3)
assert_eq(c3:get({4}), 4)
print("Cat 1D passed")

-- Test cat 2D dim 1 (rows)
local m1 = Tensor.new({1, 2, 3, 4}, {2, 2})
local m2 = Tensor.new({5, 6, 7, 8}, {2, 2})
local m3 = Tensor.cat({m1, m2}, 1) -- 4x2
assert_eq(m3.shape[1], 4)
assert_eq(m3.shape[2], 2)
assert_eq(m3:get({1, 1}), 1)
assert_eq(m3:get({3, 1}), 5)
print("Cat 2D dim 1 passed")

-- Test cat 2D dim 2 (cols)
local m4 = Tensor.cat({m1, m2}, 2) -- 2x4
assert_eq(m4.shape[1], 2)
assert_eq(m4.shape[2], 4)
assert_eq(m4:get({1, 1}), 1)
assert_eq(m4:get({1, 2}), 2)
assert_eq(m4:get({1, 3}), 5)
assert_eq(m4:get({1, 4}), 6)
assert_eq(m4:get({2, 1}), 3)
assert_eq(m4:get({2, 3}), 7)
print("Cat 2D dim 2 passed")

print("All Tensor enhancement tests passed!")
