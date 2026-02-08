local Tensor = require("luanet.tensor")

local function assert_eq(a, b, msg)
    if math.abs(a - b) > 1e-6 then
        error("Assertion failed: " .. a .. " ~= " .. b .. " " .. (msg or ""))
    end
end

print("Testing Tensor...")

-- Test creation
local t1 = Tensor.new({1, 2, 3, 4}, {2, 2})
print("t1:", t1)

local t2 = Tensor.zeros({2, 2})
print("t2:", t2)

-- Test element-wise add
local t3 = t1 + t2
assert_eq(t3:get({1,1}), 1)
assert_eq(t3:get({2,2}), 4)
print("Add passed")

-- Test matmul
-- [[1, 2], [3, 4]] * [[1, 0], [0, 1]] = [[1, 2], [3, 4]]
local eye = Tensor.new({1, 0, 0, 1}, {2, 2})
local t4 = t1:matmul(eye)
assert_eq(t4:get({1,1}), 1)
assert_eq(t4:get({1,2}), 2)
assert_eq(t4:get({2,1}), 3)
assert_eq(t4:get({2,2}), 4)
print("Matmul Identity passed")

-- [[1, 2], [3, 4]] * [[0, 1], [1, 0]] = [[2, 1], [4, 3]]
local swap = Tensor.new({0, 1, 1, 0}, {2, 2})
local t5 = t1:matmul(swap)
assert_eq(t5:get({1,1}), 2)
assert_eq(t5:get({1,2}), 1)
assert_eq(t5:get({2,1}), 4)
assert_eq(t5:get({2,2}), 3)
print("Matmul Swap passed")

-- Test transpose
local t6 = t1:transpose()
-- t1: [[1, 2], [3, 4]] -> t6: [[1, 3], [2, 4]]
assert_eq(t6:get({1,1}), 1)
assert_eq(t6:get({1,2}), 3)
assert_eq(t6:get({2,1}), 2)
assert_eq(t6:get({2,2}), 4)
print("Transpose passed")

-- Test view
local t7 = t1:view({4})
assert_eq(t7.shape[1], 4)
assert_eq(t7:get({1}), 1)
assert_eq(t7:get({4}), 4)
print("View passed")

print("All Tensor tests passed!")
