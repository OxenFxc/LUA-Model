local luanet = require("luanet.init")
local Tensor = require("luanet.tensor") -- luanet.init exports nothing yet
local layers = require("luanet.layers.init")
local optim = require("luanet.optim.init")
local loss = require("luanet.loss.init")

-- Since luanet.init is mostly empty comments, I require submodules explicitly for now.

-- XOR data
-- [[0,0] -> 0, [0,1] -> 1, [1,0] -> 1, [1,1] -> 0]
local input_flat = {0,0, 0,1, 1,0, 1,1}
local target_flat = {0, 1, 1, 0}

local input = Tensor.new(input_flat, {4, 2})
local target = Tensor.new(target_flat, {4, 1})

-- Model: 2 -> 4 -> 1
-- Model: 2 -> 8 -> 1
local model = layers.Sequential:new(
    layers.Linear:new(2, 8, true),
    layers.Tanh:new(),
    layers.Linear:new(8, 1, true),
    layers.Sigmoid:new()
)

local criterion = loss.MSELoss:new()

local params, grads = model:parameters()
local optimizer = optim.SGD:new(params, grads, 0.1, 0.9) -- LR 0.1, Momentum 0.9

print("Training XOR model...")
math.randomseed(12345) -- Fixed Seed

for i = 1, 10000 do
    optimizer:zero_grad()

    local output = model:forward(input)
    local l = criterion:forward(output, target)
    local gradOutput = criterion:backward()

    model:backward(gradOutput)
    optimizer:step()

    if i % 500 == 0 then
        print(string.format("Step %d, Loss: %.4f", i, l))
    end
end

print("Testing...")
local output = model:forward(input)
print("Input:\n" .. tostring(input))
print("Target:\n" .. tostring(target))
print("Output:\n" .. tostring(output))

-- Check accuracy
local success = true
for i = 1, 4 do
    local expected = target:get({i, 1})
    local got = output:get({i, 1})
    if math.abs(expected - got) > 0.4 then -- looser tolerance
        success = false
    end
end

if success then
    print("XOR Solved!")
else
    print("XOR Failed to converge perfectly.")
end
