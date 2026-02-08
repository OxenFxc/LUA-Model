local Module = require("luanet.module")
local Tensor = require("luanet.tensor")

local Linear = setmetatable({}, {__index = Module})
Linear.__index = Linear

function Linear:new(in_features, out_features, bias)
    local self = setmetatable({}, Linear)
    self:init()

    self.in_features = in_features
    self.out_features = out_features

    -- Initialize weights: (in_features, out_features)
    -- Simple random initialization
    self.weight = Tensor.randn({in_features, out_features})

    -- Scale weights (Xavier-like)
    local stdv = 1.0 / math.sqrt(in_features)

    -- Apply scaling
    local function scale(x) return x * stdv end
    self.weight = self.weight:apply(scale)

    self.gradWeight = Tensor.zeros({in_features, out_features})

    self:register_parameter("weight", self.weight, self.gradWeight)

    if bias ~= false then
        self.bias = Tensor.zeros({1, out_features})
        self.gradBias = Tensor.zeros({1, out_features})
        self:register_parameter("bias", self.bias, self.gradBias)
    else
        self.bias = nil
        self.gradBias = nil
    end

    return self
end

function Linear:forward(input)
    self.input = input
    local output = input:matmul(self.weight)
    if self.bias then
        output = output + self.bias
    end
    return output
end

function Linear:backward(gradOutput)
    -- gradOutput: (batch, out)
    -- input: (batch, in)
    -- weight: (in, out)

    -- gradInput = gradOutput * weight.T
    local gradInput = gradOutput:matmul(self.weight:transpose())

    -- gradWeight = input.T * gradOutput
    local gw = self.input:transpose():matmul(gradOutput)

    -- Accumulate gradients in-place
    if #self.gradWeight.data ~= #gw.data then error("Gradient shape mismatch") end
    for i=1, #self.gradWeight.data do
        self.gradWeight.data[i] = self.gradWeight.data[i] + gw.data[i]
    end

    if self.bias then
        local gb = gradOutput:sum(1) -- Sum over batch (dim 1) -> (1, out)
        if #self.gradBias.data ~= #gb.data then error("Bias Gradient shape mismatch") end
        for i=1, #self.gradBias.data do
            self.gradBias.data[i] = self.gradBias.data[i] + gb.data[i]
        end
    end

    return gradInput
end

return Linear
