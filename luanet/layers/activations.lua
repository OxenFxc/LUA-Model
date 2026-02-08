local Module = require("luanet.module")
local Tensor = require("luanet.tensor")

local function tanh_func(x)
    -- Safe tanh
    if x > 20 then return 1 end
    if x < -20 then return -1 end
    local e2x = math.exp(2 * x)
    return (e2x - 1) / (e2x + 1)
end

local Activations = {}

-- ReLU --
local ReLU = setmetatable({}, {__index = Module})
ReLU.__index = ReLU

function ReLU:new()
    local self = setmetatable({}, ReLU)
    self:init()
    return self
end

function ReLU:forward(input)
    self.input = input
    return input:apply(function(x) return x > 0 and x or 0 end)
end

function ReLU:backward(gradOutput)
    local mask = self.input:apply(function(x) return x > 0 and 1 or 0 end)
    return gradOutput:mul(mask)
end

Activations.ReLU = ReLU

-- Sigmoid --
local Sigmoid = setmetatable({}, {__index = Module})
Sigmoid.__index = Sigmoid

function Sigmoid:new()
    local self = setmetatable({}, Sigmoid)
    self:init()
    return self
end

function Sigmoid:forward(input)
    self.output = input:apply(function(x) return 1 / (1 + math.exp(-x)) end)
    return self.output
end

function Sigmoid:backward(gradOutput)
    local deriv = self.output:apply(function(x) return x * (1 - x) end)
    return gradOutput:mul(deriv)
end

Activations.Sigmoid = Sigmoid

-- Tanh --
local Tanh = setmetatable({}, {__index = Module})
Tanh.__index = Tanh

function Tanh:new()
    local self = setmetatable({}, Tanh)
    self:init()
    return self
end

function Tanh:forward(input)
    self.output = input:apply(tanh_func)
    return self.output
end

function Tanh:backward(gradOutput)
    local deriv = self.output:apply(function(x) return 1 - x*x end)
    return gradOutput:mul(deriv)
end

Activations.Tanh = Tanh

-- GELU --
local GELU = setmetatable({}, {__index = Module})
GELU.__index = GELU

function GELU:new()
    local self = setmetatable({}, GELU)
    self:init()
    return self
end

function GELU:forward(input)
    self.input = input
    -- 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    local coeff = math.sqrt(2 / math.pi)

    local x_pow3 = input:pow(3)
    local inner_poly = input + x_pow3:mul(0.044715)

    -- Correct: sqrt(2/pi) * (...)
    local inner = inner_poly:mul(coeff)

    self.tanh_out = inner:apply(tanh_func)

    -- y = 0.5 * x * (1 + tanh(...))
    local y = input:mul(0.5):mul(self.tanh_out + 1.0)
    return y
end

function GELU:backward(gradOutput)
    local x = self.input
    local coeff = math.sqrt(2 / math.pi)

    -- term1 = 0.5 * (1 + tanh(z))
    local term1 = (self.tanh_out + 1.0):mul(0.5)

    -- term2 = 0.5 * x * sech^2(z) * dz/dx
    local sech2 = self.tanh_out:apply(function(t) return 1 - t*t end)

    local x_sq = x:pow(2)
    -- dz/dx = coeff * (1 + 3 * 0.044715 * x^2)
    local dzdx = (x_sq:mul(3 * 0.044715) + 1.0):mul(coeff)

    -- term2 calculation: broadcasting issues?
    -- x: (B, S, D), sech2: (B, S, D), dzdx: (B, S, D)
    -- mul is element-wise.

    local term2 = x:mul(sech2):mul(dzdx):mul(0.5)

    local grad = term1 + term2
    return gradOutput:mul(grad)
end

Activations.GELU = GELU

-- Softmax --
local Softmax = setmetatable({}, {__index = Module})
Softmax.__index = Softmax

function Softmax:new(dim)
    local self = setmetatable({}, Softmax)
    self:init()
    self.dim = dim or 2
    return self
end

function Softmax:forward(input)
    self.output = input:apply(math.exp)
    local sum = self.output:sum(self.dim)
    self.output = self.output:div(sum)
    return self.output
end

function Softmax:backward(gradOutput)
    local dot = gradOutput:mul(self.output):sum(self.dim)
    local sub = gradOutput:sub(dot)
    local gradInput = self.output:mul(sub)
    return gradInput
end

Activations.Softmax = Softmax

return Activations
