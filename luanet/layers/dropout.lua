local Module = require("luanet.module")
local Tensor = require("luanet.tensor")

local Dropout = setmetatable({}, {__index = Module})
Dropout.__index = Dropout

function Dropout:new(p)
    local self = setmetatable({}, Dropout)
    self:init()
    self.p = p or 0.5
    return self
end

function Dropout:forward(input)
    self.input = input

    if (not self.training) or self.p == 0 then
        return input
    end

    -- Create mask
    self.mask = Tensor.zeros(input.shape)
    local size = #input.data
    local scale = 1.0 / (1.0 - self.p)
    local m_data = self.mask.data

    for i = 1, size do
        if math.random() > self.p then
            m_data[i] = scale
        else
            m_data[i] = 0.0
        end
    end

    -- Apply mask: input * mask
    return input:mul(self.mask)
end

function Dropout:backward(gradOutput)
    if (not self.training) or self.p == 0 then
        return gradOutput
    end

    return gradOutput:mul(self.mask)
end

return Dropout
