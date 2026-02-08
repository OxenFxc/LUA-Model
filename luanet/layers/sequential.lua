local Module = require("luanet.module")

local Sequential = setmetatable({}, {__index = Module})
Sequential.__index = Sequential

function Sequential:new(...)
    local self = setmetatable({}, Sequential)
    self:init()
    local modules = {...}
    for i, m in ipairs(modules) do
        self:add_module(tostring(i), m)
    end
    return self
end

function Sequential:forward(input)
    self.input = input
    local x = input
    for _, name in ipairs(self._ordered_modules) do
        x = self._modules[name]:forward(x)
    end
    return x
end

function Sequential:backward(gradOutput)
    local grad = gradOutput
    -- Iterate in reverse
    for i = #self._ordered_modules, 1, -1 do
        local name = self._ordered_modules[i]
        grad = self._modules[name]:backward(grad)
    end
    return grad
end

return Sequential
