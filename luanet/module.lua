local Module = {}
Module.__index = Module

function Module:new()
    local o = setmetatable({}, self)
    o:init()
    return o
end

function Module:init()
    self.training = true
    self._modules = {}
    self._parameters = {} -- Table: name -> param_tensor
    self._grads = {}      -- Table: name -> grad_tensor
    self._ordered_params = {} -- List of names to keep order
    self._ordered_modules = {} -- List of names to keep order
end

function Module:forward(input)
    error("Not implemented")
end

function Module:backward(gradOutput)
    error("Not implemented")
end

-- Register a parameter
function Module:register_parameter(name, param, grad)
    if self._parameters[name] then error("Parameter " .. name .. " already exists") end
    self._parameters[name] = param
    self._grads[name] = grad
    table.insert(self._ordered_params, name)
end

-- Register a submodule
function Module:add_module(name, module)
    if self._modules[name] then error("Module " .. name .. " already exists") end
    self._modules[name] = module
    table.insert(self._ordered_modules, name)
end

function Module:parameters()
    local params = {}
    local grads = {}

    -- Own parameters in order
    for _, name in ipairs(self._ordered_params) do
        table.insert(params, self._parameters[name])
        table.insert(grads, self._grads[name])
    end

    -- Submodules in order
    for _, name in ipairs(self._ordered_modules) do
        local sub_params, sub_grads = self._modules[name]:parameters()
        for _, v in ipairs(sub_params) do
            table.insert(params, v)
        end
        for _, v in ipairs(sub_grads) do
            table.insert(grads, v)
        end
    end

    return params, grads
end

function Module:state_dict(prefix, state)
    prefix = prefix or ""
    state = state or {}

    for name, param in pairs(self._parameters) do
        state[prefix .. name] = param
    end

    for name, module in pairs(self._modules) do
        module:state_dict(prefix .. name .. ".", state)
    end

    return state
end

function Module:load_state_dict(state_dict, prefix)
    prefix = prefix or ""

    for name, param in pairs(self._parameters) do
        local key = prefix .. name
        if state_dict[key] then
            -- Copy data
            local saved = state_dict[key]
            if #param.data ~= #saved.data then
                print("Warning: Size mismatch loading state dict for " .. key .. ": expected " .. #param.data .. ", got " .. #saved.data)
            else
                for i = 1, #param.data do
                    param.data[i] = saved.data[i]
                end
            end
        else
            print("Warning: Missing key " .. key .. " in state_dict")
        end
    end

    for name, module in pairs(self._modules) do
        module:load_state_dict(state_dict, prefix .. name .. ".")
    end
end

function Module:zeroGrad()
    local _, grads = self:parameters()
    for _, g in ipairs(grads) do
        if g then g:zero() end
    end
end

function Module:train()
    self.training = true
    for _, name in ipairs(self._ordered_modules) do
        self._modules[name]:train()
    end
end

function Module:eval()
    self.training = false
    for _, name in ipairs(self._ordered_modules) do
        self._modules[name]:eval()
    end
end

function Module:__call(...)
    return self:forward(...)
end

return Module
