local Tensor = require("luanet.tensor")

local DataLoader = {}
DataLoader.__index = DataLoader

function DataLoader:new(dataset, batch_size, shuffle)
    local self = setmetatable({}, DataLoader)
    self.dataset = dataset
    self.batch_size = batch_size or 1
    self.shuffle = shuffle or false

    local len = dataset:len()
    self.indices = {}
    for i = 1, len do self.indices[i] = i end

    return self
end

function DataLoader:iterator()
    local idx = 1
    local indices = {}
    for i=1, #self.indices do indices[i] = self.indices[i] end

    if self.shuffle then
        -- Fisher-Yates shuffle
        for i = #indices, 2, -1 do
            local j = math.random(i)
            indices[i], indices[j] = indices[j], indices[i]
        end
    end

    return function()
        if idx > #indices then return nil end

        local batch_indices = {}
        local count = 0
        while count < self.batch_size and idx <= #indices do
            table.insert(batch_indices, indices[idx])
            idx = idx + 1
            count = count + 1
        end

        local samples = {}
        for _, i in ipairs(batch_indices) do
            table.insert(samples, self.dataset:getitem(i))
        end

        if #samples == 0 then return nil end

        local first = samples[1]

        if type(first) == "number" then
            return Tensor.new(samples, {#samples})
        elseif getmetatable(first) == Tensor then
            local batch_tensor = Tensor.cat(samples, 1)
            local new_shape = {#samples}
            for i=1, #first.shape do table.insert(new_shape, first.shape[i]) end
            return batch_tensor:view(new_shape)
        elseif type(first) == "table" then
            -- Check if it looks like a tensor but isn't metatable'd (unlikely with our code)
            -- Assume list/dict of tensors
            local collated = {}
            for k, v in pairs(first) do
                local list = {}
                for _, s in ipairs(samples) do
                    table.insert(list, s[k])
                end

                local v_first = list[1]
                if type(v_first) == "number" then
                    collated[k] = Tensor.new(list, {#list})
                elseif getmetatable(v_first) == Tensor then
                     local batch_t = Tensor.cat(list, 1)
                     local new_shape = {#list}
                     for i=1, #v_first.shape do table.insert(new_shape, v_first.shape[i]) end
                     collated[k] = batch_t:view(new_shape)
                else
                    collated[k] = list
                end
            end
            return collated
        else
            return samples
        end
    end
end

return DataLoader
